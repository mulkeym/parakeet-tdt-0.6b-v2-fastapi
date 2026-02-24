from __future__ import annotations
import itertools
import numpy as np
from typing import List, Tuple
from torch.hub import load as torch_hub_load

vad_model, vad_utils = torch_hub_load("snakers4/silero-vad", "silero_vad")
(_, _, _, VADIterator, _) = vad_utils

# TODO: Update to read from .env
SAMPLE_RATE              = 16_000         # model is trained for 16 kHz
WINDOW_SAMPLES           = 512            # 32 ms frame
THRESHOLD                = 0.60           # voice prob >= 0.60 -> speech
MIN_SILENCE_MS           = 250            # flush after >=250 ms quiet
SPEECH_PAD_MS            = 120            # keep 120 ms context before/after
MAX_SPEECH_MS            = 8_000          # hard stop at 8 s

# Global chunk ID counter (unique across all sessions within one process)
_chunk_counter = itertools.count()

# Type alias for a VAD result: (unique_id, float32_audio_array)
ChunkResult = Tuple[str, np.ndarray]


class StreamingVAD:
    """
    Feed successive 20-40 ms PCM frames (16 kHz, int16 mono).
    Emits (chunk_id, np.ndarray) tuples when a full utterance is detected.
    The array is float32 in [-1, 1], ready for NeMo's model.transcribe().
    """

    def __init__(self):
        self.vad = VADIterator(
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=THRESHOLD,
            min_silence_duration_ms=MIN_SILENCE_MS,
            speech_pad_ms=SPEECH_PAD_MS,
        )
        self._f32_buffer: list[np.ndarray] = []  # accumulate float32 windows
        self.speech_ms = 0

    def _flush(self) -> List[ChunkResult]:
        if not self._f32_buffer:
            return []
        audio = np.concatenate(self._f32_buffer)
        chunk_id = f"chunk-{next(_chunk_counter)}"
        self._f32_buffer.clear()
        self.speech_ms = 0
        self.vad.reset_states()
        return [(chunk_id, audio)]

    def feed(self, frame_bytes: bytes) -> List[ChunkResult]:
        out: List[ChunkResult] = []

        pcm_f32 = np.frombuffer(frame_bytes, np.int16).astype("float32") / 32768
        for start in range(0, len(pcm_f32), WINDOW_SAMPLES):
            window = pcm_f32[start:start + WINDOW_SAMPLES]
            if len(window) < WINDOW_SAMPLES:
                break  # wait for full 32 ms window

            voice_event = self.vad(window, return_seconds=False)
            self._f32_buffer.append(window.copy())
            self.speech_ms += 32

            # Flush on trailing-silence event or max-length guard
            if voice_event and voice_event.get("end"):
                out.extend(self._flush())
            elif self.speech_ms >= MAX_SPEECH_MS:
                out.extend(self._flush())

        return out
