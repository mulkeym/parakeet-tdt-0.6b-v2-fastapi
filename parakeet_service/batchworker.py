import asyncio
import logging
import time
import torch
import numpy as np
from typing import List, Tuple

from parakeet_service.config import BATCH_SIZE, BATCH_WINDOW_MS

logger = logging.getLogger("batcher")
logger.setLevel(logging.DEBUG)

# -------- shared state -------------------------------------------------------
# Queue items are (chunk_id, audio_array) tuples
transcription_queue: asyncio.Queue[Tuple[str, np.ndarray]] = asyncio.Queue(maxsize=64)
condition = asyncio.Condition()          # wakes websocket consumers
results: dict[str, dict] = {}             # chunk_id -> {"text": ..., "confidence": ..., "words": ...}


# -------- sync inference (runs in thread pool) -------------------------------
def _run_inference(model, audio_arrays: List[np.ndarray], batch_size: int):
    """Run NeMo transcribe on a list of numpy arrays. Called via asyncio.to_thread()."""
    with torch.inference_mode():
        return model.transcribe(
            audio_arrays, batch_size=batch_size,
            return_hypotheses=True, verbose=False,
        )


def _extract_result(hypothesis) -> dict:
    """Extract text, confidence, and per-word scores from a Hypothesis object."""
    text = getattr(hypothesis, "text", "") or ""
    word_conf = getattr(hypothesis, "word_confidence", None)

    if word_conf:
        word_scores = [round(float(c), 4) for c in word_conf]
        confidence = round(sum(word_scores) / len(word_scores), 4)
    else:
        word_scores = None
        confidence = None

    return {"text": text, "confidence": confidence, "words": word_scores}


# -------- main worker --------------------------------------------------------
async def batch_worker(model, batch_ms: float = None, max_batch: int = None):
    """Forever drain `transcription_queue` -> ASR -> `results`.

    Inference runs in a thread pool so the asyncio event loop stays responsive.
    """
    if batch_ms is None:
        batch_ms = float(BATCH_WINDOW_MS)
    if max_batch is None:
        max_batch = BATCH_SIZE

    logger.info("worker started (batch <=%d, window %.0f ms, model id=%s)",
                max_batch, batch_ms, id(model))

    while True:
        # Block until at least one item arrives
        first_item = await transcription_queue.get()
        batch: List[Tuple[str, np.ndarray]] = [first_item]

        # ---------- micro-batch gathering with timeout ----------
        deadline = time.monotonic() + batch_ms / 1000
        while len(batch) < max_batch:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                nxt = await asyncio.wait_for(transcription_queue.get(), remaining)
                batch.append(nxt)
            except asyncio.TimeoutError:
                break

        chunk_ids = [item[0] for item in batch]
        audio_arrays = [item[1] for item in batch]

        logger.debug("processing %d-item batch", len(batch))

        # ---------- inference (in thread pool) ----------
        try:
            outs = await asyncio.to_thread(
                _run_inference, model, audio_arrays, len(audio_arrays)
            )
        except Exception as exc:
            logger.exception("ASR failed: %s", exc)
            for cid, _ in batch:
                results[cid] = {"text": "", "confidence": None, "words": None}
                transcription_queue.task_done()
            async with condition:
                condition.notify_all()
            continue

        # ---------- store results & notify ----------
        for cid, h in zip(chunk_ids, outs):
            results[cid] = _extract_result(h)
            transcription_queue.task_done()
        async with condition:
            condition.notify_all()
