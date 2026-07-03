from __future__ import annotations
import asyncio
import shutil
import tempfile
from pathlib import Path
from collections import defaultdict

from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile, status, Request, Form
from fastapi.responses import FileResponse

from .audio import ensure_mono_16k, probe_duration, schedule_cleanup
from .auth import require_api_key
from .model import _to_builtin
from .schemas import TranscriptionResponse
from .config import MAX_AUDIO_DURATION, MAX_UPLOAD_BYTES, PROCESSING_TIMEOUT, SERVE_DASHBOARD, logger

from parakeet_service.model import reset_fast_path
from parakeet_service.chunker import vad_chunk_lowmem


router = APIRouter(tags=["speech"])


async def _stream_to_file(file: UploadFile, dest: Path) -> None:
    """Copy an upload to *dest*, aborting if it exceeds MAX_UPLOAD_BYTES.

    Raises HTTP 413 when the limit is exceeded so an attacker cannot exhaust
    disk by streaming an unbounded body.
    """
    total = 0
    with open(dest, "wb") as f:
        while True:
            chunk = await file.read(8192)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_UPLOAD_BYTES:
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=f"Upload exceeds maximum allowed size of {MAX_UPLOAD_BYTES} bytes",
                )
            f.write(chunk)


_STATIC_DIR = Path(__file__).parent / "static"


@router.get("/", include_in_schema=False)
def dashboard():
    """Serve the live-transcription demo dashboard (public, no auth).

    Returns 404 when the dashboard is disabled (SERVE_DASHBOARD=0).
    """
    if not SERVE_DASHBOARD:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND)
    return FileResponse(_STATIC_DIR / "index.html")


@router.get("/healthz", summary="Liveness/readiness probe")
def health():
    return {"status": "ok"}


@router.post(
    "/transcribe",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
)
@router.post(
    "/audio/transcriptions",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
)
@router.post(
    "/v1/audio/transcriptions",
    response_model=TranscriptionResponse,
    summary="Transcribe an audio file",
)
async def transcribe_audio(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., media_type="audio/*"),
    include_timestamps: bool = Form(
        False, description="Return char/word/segment offsets",
    ),
    should_chunk: bool = Form(True,
        description="If true (default), split long audio into "
                    "~60s VAD-aligned chunks for batching"),
    _auth: None = Depends(require_api_key),
):
    # Create temp file with appropriate extension
    suffix = Path(file.filename or "").suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp_path = Path(tmp.name)
    
    # Stream upload directly to processing with cancellation handling
    try:
        # Use FFmpeg for MP3 files to fix header issues
        # Create temp MP3 file if needed
        mp3_tmp_path = None
        if suffix.lower() == ".mp3":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as mp3_tmp:
                mp3_tmp_path = Path(mp3_tmp.name)
            # Write entire MP3 to temporary file (size-capped)
            await _stream_to_file(file, mp3_tmp_path)

            # Update FFmpeg command to read from file
            ffmpeg_cmd = [
                "ffmpeg", "-v", "error", "-nostdin", "-y",
                "-i", str(mp3_tmp_path),
                "-acodec", "pcm_s16le", "-ac", "1", "-ar", "16000",
                "-f", "wav", str(tmp_path)
            ]
        else:
            ffmpeg_cmd = None
            # For non-MP3, stream directly to file (size-capped)
            await _stream_to_file(file, tmp_path)
        
        # Run FFmpeg if processing MP3
        if ffmpeg_cmd:
            process = await asyncio.create_subprocess_exec(
                *ffmpeg_cmd,
                stdout=asyncio.subprocess.DEVNULL,  # We don't need stdout
                stderr=asyncio.subprocess.PIPE
            )

            # Read stderr and wait for completion under a hard timeout so a
            # crafted input cannot hang a worker indefinitely.
            try:
                stderr_bytes = await asyncio.wait_for(
                    process.communicate(), timeout=PROCESSING_TIMEOUT
                )
                return_code = process.returncode
            except asyncio.TimeoutError:
                process.kill()
                await process.wait()
                logger.error("FFmpeg timed out after %ss", PROCESSING_TIMEOUT)
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail="Audio processing timed out",
                )

            stderr_str = (stderr_bytes[1] or b"").decode(errors="replace").strip()

            if return_code != 0:
                # Log full detail server-side; return a generic message to the client.
                logger.error("FFmpeg failed (rc=%s): %s", return_code, stderr_str)
                raise HTTPException(
                    status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                    detail="Invalid or unsupported audio format",
                )
            else:
                logger.debug("FFmpeg completed successfully")
    except HTTPException:
        # Rejected upload (oversized / bad format / timeout): drop temp files so
        # a stream of rejected requests cannot accumulate disk usage.
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise
    except asyncio.CancelledError:
        # Clean up temporary files if processing was cancelled
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise
    except BrokenPipeError:
        logger.error("FFmpeg process terminated unexpectedly")
        if tmp_path.exists():
            tmp_path.unlink()
        if mp3_tmp_path and mp3_tmp_path.exists():
            mp3_tmp_path.unlink()
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Audio processing failed due to FFmpeg crash"
        )
    finally:
        await file.close()

    # Process audio to ensure mono 16kHz. On any rejection past this point,
    # BackgroundTasks do NOT run (FastAPI skips them for error responses), so
    # temp files must be unlinked directly.
    try:
        original, to_model = ensure_mono_16k(tmp_path)

        # Reject audio exceeding the configured ceiling (bounds GPU work / DoS).
        duration = probe_duration(to_model)
        if MAX_AUDIO_DURATION and duration > MAX_AUDIO_DURATION:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=f"Audio duration {duration:.1f}s exceeds maximum of {MAX_AUDIO_DURATION}s",
            )
    except HTTPException:
        for p in {tmp_path, locals().get("to_model")}:
            if isinstance(p, Path) and p.exists():
                p.unlink(missing_ok=True)
        raise

    if should_chunk:
        # Use low-memory chunker for non-streaming requests. vad_chunk_lowmem
        # returns a list of Paths (no per-chunk offset), so normalise each item
        # to a (path, offset) pair; fall back to the whole file if it yields none.
        raw_chunks = vad_chunk_lowmem(to_model) or [to_model]
        chunk_data = [c if isinstance(c, tuple) else (c, 0.0) for c in raw_chunks]
    else:
        chunk_data = [(to_model, 0.0)]

    chunk_paths = [path for path, _ in chunk_data]
    chunk_offsets = [offset for _, offset in chunk_data]

    logger.info("transcribe(): sending %d chunks to ASR", len(chunk_paths))

    # Clean up all temporary files
    cleanup_files = [original, to_model] + chunk_paths
    if mp3_tmp_path:
        cleanup_files.append(mp3_tmp_path)
    schedule_cleanup(background_tasks, *cleanup_files)

    # 2 – run ASR
    model = request.app.state.asr_model

    try:
        outs = model.transcribe(
            [str(p) for p in chunk_paths],
            batch_size=2,
            timestamps=include_timestamps,
        )
        if (
          not include_timestamps                     # switch back to model fast-path if timestamps turned off
          and getattr(model.cfg.decoding, "compute_timestamps", False)
        ):
          reset_fast_path(model)                    
    except RuntimeError as exc:
        # Log the underlying error server-side; do not leak internals to clients.
        logger.exception("ASR failed")
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Failed to transcribe audio") from exc

    if isinstance(outs, tuple):
      outs = outs[0]
    texts = []
    ts_agg = [] if include_timestamps else None
    merged = defaultdict(list)

    for idx, h in enumerate(outs):
        texts.append(getattr(h, "text", str(h)))
        if include_timestamps:
            offset = chunk_offsets[idx]
            for k, v in _to_builtin(getattr(h, "timestamp", {})).items():
                # Adjust timestamps by adding the chunk offset
                if isinstance(v, list) and len(v) > 0:
                    adjusted = []
                    for item in v:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            # Assume format is [start, end, ...] or (start, end, ...)
                            adjusted_item = list(item)
                            adjusted_item[0] = item[0] + offset
                            adjusted_item[1] = item[1] + offset
                            adjusted.append(adjusted_item)
                        elif isinstance(item, dict):
                            # Handle dict format with 'start' and 'end' keys
                            adjusted_item = item.copy()
                            if 'start' in adjusted_item:
                                adjusted_item['start'] += offset
                            if 'end' in adjusted_item:
                                adjusted_item['end'] += offset
                            adjusted.append(adjusted_item)
                        else:
                            adjusted.append(item)
                    merged[k].extend(adjusted)
                else:
                    merged[k].extend(v)

    merged_text = " ".join(texts).strip()
    timestamps  = dict(merged) if include_timestamps else None

    return TranscriptionResponse(text=merged_text, timestamps=timestamps)