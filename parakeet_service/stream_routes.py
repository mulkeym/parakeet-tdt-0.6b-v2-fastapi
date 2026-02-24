import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from parakeet_service.streaming_vad import StreamingVAD
from parakeet_service.batchworker import transcription_queue, condition, results
import asyncio

logger = logging.getLogger("stream_routes")

router = APIRouter()

# Monotonic counter for WebSocket session IDs (process-wide)
_ws_counter = 0


@router.websocket("/ws")
async def ws_asr(ws: WebSocket):
    global _ws_counter
    _ws_counter += 1
    ws_id = _ws_counter

    await ws.accept()
    vad = StreamingVAD()
    t_start = time.monotonic()
    frame_count = 0

    logger.info("[WS#%d] connected", ws_id)

    # Track which chunk IDs belong to THIS session
    pending_ids: set[str] = set()

    async def producer():
        """Push VAD chunks into the global transcription queue."""
        nonlocal frame_count
        try:
            while True:
                frame = await ws.receive_bytes()
                frame_count += 1
                for chunk_id, audio_array in vad.feed(frame):
                    pending_ids.add(chunk_id)
                    elapsed = time.monotonic() - t_start
                    logger.info(
                        "[WS#%d] VAD chunk %s queued (%.2f s audio) "
                        "after %d frames @ %.3fs",
                        ws_id, chunk_id, len(audio_array) / 16000,
                        frame_count, elapsed,
                    )
                    await transcription_queue.put((chunk_id, audio_array))
                    await ws.send_json({"status": "queued"})
        except WebSocketDisconnect:
            elapsed = time.monotonic() - t_start
            logger.info("[WS#%d] disconnected after %d frames (%.3fs)",
                        ws_id, frame_count, elapsed)

    async def consumer():
        """Stream results back as soon as they're ready."""
        try:
            while True:
                async with condition:
                    await condition.wait()
                flushed = []
                for cid in list(pending_ids):
                    if cid in results:
                        result = results.pop(cid)
                        elapsed = time.monotonic() - t_start
                        text_preview = (result.get("text", "") or "")[:80]
                        logger.info(
                            "[WS#%d] sending result %s @ %.3fs: %s",
                            ws_id, cid, elapsed, text_preview,
                        )
                        await ws.send_json(result)
                        flushed.append(cid)
                for cid in flushed:
                    pending_ids.discard(cid)
        except asyncio.CancelledError:
            pass

    consumer_task = asyncio.create_task(consumer())
    try:
        await producer()
    finally:
        await asyncio.sleep(0.5)
        consumer_task.cancel()
        try:
            await consumer_task
        except asyncio.CancelledError:
            pass
        dropped = len(pending_ids)
        for cid in list(pending_ids):
            results.pop(cid, None)
        pending_ids.clear()
        # Release VAD buffers and per-session references
        vad._f32_buffer.clear()
        elapsed = time.monotonic() - t_start
        logger.info("[WS#%d] cleanup done (%.3fs total, %d pending dropped)",
                    ws_id, elapsed, dropped)
