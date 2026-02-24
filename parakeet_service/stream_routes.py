from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from parakeet_service.streaming_vad import StreamingVAD
from parakeet_service.batchworker import transcription_queue, condition, results
import asyncio

router = APIRouter()


@router.websocket("/ws")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    vad = StreamingVAD()

    # Track which chunk IDs belong to THIS session
    pending_ids: set[str] = set()

    async def producer():
        """Push VAD chunks into the global transcription queue."""
        try:
            while True:
                frame = await ws.receive_bytes()
                for chunk_id, audio_array in vad.feed(frame):
                    pending_ids.add(chunk_id)
                    await transcription_queue.put((chunk_id, audio_array))
                    await ws.send_json({"status": "queued"})
        except WebSocketDisconnect:
            pass

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
        for cid in list(pending_ids):
            results.pop(cid, None)
