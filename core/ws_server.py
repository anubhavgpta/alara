"""
alara/core/ws_server.py

WebSocket bridge between ALARA Python backend and Electron UI.
"""

from __future__ import annotations

import asyncio
import json
import threading
from typing import Any

from loguru import logger
import websockets


_ACTIVE_SERVER: "AlaraWSServer | None" = None


def broadcast(message: dict[str, Any]) -> None:
    """Broadcast a message to all connected UI clients, if server is running."""
    global _ACTIVE_SERVER
    if _ACTIVE_SERVER is not None:
        _ACTIVE_SERVER.broadcast(message)


class AlaraWSServer:
    def __init__(self, intent_engine, executor, transcriber, recorder, host: str = "localhost", port: int = 8765):
        self.intent_engine = intent_engine
        self.executor = executor
        self.transcriber = transcriber
        self.recorder = recorder
        self.host = host
        self.port = port

        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server = None
        self._clients: set[Any] = set()
        self._recording_task: asyncio.Task | None = None

    def start_background(self) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._thread = threading.Thread(target=self._run, daemon=True, name="alara-ws-server")
        self._thread.start()
        logger.success(f"WebSocket server starting on ws://{self.host}:{self.port}")

    def _run(self) -> None:
        global _ACTIVE_SERVER
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        async def runner():
            self._server = await websockets.serve(self._client_handler, self.host, self.port)

        self._loop.run_until_complete(runner())
        _ACTIVE_SERVER = self
        self._loop.run_forever()

    async def _client_handler(self, websocket):
        self._clients.add(websocket)
        logger.info(f"UI client connected ({len(self._clients)} total)")
        try:
            async for raw in websocket:
                await self._handle_message(websocket, raw)
        except websockets.ConnectionClosed:
            pass
        finally:
            self._clients.discard(websocket)
            logger.info(f"UI client disconnected ({len(self._clients)} total)")

    async def _handle_message(self, websocket, raw: str) -> None:
        try:
            payload = json.loads(raw)
        except Exception:
            return

        msg_type = payload.get("type", "")

        if msg_type == "text_command":
            text = str(payload.get("text", "")).strip()
            if not text:
                return
            await self._send(websocket, {"type": "state", "state": "processing"})
            result_payload = await asyncio.to_thread(self._run_text_command, text)
            await self._send(websocket, result_payload)
            await self._send(websocket, {"type": "state", "state": "idle"})
            return

        if msg_type == "start_listening":
            await self._send(websocket, {"type": "state", "state": "listening"})
            if self._recording_task and not self._recording_task.done():
                return
            self._recording_task = asyncio.create_task(self._record_then_process(websocket))
            return

        if msg_type == "stop_listening":
            self.recorder.stop()
            await self._send(websocket, {"type": "state", "state": "idle"})
            return

    def _run_text_command(self, text: str) -> dict[str, Any]:
        action = self.intent_engine.parse(text)
        result = self.executor.execute(action)
        return {
            "type": "result",
            "action": action.action,
            "success": result.success,
            "message": result.message,
        }

    async def _record_then_process(self, websocket) -> None:
        wav_bytes = await asyncio.to_thread(self.recorder.record)
        if not wav_bytes:
            await self._send(websocket, {"type": "state", "state": "idle"})
            return

        await self._send(websocket, {"type": "state", "state": "processing"})
        transcription = await asyncio.to_thread(self.transcriber.transcribe, wav_bytes)
        if not transcription:
            await self._send(
                websocket,
                {
                    "type": "result",
                    "action": "unknown",
                    "success": False,
                    "message": "Could not transcribe audio.",
                },
            )
            await self._send(websocket, {"type": "state", "state": "idle"})
            return

        result_payload = await asyncio.to_thread(self._run_text_command, transcription)
        await self._send(websocket, result_payload)
        await self._send(websocket, {"type": "state", "state": "idle"})

    async def _send(self, websocket, message: dict[str, Any]) -> None:
        try:
            await websocket.send(json.dumps(message))
        except Exception:
            pass

    async def _broadcast_async(self, message: dict[str, Any]) -> None:
        if not self._clients:
            return

        encoded = json.dumps(message)
        dead = []
        for ws in self._clients:
            try:
                await ws.send(encoded)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    def broadcast(self, message: dict[str, Any]) -> None:
        if not self._loop:
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(message), self._loop)
