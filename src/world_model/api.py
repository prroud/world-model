"""FastAPI WebSocket interface for interactive world-model inference."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Callable

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from PIL import Image

from world_model.engine import WorldEngine


EngineFactory = Callable[[], WorldEngine]


def encode_frame(frame: np.ndarray) -> str:
    """Encode an RGB uint8 image as a Base64 PNG string."""
    image_buffer = io.BytesIO()
    Image.fromarray(frame).save(image_buffer, format="PNG")
    return base64.b64encode(image_buffer.getvalue()).decode("ascii")


def create_app(engine_factory: EngineFactory = WorldEngine) -> FastAPI:
    """Create the API application, with a separate engine per WebSocket client."""
    app = FastAPI(title="World Model Demo API", version="0.1.0")

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.websocket("/ws/dream")
    async def dream_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            engine = engine_factory()
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1011)
            return

        try:
            while True:
                raw_message = await websocket.receive_text()
                try:
                    message = json.loads(raw_message)
                    if not isinstance(message, dict):
                        raise ValueError("Message must be a JSON object")
                    message_type = message.get("type")

                    if message_type == "reset":
                        frame = engine.reset(
                            seed=message.get("seed"),
                            episode_index=message.get("episode_index"),
                            frame_index=message.get("frame_index"),
                        )
                        response = {
                            "type": "frame",
                            "image": encode_frame(frame),
                            "latent": engine.latent.tolist(),
                            "temperature": engine.temperature,
                        }
                    elif message_type == "action":
                        if "temperature" in message:
                            engine.set_temperature(float(message["temperature"]))
                        frame = engine.step(message.get("action"))
                        response = {
                            "type": "frame",
                            "image": encode_frame(frame),
                            "latent": engine.latent.tolist(),
                            "temperature": engine.temperature,
                        }
                    elif message_type == "temperature":
                        engine.set_temperature(float(message.get("value")))
                        response = {
                            "type": "temperature",
                            "value": engine.temperature,
                        }
                    else:
                        raise ValueError("type must be reset, action, or temperature")
                    await websocket.send_json(response)
                except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
        except WebSocketDisconnect:
            pass

    return app


app = create_app()
