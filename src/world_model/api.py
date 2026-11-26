"""FastAPI WebSocket interface for interactive world-model inference."""

from __future__ import annotations

import base64
import io
import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import gymnasium as gym
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image
from starlette.concurrency import run_in_threadpool

from world_model.engine import WorldEngine


EngineFactory = Callable[[], WorldEngine]


def encode_frame(frame: np.ndarray) -> str:
    """Encode an RGB uint8 image as a Base64 PNG string."""
    image_buffer = io.BytesIO()
    Image.fromarray(frame).save(image_buffer, format="PNG")
    return base64.b64encode(image_buffer.getvalue()).decode("ascii")


def resize_reference(frame: np.ndarray) -> np.ndarray:
    """Resize a Gymnasium RGB observation to the model's 64x64 display size."""
    return np.asarray(
        Image.fromarray(frame).resize((64, 64), Image.Resampling.BILINEAR),
        dtype=np.uint8,
    )


def create_app(engine_factory: EngineFactory = WorldEngine) -> FastAPI:
    """Create the API application, with a separate engine per WebSocket client."""
    app = FastAPI(title="World Model Demo API", version="0.1.0")
    static_dir = Path(__file__).parent / "static"

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(static_dir / "index.html")

    @app.get("/live")
    async def live() -> dict[str, str]:
        """Report that the API process is running."""
        return {"status": "ok"}

    @app.get("/health")
    async def health() -> JSONResponse:
        """Report whether model assets are present for inference."""
        readiness = WorldEngine.readiness()
        status_code = 200 if readiness["status"] == "ready" else 503
        return JSONResponse(readiness, status_code=status_code)

    @app.websocket("/ws/dream")
    async def dream_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            engine = await run_in_threadpool(engine_factory)
        except Exception as exc:
            await websocket.send_json({"type": "error", "message": str(exc)})
            await websocket.close(code=1011)
            return

        comparison_env = None
        try:
            while True:
                raw_message = await websocket.receive_text()
                try:
                    message = json.loads(raw_message)
                    if not isinstance(message, dict):
                        raise ValueError("Message must be a JSON object")
                    message_type = message.get("type")

                    if message_type == "reset":
                        frame = await run_in_threadpool(
                            engine.reset,
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
                        if comparison_env is not None:
                            observation, _ = await run_in_threadpool(
                                comparison_env.reset, seed=message.get("seed")
                            )
                            response["reference_image"] = encode_frame(
                                resize_reference(observation)
                            )
                    elif message_type == "seed_options":
                        response = {
                            "type": "seed_options",
                            "episodes": await run_in_threadpool(engine.seed_options),
                        }
                    elif message_type == "action":
                        if "temperature" in message:
                            engine.set_temperature(float(message["temperature"]))
                        action = np.asarray(message.get("action"), dtype=np.float32)
                        if action.shape != (3,) or not np.isfinite(action).all():
                            raise ValueError("action must contain three finite values")
                        action = np.clip(action, [-1.0, 0.0, 0.0], [1.0, 1.0, 1.0])
                        frame = await run_in_threadpool(engine.step, action)
                        response = {
                            "type": "frame",
                            "image": encode_frame(frame),
                            "latent": engine.latent.tolist(),
                            "temperature": engine.temperature,
                        }
                        if comparison_env is not None:
                            observation, _, terminated, truncated, _ = await run_in_threadpool(
                                comparison_env.step, action
                            )
                            reference_restarted = terminated or truncated
                            if reference_restarted:
                                observation, _ = await run_in_threadpool(comparison_env.reset)
                            response["reference_image"] = encode_frame(
                                resize_reference(observation)
                            )
                            response["reference_restarted"] = reference_restarted
                    elif message_type == "compare":
                        enabled = bool(message.get("enabled"))
                        if enabled and comparison_env is None:
                            comparison_env = gym.make("CarRacing-v3", render_mode="rgb_array")
                            observation, _ = await run_in_threadpool(
                                comparison_env.reset, seed=message.get("seed")
                            )
                            response = {
                                "type": "comparison",
                                "enabled": True,
                                "reference_image": encode_frame(resize_reference(observation)),
                                "note": "The Gymnasium reference starts from an independent state.",
                            }
                        elif not enabled and comparison_env is not None:
                            await run_in_threadpool(comparison_env.close)
                            comparison_env = None
                            response = {"type": "comparison", "enabled": False}
                        else:
                            response = {"type": "comparison", "enabled": enabled}
                    elif message_type == "temperature":
                        engine.set_temperature(float(message.get("value")))
                        response = {
                            "type": "temperature",
                            "value": engine.temperature,
                        }
                    elif message_type == "latent":
                        frame = await run_in_threadpool(
                            engine.set_latent, message.get("values")
                        )
                        response = {
                            "type": "latent",
                            "image": encode_frame(frame),
                            "latent": engine.latent.tolist(),
                        }
                    else:
                        raise ValueError(
                            "type must be seed_options, reset, action, temperature, compare, or latent"
                        )
                    await websocket.send_json(response)
                except Exception as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
        except WebSocketDisconnect:
            pass
        finally:
            if comparison_env is not None:
                await run_in_threadpool(comparison_env.close)

    app.mount("/static", StaticFiles(directory=static_dir), name="static")
    return app


app = create_app()
