#!/usr/bin/env python3
"""
Realtime-Venus-Audio full-duplex server.

Loads the model once at startup, then exposes a WebSocket endpoint that
mirrors the chunk-by-chunk streaming_prefill()/streaming_generate() loop
from the official audio_duplex_chat.py example, but driven by a live
browser mic stream instead of a pre-recorded file.

Protocol (binary WebSocket frames both ways):
  Client -> Server: raw PCM16LE mono audio, 16000 Hz, arbitrary chunk size.
                     The server buffers and re-chunks to duplex.CHUNK_MS.
  Server -> Client: JSON text frames for state/text events:
                     {"type": "state", "state": "listen"|"speak"}
                     {"type": "text", "text": "..."}
                     {"type": "error", "message": "..."}
                     PCM16LE mono 24000 Hz binary frames for generated speech.
"""

import argparse
import asyncio
import io
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import AutoModel, set_seed

INPUT_SAMPLE_RATE = 16000
TTS_SAMPLE_RATE = 24000
DEFAULT_SYSTEM_PROMPT = "Streaming Omni Conversation."

app = FastAPI()

# Populated by load_model() at startup (see main()).
STATE = {"model": None, "model_path": None}


def load_model(model_dir: Path):
    set_seed(42)
    print(f"Loading model from {model_dir} ...", flush=True)
    model = AutoModel.from_pretrained(
        str(model_dir),
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation="sdpa",
        torch_dtype=torch.bfloat16,
        init_vision=False,
        init_audio=True,
        init_tts=True,
    )
    model.eval().cuda()
    print("Model loaded.", flush=True)
    return model


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    return np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0


def float32_to_pcm16_bytes(samples: np.ndarray) -> bytes:
    samples = np.clip(samples, -1.0, 1.0)
    return (samples * 32767.0).astype(np.int16).tobytes()


def audio_samples(waveform) -> np.ndarray:
    """Mirrors audio_samples() in audio_duplex_chat.py."""
    samples = np.asarray(waveform, dtype=np.float32).squeeze()
    if samples.ndim == 1:
        return samples[:, None]
    if samples.ndim != 2:
        raise ValueError(f"Unexpected generated audio shape: {samples.shape}")
    return samples.T if samples.shape[0] <= 8 else samples


@app.get("/")
async def index():
    return FileResponse(Path(__file__).parent / "static" / "index.html")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    model = STATE["model"]
    if model is None:
        await ws.send_text(json.dumps({"type": "error", "message": "Model not loaded on server."}))
        await ws.close()
        return

    duplex = model.as_duplex(generate_audio=True)
    duplex.prepare(prefix_system_prompt=DEFAULT_SYSTEM_PROMPT)
    chunk_samples = round(duplex.CHUNK_MS * duplex.SAMPLE_RATE / 1000)

    # Rolling buffer of incoming float32 audio not yet consumed as a full chunk.
    pending = np.zeros((0,), dtype=np.float32)
    loop = asyncio.get_event_loop()

    async def run_step(chunk: np.ndarray, text_in: str | None):
        """Run one prefill+generate step off the event loop (it's blocking CUDA work)."""

        def _step():
            with torch.inference_mode():
                prefill = duplex.streaming_prefill(
                    audio_waveform=chunk,
                    text_list=[text_in] if text_in else None,
                )
                if not prefill.get("success"):
                    reason = prefill.get("reason") or "unknown prefill error"
                    raise RuntimeError(reason)
                result = duplex.streaming_generate(
                    max_new_speak_tokens_per_chunk=20,
                    decode_mode="sampling",
                    temperature=0.7,
                    top_k=20,
                    top_p=0.8,
                    listen_prob_scale=1.0,
                )
                return result

        return await loop.run_in_executor(None, _step)

    try:
        while True:
            msg = await ws.receive()
            if "bytes" in msg and msg["bytes"] is not None:
                incoming = pcm16_bytes_to_float32(msg["bytes"])
                pending = np.concatenate([pending, incoming])

                while len(pending) >= chunk_samples:
                    chunk = pending[:chunk_samples]
                    pending = pending[chunk_samples:]

                    result = await run_step(chunk, None)
                    await emit_result(ws, result)

            elif "text" in msg and msg["text"] is not None:
                payload = json.loads(msg["text"])
                if payload.get("type") == "inject_text":
                    # Pad any leftover partial chunk with silence and send the
                    # text with it, same as the CLI example does on the last chunk.
                    chunk = np.pad(pending, (0, max(0, chunk_samples - len(pending))))
                    chunk = chunk[:chunk_samples]
                    pending = np.zeros((0,), dtype=np.float32)
                    result = await run_step(chunk, payload.get("text", ""))
                    await emit_result(ws, result)
                elif payload.get("type") == "flush":
                    # End of session tail silence, matching INPUT_TAIL_SECONDS behavior.
                    tail = np.zeros((chunk_samples,), dtype=np.float32)
                    for _ in range(10):
                        result = await run_step(tail, None)
                        await emit_result(ws, result)

    except WebSocketDisconnect:
        pass
    except Exception as exc:  # noqa: BLE001
        try:
            await ws.send_text(json.dumps({"type": "error", "message": str(exc)}))
        except Exception:
            pass
    finally:
        try:
            model.as_simplex()
        except Exception:
            pass


async def emit_result(ws: WebSocket, result: dict):
    state = "listen" if result.get("is_listen") else "speak"
    await ws.send_text(json.dumps({"type": "state", "state": state}))

    text = result.get("text", "")
    if text:
        await ws.send_text(json.dumps({"type": "text", "text": text}))

    waveform = result.get("audio_waveform")
    if waveform is not None and not result.get("is_listen"):
        samples = audio_samples(waveform)  # (n, channels)
        mono = samples.mean(axis=1)
        await ws.send_bytes(float32_to_pcm16_bytes(mono))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True, help="Local Realtime-Venus-Audio checkpoint dir")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_path).expanduser().resolve()
    STATE["model"] = load_model(model_dir)
    STATE["model_path"] = str(model_dir)

    import uvicorn

    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
