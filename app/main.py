"""FastAPI server: background stream loop + REST + WebSocket broadcast."""
import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse

from .engine import DemandEngine

TICK_SECONDS = float(os.getenv("TICK_SECONDS", "1.5"))  # real seconds per simulated hour
STATIC = Path(__file__).parent.parent / "static"

engine = DemandEngine()
clients: set[WebSocket] = set()
state = {"snapshot": None}


async def stream_loop():
    while True:
        state["snapshot"] = engine.tick()
        dead = []
        for ws in list(clients):
            try:
                await ws.send_json(state["snapshot"])
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)
        await asyncio.sleep(TICK_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    engine.warmup()
    state["snapshot"] = engine.snapshot()
    task = asyncio.create_task(stream_loop())
    yield
    task.cancel()


app = FastAPI(title="DemandPulse", lifespan=lifespan)


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/healthz")
def healthz():
    return {"status": "ok", "hour": engine.hour}


@app.get("/api/state")
def api_state():
    return state["snapshot"]


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    clients.add(ws)
    try:
        await ws.send_json(state["snapshot"])
        while True:
            await ws.receive_text()  # keep connection open
    except WebSocketDisconnect:
        clients.discard(ws)
