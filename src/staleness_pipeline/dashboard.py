"""FastAPI live dashboard: consumes the simulated Kafka feed, maintains
rolling state via live_store.LiveSeriesStore, and pushes updates to
connected browsers over WebSocket.

Deployment note: this app has NO authentication built in on purpose —
access control belongs one layer up (nginx), the same "auth happens in
front, not in the app" pattern already used for MLflow on host241
(mlflow_proxy.py does the Hub-login check there; nginx's auth_basic will
do the equivalent job here — see the deployment runbook).

Everything environment-specific is read from env vars with local-dev
defaults, so this exact file runs unchanged both on a laptop against a
local Kafka broker and later on host241 against the same broker moved
there — only the systemd/nginx layer around it differs.
"""

from __future__ import annotations

import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from functools import partial
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from staleness_pipeline.chronos_model import get_chronos_pipeline
from staleness_pipeline.kafka_consumer import (
    DEFAULT_BOOTSTRAP_SERVERS,
    DEFAULT_TOPIC,
    build_consumer,
    consume_messages,
)
from staleness_pipeline.live_store import LiveSeriesStore, UpdateEvent
from staleness_pipeline.reconstruction import reconstruct_stale_window
from staleness_pipeline.storage import ImputationConfidence

logger = logging.getLogger(__name__)

KAFKA_BOOTSTRAP_SERVERS = os.environ.get("STALENESS_KAFKA_BOOTSTRAP_SERVERS", DEFAULT_BOOTSTRAP_SERVERS)
KAFKA_TOPIC = os.environ.get("STALENESS_KAFKA_TOPIC", DEFAULT_TOPIC)
MIN_STUCK_HOURS = float(os.environ.get("STALENESS_MIN_STUCK_HOURS", "0.25"))
MAX_POINTS = int(os.environ.get("STALENESS_MAX_POINTS", "2000"))
# "earliest" so a freshly-started dashboard immediately shows whatever
# the producer has already sent this session, instead of an empty chart
# until the next new point arrives. Set to "latest" via env var if you'd
# rather a restart only pick up brand-new messages.
KAFKA_AUTO_OFFSET_RESET = os.environ.get("STALENESS_KAFKA_AUTO_OFFSET_RESET", "earliest")

STATIC_DIR = Path(__file__).parent / "dashboard_static"


class ConnectionManager:
    """Tracks connected WebSocket clients and broadcasts JSON to all of
    them, dropping any that have disconnected instead of letting one dead
    client break the broadcast for everyone else."""

    def __init__(self) -> None:
        self.active: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in self.active:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


manager = ConnectionManager()
store: LiveSeriesStore | None = None


def event_to_messages(event: UpdateEvent) -> list[dict]:
    """Convert one LiveSeriesStore.UpdateEvent into the JSON message(s)
    to broadcast over WebSocket — one 'point' message always, plus one
    'reconstruction' message per newly (re)computed stuck window."""
    messages: list[dict] = [
        {
            "type": "point",
            "point_id": event.point_id,
            "sensor": event.sensor,
            "timestamp": event.timestamp.isoformat(),
            "value": event.value,
        }
    ]
    for period, recon in zip(event.new_periods, event.new_reconstructions):
        is_open_ended = recon.method == "chronos-bolt-small-forward-only"
        messages.append(
            {
                "type": "reconstruction",
                "point_id": event.point_id,
                "sensor": event.sensor,
                "start_time": period.start_time.isoformat(),
                "end_time": period.end_time.isoformat(),
                "method": recon.method,
                "confidence": (
                    ImputationConfidence.PROVISIONAL.value
                    if is_open_ended
                    else ImputationConfidence.RECONCILED.value
                ),
                "values": [
                    {"timestamp": ts.isoformat(), "value": float(v)} for ts, v in recon.values.items()
                ],
            }
        )
    return messages


def _kafka_consumer_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Runs in a background OS thread — kafka-python is blocking/sync, so
    it can't run directly on the asyncio event loop. Each message is
    ingested into the shared store, and any resulting broadcast messages
    are handed back to the event loop via run_coroutine_threadsafe, since
    WebSocket sends must happen there, not on this thread."""
    assert store is not None
    logger.info("Connecting Kafka consumer: topic=%s servers=%s", KAFKA_TOPIC, KAFKA_BOOTSTRAP_SERVERS)
    consumer = build_consumer(
        topic=KAFKA_TOPIC,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        auto_offset_reset=KAFKA_AUTO_OFFSET_RESET,
    )

    for message in consume_messages(consumer):
        try:
            event = store.ingest(message)
        except Exception:
            logger.exception("Failed to ingest message: %r", message)
            continue

        for out in event_to_messages(event):
            asyncio.run_coroutine_threadsafe(manager.broadcast(out), loop)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global store
    logger.info("Loading Chronos pipeline (first run downloads ~100MB if not cached)...")
    pipeline = get_chronos_pipeline()
    store = LiveSeriesStore(
        reconstruct_fn=partial(reconstruct_stale_window, pipeline),
        max_points=MAX_POINTS,
        min_stuck_hours=MIN_STUCK_HOURS,
    )

    loop = asyncio.get_event_loop()
    thread = threading.Thread(target=_kafka_consumer_loop, args=(loop,), daemon=True)
    thread.start()
    logger.info("Live dashboard ready.")
    yield


app = FastAPI(title="Staleness live dashboard", lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
async def index() -> str:
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/series")
async def list_series() -> list[dict]:
    """All (point_id, sensor) pairs seen so far — lets the frontend
    discover what to chart without hardcoding sensor names."""
    if store is None:
        return []
    return [{"point_id": pid, "sensor": sensor} for pid, sensor in store.keys()]


@app.get("/api/series/{point_id}/{sensor}")
async def get_series(point_id: str, sensor: str) -> dict:
    """Full current buffer for one sensor — used to populate a chart on
    initial page load / reconnect, before any new WebSocket messages
    arrive."""
    if store is None:
        return {"timestamps": [], "values": []}
    series = store.get_series((point_id, sensor))
    if series is None:
        return {"timestamps": [], "values": []}
    return {
        "timestamps": [ts.isoformat() for ts in series.index],
        "values": [None if pd.isna(v) else float(v) for v in series.to_numpy()],
    }


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    await manager.connect(websocket)
    try:
        while True:
            # The client never sends anything meaningful — this just
            # keeps the connection open and lets us detect disconnects.
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)
