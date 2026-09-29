"""Web dashboard and runtime controller for the Automated Fruit Sorter."""
from __future__ import annotations

import asyncio
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock, Thread

try:
    import cv2
except ImportError:  # Keeps status endpoints usable during an incomplete install.
    cv2 = None

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from core.FruitClassifier import FruitClassifier
from core.database import SorterDatabase


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CAMERA_URL = "http://10.11.231.215:8080/stream"
CAMERA_SOURCE = os.getenv("FRUIT_SORTER_CAMERA_URL", DEFAULT_CAMERA_URL)
DATA_DIRECTORY = Path(os.getenv("FRUIT_SORTER_DATA_DIR", str(ROOT / "data"))).resolve()
FRAME_DIRECTORY = DATA_DIRECTORY / "frames"
MODEL_PATH = ROOT / "models" / "FruitClassifier.keras"
CLASSES_PATH = ROOT / "models" / "FruitClassifier.json"
ENABLE_DEMO = os.getenv("FRUIT_SORTER_ENABLE_DEMO", "false").lower() in {"1", "true", "yes"}


def _positive_int(name: str, default: int) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


FRAME_LIMIT = _positive_int("FRUIT_SORTER_FRAME_LIMIT", 500)
DATA_DIRECTORY.mkdir(parents=True, exist_ok=True)
FRAME_DIRECTORY.mkdir(parents=True, exist_ok=True)
DATABASE = SorterDatabase(DATA_DIRECTORY / "fruit_sorter.sqlite3")


class SorterRuntime:
    """Keeps camera access and ML inference in one background thread."""

    def __init__(self) -> None:
        self.lock = Lock()
        self.running = False
        self.camera_available = False
        self.model_ready = False
        self.error: str | None = None
        self.frame: bytes | None = None
        self.total_sorted = DATABASE.total_events()
        self.started_at: float | None = None
        self._thread: Thread | None = None
        self._classifier: FruitClassifier | None = None
        self.last_result = {"label": "Waiting", "fruit": "—", "quality": "Waiting", "confidence": None, "frame_path": None}
        latest = DATABASE.latest_event()
        if latest:
            self.last_result = latest

    def start(self) -> None:
        with self.lock:
            if self.running:
                return
            self.running = True
            self.error = None
            self.started_at = time.time()
        self._thread = Thread(target=self._run, daemon=True, name="fruit-sorter")
        self._thread.start()

    def stop(self) -> None:
        with self.lock:
            self.running = False

    def join(self, timeout: float = 5) -> None:
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)

    def _load_model(self) -> None:
        if not MODEL_PATH.exists():
            raise RuntimeError(f"Model not found: {MODEL_PATH.name}")
        if not CLASSES_PATH.exists():
            raise RuntimeError(f"Class labels not found: {CLASSES_PATH.name}")
        self._classifier = FruitClassifier(str(MODEL_PATH), str(CLASSES_PATH))
        self.model_ready = True

    @staticmethod
    def _format_result(label: str) -> dict:
        value = label.lower().replace(" ", "")
        fruit = label.lower().replace("rotten", "").replace("unripe", "").strip().title()
        quality = "Rotten" if "rotten" in value else "Unripe" if "unripe" in value else "Healthy"
        return {"label": label, "fruit": fruit, "quality": quality, "confidence": None}

    def _prune_frames(self) -> None:
        if FRAME_LIMIT <= 0:
            return
        frames = sorted(FRAME_DIRECTORY.glob("event-*.jpg"), key=lambda item: item.stat().st_mtime, reverse=True)
        for path in frames[FRAME_LIMIT:]:
            try:
                path.unlink()
            except OSError:
                pass

    def _save_result(self, result: dict, image=None, source: str = "camera") -> None:
        frame_path = None
        if image is not None and cv2 is not None and FRAME_LIMIT > 0:
            filename = f"event-{int(time.time() * 1000)}.jpg"
            if cv2.imwrite(str(FRAME_DIRECTORY / filename), image):
                frame_path = filename
                self._prune_frames()
        self.last_result = DATABASE.record(result, source, frame_path)
        self.total_sorted += 1

    def _run(self) -> None:
        if cv2 is None:
            with self.lock:
                self.error = "OpenCV is not installed. Install the project dependencies, then restart the sorter."
                self.running = False
            return
        cap = cv2.VideoCapture(CAMERA_SOURCE)
        with self.lock:
            self.camera_available = cap.isOpened()
        if not cap.isOpened():
            with self.lock:
                self.error = f"Camera stream could not be opened: {CAMERA_SOURCE}"
                self.running = False
            return
        try:
            try:
                self._load_model()
            except Exception as exc:
                with self.lock:
                    self.error = f"Model unavailable: {exc}"
            last_inference = 0.0
            while self.running:
                ok, image = cap.read()
                if not ok:
                    with self.lock:
                        self.error = "Camera frame could not be read."
                    break
                now = time.time()
                if self._classifier and now - last_inference >= 0.7:
                    try:
                        with self.lock:
                            self._save_result(self._format_result(self._classifier.predict(image)), image=image)
                    except Exception as exc:
                        with self.lock:
                            self.error = f"Prediction failed: {exc}"
                    last_inference = now
                ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 75])
                if ok:
                    with self.lock:
                        self.frame = encoded.tobytes()
                time.sleep(0.02)
        finally:
            cap.release()
            with self.lock:
                self.running = False
                self.camera_available = False

    def status(self) -> dict:
        with self.lock:
            uptime = int(time.time() - self.started_at) if self.running and self.started_at else 0
            return {"running": self.running, "camera_available": self.camera_available, "camera_source": CAMERA_SOURCE, "model_ready": self.model_ready, "error": self.error, "result": self.last_result, "total_sorted": self.total_sorted, "uptime_seconds": uptime, "report": DATABASE.report()}


runtime = SorterRuntime()


@asynccontextmanager
async def lifespan(_: FastAPI):
    yield
    runtime.stop()
    runtime.join()


app = FastAPI(title="Orchard Sorter", version="1.0.0", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "API" / "static"), name="static")


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def dashboard() -> HTMLResponse:
    return HTMLResponse((ROOT / "API" / "static" / "index.html").read_text(encoding="utf-8"))


@app.get("/healthz", include_in_schema=False)
async def healthz() -> dict:
    """Liveness probe: the HTTP service and persistent store are available."""
    return {"status": "ok"}


@app.get("/readyz", include_in_schema=False)
async def readyz() -> dict:
    """Readiness probe: all files needed to run a classification are present."""
    if not (MODEL_PATH.is_file() and CLASSES_PATH.is_file() and DATA_DIRECTORY.is_dir()):
        raise HTTPException(status_code=503, detail="Runtime assets are unavailable")
    return {"status": "ready"}


@app.get("/api/status")
async def status() -> dict:
    return runtime.status()


@app.get("/api/events")
async def events(limit: int = 50) -> dict:
    return {"events": DATABASE.recent_events(max(1, min(limit, 200)))}


@app.get("/api/report")
async def report() -> dict:
    return DATABASE.report()


@app.get("/api/frames/{filename}")
async def saved_frame(filename: str):
    path = FRAME_DIRECTORY / Path(filename).name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Frame not found")
    return FileResponse(path, media_type="image/jpeg")


@app.post("/api/sorter/start")
async def start_sorter() -> dict:
    runtime.start()
    return runtime.status()


@app.post("/api/sorter/stop")
async def stop_sorter() -> dict:
    runtime.stop()
    return runtime.status()


@app.get("/api/camera")
async def camera_feed():
    async def stream():
        while True:
            with runtime.lock:
                frame, running = runtime.frame, runtime.running
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            elif not running:
                break
            await asyncio.sleep(0.08)

    return StreamingResponse(stream(), media_type="multipart/x-mixed-replace; boundary=frame")


class DemoResult(BaseModel):
    label: str


@app.post("/api/demo-result", include_in_schema=False)
async def demo_result(payload: DemoResult) -> dict:
    if not ENABLE_DEMO:
        raise HTTPException(status_code=404, detail="Demo endpoint is disabled")
    label = payload.label.strip()
    if not label or len(label) > 100:
        raise HTTPException(status_code=422, detail="A label between 1 and 100 characters is required")
    with runtime.lock:
        runtime._save_result(runtime._format_result(label), source="demo")
    return runtime.status()


def run() -> None:
    """Run the production HTTP server from the installed console script."""
    import uvicorn

    uvicorn.run("API.SERVE:app", host="0.0.0.0", port=8000)
