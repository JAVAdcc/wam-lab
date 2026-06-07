"""Tiny HTTP/MJPEG live video server for benchmark render frames.

The server is intentionally dependency-light and process-local. Benchmarks
already publish RGB frames through ``EpisodeRecorder.record_video``; this
module lets the same frames be watched from a browser while the episode is
running. It does not change policy observations, environment stepping, or the
mp4 recorder.
"""

from __future__ import annotations

import atexit
import html
import io
import json
import logging
import threading
import time
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LiveVideoConfig:
    host: str = "127.0.0.1"
    port: int = 8765
    title: str = "WAM-Lab live simulation"
    jpeg_quality: int = 85

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None = None) -> "LiveVideoConfig":
        data = data or {}
        return cls(
            host=str(data.get("host", cls.host)),
            port=int(data.get("port", cls.port)),
            title=str(data.get("title", cls.title)),
            jpeg_quality=int(data.get("jpeg_quality", cls.jpeg_quality)),
        )


class LiveVideoServer:
    """Background MJPEG server that publishes the latest RGB frame."""

    def __init__(self, config: LiveVideoConfig) -> None:
        self.config = config
        self._condition = threading.Condition()
        self._frame_id = 0
        self._jpeg: bytes | None = None
        self._metadata: dict[str, Any] = {}
        self._last_update = 0.0
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    @property
    def url(self) -> str:
        return f"http://{self.config.host}:{self.config.port}/"

    def start(self) -> None:
        if self._httpd is not None:
            return
        handler_cls = _make_handler(self)
        self._httpd = ThreadingHTTPServer((self.config.host, self.config.port), handler_cls)
        self._httpd.daemon_threads = True
        self._thread = threading.Thread(
            target=self._httpd.serve_forever,
            name=f"vla-live-video-{self.config.port}",
            daemon=True,
        )
        self._thread.start()
        logger.info("Live simulation video stream: %s", self.url)

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        self._thread = None

    def publish(self, frame: np.ndarray, metadata: dict[str, Any] | None = None) -> None:
        jpeg = _encode_jpeg(frame, quality=self.config.jpeg_quality)
        with self._condition:
            self._frame_id += 1
            self._jpeg = jpeg
            self._metadata = dict(metadata or {})
            self._last_update = time.time()
            self._condition.notify_all()

    def mark_episode(self, metadata: dict[str, Any]) -> None:
        with self._condition:
            self._metadata = {**self._metadata, **metadata}
            self._last_update = time.time()
            self._condition.notify_all()

    def snapshot(self) -> tuple[int, bytes | None, dict[str, Any], float]:
        with self._condition:
            return self._frame_id, self._jpeg, dict(self._metadata), self._last_update

    def wait_for_frame(self, last_seen: int, timeout: float = 5.0) -> tuple[int, bytes | None]:
        deadline = time.monotonic() + timeout
        with self._condition:
            while self._frame_id <= last_seen:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._condition.wait(timeout=remaining)
            return self._frame_id, self._jpeg


_SERVERS: dict[tuple[str, int], LiveVideoServer] = {}
_SERVERS_LOCK = threading.Lock()


def ensure_live_video_server(config: dict[str, Any] | LiveVideoConfig | None = None) -> LiveVideoServer:
    cfg = config if isinstance(config, LiveVideoConfig) else LiveVideoConfig.from_dict(config)
    key = (cfg.host, cfg.port)
    with _SERVERS_LOCK:
        server = _SERVERS.get(key)
        if server is None:
            server = LiveVideoServer(cfg)
            server.start()
            _SERVERS[key] = server
        return server


def stop_all_live_video_servers() -> None:
    with _SERVERS_LOCK:
        servers = list(_SERVERS.values())
        _SERVERS.clear()
    for server in servers:
        server.stop()


atexit.register(stop_all_live_video_servers)


def _encode_jpeg(frame: np.ndarray, *, quality: int) -> bytes:
    if frame is None:
        raise ValueError("frame must not be None")
    arr = np.asarray(frame)
    if arr.ndim != 3 or arr.shape[2] not in (3, 4):
        raise ValueError(f"expected HxWx3/4 RGB frame, got shape={arr.shape}")
    if arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    arr = np.ascontiguousarray(arr)
    try:
        from PIL import Image
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError("live_video requires Pillow. Install pillow or disable recording.live_video.") from exc
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="JPEG", quality=max(1, min(95, int(quality))))
    return buf.getvalue()


def _make_handler(server: LiveVideoServer) -> type[BaseHTTPRequestHandler]:
    class LiveVideoHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt: str, *args: Any) -> None:
            logger.debug("live_video: " + fmt, *args)

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            path = self.path.split("?", 1)[0]
            if path in ("", "/", "/index.html"):
                self._send_html()
            elif path == "/stream.mjpg":
                self._send_stream()
            elif path == "/latest.jpg":
                self._send_latest()
            elif path == "/status.json":
                self._send_status()
            else:
                self.send_error(HTTPStatus.NOT_FOUND)

        def _send_html(self) -> None:
            title = html.escape(server.config.title)
            body = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    html, body {{ margin: 0; height: 100%; background: #101114; color: #f4f4f5; font: 14px system-ui, sans-serif; }}
    body {{ display: grid; grid-template-rows: auto 1fr; }}
    header {{ display: flex; justify-content: space-between; gap: 16px; padding: 10px 14px; background: #181a20; border-bottom: 1px solid #2b2f3a; }}
    main {{ display: grid; place-items: center; min-height: 0; }}
    img {{ max-width: 100vw; max-height: calc(100vh - 45px); object-fit: contain; image-rendering: auto; }}
    code {{ color: #c8d3ff; }}
  </style>
</head>
<body>
  <header>
    <div>{title}</div>
    <div id="status"><code>waiting for frames</code></div>
  </header>
  <main><img src="/stream.mjpg" alt="live simulation stream"></main>
  <script>
    async function tick() {{
      try {{
        const r = await fetch('/status.json', {{cache: 'no-store'}});
        const s = await r.json();
        const age = s.last_update_ts ? Math.max(0, Date.now() / 1000 - s.last_update_ts).toFixed(1) : 'n/a';
        const task = s.metadata && s.metadata.task_name ? ' | ' + s.metadata.task_name : '';
        document.getElementById('status').innerHTML = `<code>frame ${{s.frame_id}} | age ${{age}}s${{task}}</code>`;
      }} catch (e) {{}}
    }}
    setInterval(tick, 1000);
    tick();
  </script>
</body>
</html>
"""
            payload = body.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _send_status(self) -> None:
            frame_id, jpeg, metadata, last_update = server.snapshot()
            payload = json.dumps(
                {
                    "frame_id": frame_id,
                    "has_frame": jpeg is not None,
                    "last_update_ts": last_update or None,
                    "metadata": metadata,
                }
            ).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def _send_latest(self) -> None:
            _, jpeg, _, _ = server.snapshot()
            if jpeg is None:
                self.send_error(HTTPStatus.NOT_FOUND, "No frame has been published yet")
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(jpeg)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(jpeg)

        def _send_stream(self) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
            self.send_header("Pragma", "no-cache")
            self.end_headers()
            last_seen = -1
            while True:
                frame_id, jpeg = server.wait_for_frame(last_seen)
                if jpeg is None or frame_id == last_seen:
                    continue
                last_seen = frame_id
                try:
                    self.wfile.write(b"--frame\r\n")
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpeg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    break

    return LiveVideoHandler
