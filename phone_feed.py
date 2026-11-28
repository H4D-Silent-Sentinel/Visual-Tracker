"""Phone as the chaser drone's camera: the phone streams its back camera plus GPS, compass heading
and tilt over Wi-Fi; the Live tab's "Phone (chaser-1)" source reads it here.

  phone (Safari) --https POST /frame (JPEG + X-Pose)--> this server (port 8443) --> Live engine

Safari only allows camera, GPS and compass on a secure page, so this serves https with a
self-signed certificate made on first use (openssl). On the phone, open https://<this Mac's IP>:8443,
accept the one-time "not private" warning, and tap Start. Phone and Mac must reach each other:
same Wi-Fi (campus Wi-Fi often blocks this) or, simplest, put the Mac on the phone's hotspot.
Standard library only.
"""
from __future__ import annotations

import json
import os
import socket
import ssl
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAGE = ROOT / "dashboard" / "phone.html"
CERT_DIR = ROOT / ".venv" / "phone-cert"
PHONE_PORT = int(os.environ.get("VT_PHONE_PORT", "8443"))
CHASER_SRC = os.environ.get("VT_CHASER_SRC", "chaser-1")
POSE_FRESH_S = 3.0
MAX_FRAME = 4 * 1024 * 1024


def lan_ip() -> str:
    """This machine's address on the local network (no traffic is sent)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("10.255.255.255", 1))
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(cert_dir: Path = CERT_DIR) -> tuple[Path, Path]:
    cert_dir.mkdir(parents=True, exist_ok=True)
    cert, key = cert_dir / "cert.pem", cert_dir / "key.pem"
    if not (cert.exists() and key.exists()):
        subprocess.run(["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "825",
                        "-subj", "/CN=Visual-Tracker phone link", "-keyout", str(key), "-out", str(cert)],
                       check=True, capture_output=True)
    return cert, key


class PhoneFeed:
    """Latest frame and pose from the phone. One per process (see feed())."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg: bytes | None = None
        self.seq = 0
        self.pose: dict = {}
        self.pose_t = 0.0
        self.frame_t = 0.0
        self.server = None
        self.error: str | None = None
        self.url: str | None = None

    # -- input (from the phone) --------------------------------------------------------------
    def put(self, jpeg: bytes, pose: dict | None, now: float | None = None):
        now = time.time() if now is None else now
        with self.cond:
            self.jpeg, self.seq, self.frame_t = jpeg, self.seq + 1, now
            if pose and pose.get("lat") is not None and pose.get("lon") is not None:
                self.pose, self.pose_t = dict(pose), now
            elif pose:  # heading/tilt without a GPS fix yet: keep them, position stays unknown
                self.pose = {**{k: v for k, v in self.pose.items() if k in ("lat", "lon", "alt_m", "acc_m")},
                             **pose}
            self.cond.notify_all()

    # -- output (to the Live engine) ---------------------------------------------------------
    def wait_frame(self, after_seq: int, timeout: float = 1.0) -> tuple[int, bytes | None]:
        with self.cond:
            self.cond.wait_for(lambda: self.seq > after_seq, timeout)
            return self.seq, (self.jpeg if self.seq > after_seq else None)

    def live_pose(self, now: float | None = None) -> dict | None:
        """Camera pose for the base message, or None if the phone hasn't sent a fresh GPS fix."""
        now = time.time() if now is None else now
        with self.cond:
            p = dict(self.pose)
            fresh = self.pose_t and now - self.pose_t <= POSE_FRESH_S
        if not fresh or p.get("lat") is None or p.get("heading") is None:
            return None
        return {"cam_lat": float(p["lat"]), "cam_lon": float(p["lon"]), "cam_alt": float(p.get("alt_m") or 0.0),
                "yaw": float(p["heading"]) % 360, "pitch": float(p.get("pitch") or 0.0)}

    def status(self, now: float | None = None) -> dict:
        now = time.time() if now is None else now
        with self.cond:
            p, ft, pt = dict(self.pose), self.frame_t, self.pose_t
        return {"url": self.url, "error": self.error, "frames": self.seq,
                "frame_age_s": None if not ft else round(now - ft, 1),
                "gps": None if p.get("lat") is None else {"acc_m": p.get("acc_m"), "age_s": round(now - pt, 1)},
                "heading": p.get("heading"), "pitch": p.get("pitch")}

    # -- server ------------------------------------------------------------------------------
    def start(self, port: int = PHONE_PORT, cert_dir: Path = CERT_DIR, host: str = "0.0.0.0") -> str | None:
        """Start the https server (idempotent). Returns the URL to open on the phone, or None on error."""
        if self.server is not None:
            return self.url
        try:
            cert, key = ensure_cert(cert_dir)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            ctx.load_cert_chain(cert, key)
            srv = ThreadingHTTPServer((host, port), _handler(self))
            srv.socket = ctx.wrap_socket(srv.socket, server_side=True)
        except Exception as e:  # port busy, no openssl, ...
            self.error = f"Phone link not started: {e}"
            return None
        self.server = srv
        threading.Thread(target=srv.serve_forever, daemon=True, name="phone-feed").start()
        self.url = f"https://{lan_ip()}:{srv.server_address[1]}"
        return self.url

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server = None


def _handler(feed: PhoneFeed):
    class H(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str = "application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                return self._send(200, PAGE.read_bytes(), "text/html; charset=utf-8")
            if self.path == "/status":
                return self._send(200, json.dumps(feed.status()).encode())
            self.send_error(404)

        def do_POST(self):
            if self.path != "/frame":
                return self.send_error(404)
            n = int(self.headers.get("Content-Length") or 0)
            if not 0 < n <= MAX_FRAME:
                return self.send_error(413)
            jpeg = self.rfile.read(n)
            try:
                pose = json.loads(self.headers.get("X-Pose") or "null")
            except ValueError:
                pose = None
            feed.put(jpeg, pose if isinstance(pose, dict) else None)
            self._send(200, json.dumps(feed.status()).encode())

        def log_message(self, *a):
            pass

    return H


class PhoneCapture:
    """cv2.VideoCapture look-alike for the Live engine: frames from the phone."""

    def __init__(self, feed: PhoneFeed):
        self.feed, self.last = feed, 0

    def read(self):
        import cv2
        import numpy as np
        seq, jpeg = self.feed.wait_frame(self.last, timeout=1.0)
        if jpeg is None:
            return False, None
        self.last = seq
        img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
        return (img is not None), img

    def isOpened(self):
        return True

    def get(self, prop):
        return float(self.last)

    def set(self, *a):
        return False

    def release(self):
        pass


_FEED: PhoneFeed | None = None


def feed() -> PhoneFeed:
    global _FEED
    if _FEED is None:
        _FEED = PhoneFeed()
    return _FEED
