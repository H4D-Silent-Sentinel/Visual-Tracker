"""Link to the Silent Sentinel base station (repo H4D-Silent-Sentinel/silent-sentinel).

Spec: silent-sentinel/docs/jetson-interface.md (v1). One JSON object per UDP datagram.

  tracker -> base   "visual"  our tracks as camera boxes          base udp/47010
  base -> tracker   "cue"     where the fused track says the drone is   our udp/47011
  base -> tracker   "status"  C2 heartbeat, ~1 Hz (sentinel-c2): link health + node network state

The base learns our IP from the visual messages, so we send a heartbeat (empty
`tracks`) once a second even when nothing is in view; otherwise cues never arrive.

Configuration (environment, all optional):
  VT_BASE      base address, host[:port]          default: whoever sends us a cue/status
                                                    (127.0.0.1:47010 until then)
  VT_SRC       our camera id in the base site file   default post-1
  VT_CUE_PORT  port we listen on for cues            default 47011
Standard library only, so it runs unchanged on the Jetson.
"""
from __future__ import annotations

import json
import math
import os
import socket
import threading
import time

SCHEMA_VERSION = 1
VISUAL_PORT = 47010
CUE_PORT = 47011
DRONE_CLASSES = {"drone", "uav"}
MAX_TRACKS = 10
CUE_STALE_S = 5.0
C2_TIMEOUT_S = 3.0  # no cue/status for this long = base link down
M_PER_DEG_LAT = 110_540.0
M_PER_DEG_LON = 111_320.0


def parse_addr(addr: str | None, default_port: int = VISUAL_PORT) -> tuple[str, int]:
    addr = (addr or "").strip() or "127.0.0.1"
    host, _, port = addr.rpartition(":") if addr.count(":") == 1 else (addr, "", "")
    return host, int(port) if port else default_port


def vfov_deg(hfov: float, w: int, h: int) -> float:
    """Vertical field of view for a pinhole camera with square pixels."""
    return math.degrees(2 * math.atan(math.tan(math.radians(hfov) / 2) * h / w))


def send_targets(dets: list[dict], verify: bool) -> list[dict]:
    """Which tracker targets go to the base.

    With VLM verification on, only VLM-confirmed drones. Without it, every box the
    detector labels as a drone (the stock COCO model has no drone class, so it sends none)."""
    if verify:
        return [d for d in dets if d.get("verification", {}).get("status") == "confirmed"]
    return [d for d in dets if str(d.get("label", "")).lower() in DRONE_CLASSES]


def build_visual(src: str, t: float, dets: list[dict], frame_w: int, frame_h: int, settings: dict) -> dict:
    """One v1 `visual` message from tracker targets.

    dets: {"id", "conf", "label", "box": [x1, y1, x2, y2] in pixels, optional "range_m", "verification"}.
    settings: hfov (deg), yaw/pitch (deg, true north clockwise / up +), verify (bool),
              cam_lat/cam_lon/cam_alt (0,0 = not set: the base takes our position from its site file)."""
    hfov = float(settings.get("hfov", 60.0))
    sensor = {"yaw_deg": round(float(settings.get("yaw", 0.0)), 2),
              "pitch_deg": round(float(settings.get("pitch", 0.0)), 2),
              "hfov_deg": round(hfov, 2), "vfov_deg": round(vfov_deg(hfov, frame_w, frame_h), 2)}
    lat, lon = float(settings.get("cam_lat") or 0.0), float(settings.get("cam_lon") or 0.0)
    if lat or lon:
        sensor.update(lat=lat, lon=lon, alt_m=float(settings.get("cam_alt") or 0.0))
    tracks = []
    for d in send_targets(dets, bool(settings.get("verify")))[:MAX_TRACKS]:
        x1, y1, x2, y2 = (float(v) for v in d["box"])
        x1, x2 = max(0.0, x1), min(float(frame_w), x2)
        y1, y2 = max(0.0, y1), min(float(frame_h), y2)
        if x2 <= x1 or y2 <= y1:
            continue
        tr = {"id": d.get("id"), "conf": round(float(d["conf"]), 3), "class": "drone",
              "bbox": [round(x1 / frame_w, 5), round(y1 / frame_h, 5),
                       round((x2 - x1) / frame_w, 5), round((y2 - y1) / frame_h, 5)],
              # One pixel of the box width is one pixel of bearing; never claim better than 0.1 deg.
              "az_sigma_deg": round(max(0.1, hfov / frame_w), 3)}
        label = d.get("verification", {}).get("label")
        if label:
            tr["label"] = f"{label} (VLM)"
        rng = d.get("range_m")
        if rng and math.isfinite(rng):
            # Mil-relation range assumes a drone width we don't know yet: report it as rough.
            tr["range_m"] = round(float(rng), 1)
            tr["range_sigma_m"] = round(0.5 * float(rng), 1)
        tracks.append(tr)
    return {"v": SCHEMA_VERSION, "type": "visual", "src": src, "t": round(t, 3), "sensor": sensor, "tracks": tracks}


def parse_message(raw: bytes | str) -> dict:
    """A v1 message from the base: a validated `cue`, or a `status` heartbeat."""
    m = json.loads(raw)
    if not isinstance(m, dict) or m.get("v") != SCHEMA_VERSION:
        raise ValueError("not a v1 message")
    if m.get("type") == "status":
        return m
    return parse_cue(m)


def parse_cue(raw: bytes | str | dict) -> dict:
    m = raw if isinstance(raw, dict) else json.loads(raw)
    if m.get("v") != SCHEMA_VERSION or m.get("type") != "cue":
        raise ValueError("not a v1 cue message")
    for k in ("track_id", "lat", "lon"):
        if m.get(k) is None:
            raise ValueError(f"cue without {k}")
    return m


def cue_hint(cue: dict, age_s: float, settings: dict) -> dict:
    """Where the base's cue sits relative to our camera: left/right, distance, in view or not.

    Moves the cue forward by its velocity for the time since we received it."""
    lat = cue["lat"] + cue.get("vel_n", 0.0) * age_s / M_PER_DEG_LAT
    lon = cue["lon"] + cue.get("vel_e", 0.0) * age_s / (M_PER_DEG_LON * math.cos(math.radians(cue["lat"])))
    sigma = float(cue.get("sigma_m") or 0.0)
    hint = {"track_id": cue["track_id"], "lat": round(lat, 7), "lon": round(lon, 7), "sigma_m": sigma,
            "age_s": round(age_s, 1), "sources": cue.get("sources", []), "confirmed": cue.get("confirmed", True)}
    cam_lat, cam_lon = float(settings.get("cam_lat") or 0.0), float(settings.get("cam_lon") or 0.0)
    if not (cam_lat or cam_lon):
        hint["note"] = "Set the camera position to turn the cue into a direction."
        return hint
    de = (lon - cam_lon) * M_PER_DEG_LON * math.cos(math.radians(cam_lat))
    dn = (lat - cam_lat) * M_PER_DEG_LAT
    dist = math.hypot(de, dn)
    bearing = math.degrees(math.atan2(de, dn)) % 360
    rel = (bearing - float(settings.get("yaw", 0.0)) + 180) % 360 - 180  # + = right of centre
    alt = cue.get("alt_m")
    el = math.degrees(math.atan2(alt - float(settings.get("cam_alt") or 0.0), dist)) if alt is not None and dist > 0 else None
    hfov = float(settings.get("hfov", 60.0))
    hint.update(bearing_deg=round(bearing, 1), rel_az_deg=round(rel, 1), dist_m=round(dist, 1),
                el_deg=None if el is None else round(el, 1),
                spread_deg=round(math.degrees(math.atan2(2 * sigma, dist)), 1) if dist > 0 else 180.0,
                in_view=abs(rel) <= hfov / 2)
    return hint


class BaseLink:
    """Sends our tracks to the base and keeps the latest cue it sends back. Never blocks the tracker."""

    def __init__(self, base_addr: str | None = None, src: str | None = None, cue_port: int | None = None,
                 rate_hz: float = 5.0, heartbeat_s: float = 1.0, clock=time.time):
        given = base_addr or os.environ.get("VT_BASE")
        self.addr = parse_addr(given)
        self.auto_addr = not given  # follow the base that talks to us
        self.src = src or os.environ.get("VT_SRC", "post-1")
        self.cue_port = int(cue_port if cue_port is not None else os.environ.get("VT_CUE_PORT", CUE_PORT))
        self.min_dt, self.heartbeat_s, self.clock = 1 / rate_hz, heartbeat_s, clock
        self.lock = threading.Lock()
        self.sent = 0
        self.last_sent = None
        self.error = None
        self.cue_error = None
        self.cues: dict[int, tuple[float, dict]] = {}  # track_id -> (received at, cue)
        self.cues_received = 0
        self.c2: dict | None = None  # latest `status` heartbeat from the C2
        self.c2_heard: float | None = None  # when we last heard anything from the base
        self.c2_up = False
        self.tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.rx = None
        self.closed = False
        try:
            self.rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.rx.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.rx.bind(("0.0.0.0", self.cue_port))
            self.rx.settimeout(0.5)
            self.cue_port = self.rx.getsockname()[1]
            threading.Thread(target=self._listen, daemon=True, name="cue-listener").start()
        except OSError as e:  # e.g. another dashboard already holds the port: still send
            self.cue_error = f"Cannot listen for cues on udp/{self.cue_port}: {e}"
            if self.rx is not None:
                self.rx.close()
            self.rx = None

    def publish(self, dets: list[dict], frame_w: int, frame_h: int, settings: dict, t: float | None = None) -> dict | None:
        """Send if due (rate limit with tracks, heartbeat without). Returns the message sent, if any."""
        now = self.clock()
        msg = build_visual(self.src, t if t is not None else now, dets, frame_w, frame_h, settings)
        due = self.min_dt if msg["tracks"] else self.heartbeat_s
        if self.last_sent is not None and now - self.last_sent < due:
            return None
        try:
            self.tx.sendto(json.dumps(msg, separators=(",", ":")).encode(), self.addr)
            self.error = None
        except OSError as e:
            self.error = f"Cannot send to {self.addr[0]}:{self.addr[1]}: {e}"
        self.last_sent = now
        self.sent += 1
        return msg

    def _listen(self):
        while not self.closed:
            try:
                data, (ip, _) = self.rx.recvfrom(65535)
            except socket.timeout:
                continue
            except OSError:
                return
            self.on_message(data, ip)

    def on_message(self, data: bytes, ip: str | None = None):
        try:
            m = parse_message(data)
        except (ValueError, KeyError, TypeError) as e:
            self.cue_error = f"Bad message from base: {e}"
            return
        now = self.clock()
        with self.lock:
            self.c2_heard = now
            if m["type"] == "status":
                self.c2 = m
            else:
                self.cues[m["track_id"]] = (now, m)
                self.cues_received += 1
            self.cue_error = None
            if self.auto_addr and ip and self.addr[0] != ip:
                self.addr = (ip, VISUAL_PORT)  # the C2 found us (broadcast or cue): report back to it

    def latest_cue(self, settings: dict) -> dict | None:
        """Hint for the freshest confirmed cue (smallest uncertainty if several), or None."""
        now = self.clock()
        with self.lock:
            for tid in [k for k, (rt, _) in self.cues.items() if now - rt > CUE_STALE_S]:
                del self.cues[tid]
            live = list(self.cues.values())
        if not live:
            return None
        rt, cue = min(live, key=lambda rc: (rc[1].get("sigma_m") or 0.0, -rc[0]))
        return cue_hint(cue, now - rt, settings)

    def link_up(self) -> bool:
        """True while the base has sent us a cue or status within C2_TIMEOUT_S."""
        with self.lock:
            up = self.c2_heard is not None and self.clock() - self.c2_heard < C2_TIMEOUT_S
            self.c2_up = up
            return up

    def c2_status(self) -> dict:
        """Base link health plus what the C2 reports about the ESP32 node network."""
        up = self.link_up()
        with self.lock:
            c2 = dict(self.c2 or {})
            heard = self.c2_heard
        out = {"up": up, "last_heard_s": None if heard is None else round(self.clock() - heard, 1),
               "heartbeat": bool(c2)}
        for k in ("esp32", "nodes_online", "nodes_total", "nodes_detecting", "tracks", "confirmed", "hints_sent"):
            if k in c2:
                out[k] = c2[k]
        return out

    def status(self, settings: dict) -> dict:
        now = self.clock()
        return {"enabled": True, "base": f"{self.addr[0]}:{self.addr[1]}", "src": self.src,
                "cue_port": self.cue_port, "sent": self.sent, "cues_received": self.cues_received,
                "last_sent_s": None if self.last_sent is None else round(now - self.last_sent, 1),
                "error": self.error or self.cue_error, "cue": self.latest_cue(settings), "c2": self.c2_status()}

    def close(self):
        self.closed = True
        for s in (self.tx, self.rx):
            if s is not None:
                try:
                    s.close()
                except OSError:
                    pass
