"""Silent Sentinel base link: message shape, rate limits, cue round trip, and the live loop."""
import importlib.util
import json
import math
import os
import socket
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import base_link  # noqa: E402
from base_link import BaseLink, build_visual, cue_hint, parse_addr, parse_cue  # noqa: E402

FRAME_W, FRAME_H = 1920, 1080
CAM = {"hfov": 60.0, "yaw": 90.0, "pitch": 0.0, "cam_lat": 40.808, "cam_lon": -73.963, "cam_alt": 4.0}


def det(i=1, box=(960, 520, 1000, 560), conf=0.8, label="drone", **kw):
    return {"id": i, "box": list(box), "conf": conf, "label": label, "range_m": 120.0, **kw}


def receiver():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind(("127.0.0.1", 0))
    s.settimeout(2)
    return s


def cue(track_id=3, lat=40.808, lon=-73.963, **kw):
    return {"v": 1, "type": "cue", "t": time.time(), "track_id": track_id, "lat": lat, "lon": lon,
            "alt_m": 4.0, "sigma_m": 30.0, "vel_e": 0.0, "vel_n": 0.0, "confirmed": True,
            "sources": ["acoustic"], **kw}


def east_of(m, lat=CAM["cam_lat"], lon=CAM["cam_lon"]):
    return lon + m / (base_link.M_PER_DEG_LON * math.cos(math.radians(lat)))


def wait_for(pred, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end:
        if pred():
            return True
        time.sleep(0.02)
    return False


class MessageTests(unittest.TestCase):
    def test_addresses(self):
        self.assertEqual(parse_addr(None), ("127.0.0.1", 47010))
        self.assertEqual(parse_addr("192.168.1.20"), ("192.168.1.20", 47010))
        self.assertEqual(parse_addr("base.local:5000"), ("base.local", 5000))

    def test_visual_matches_the_base_interface(self):
        m = build_visual("post-1", 100.0, [det(verification={"status": "confirmed", "label": "drone"})],
                         FRAME_W, FRAME_H, {**CAM, "verify": True})
        self.assertEqual((m["v"], m["type"], m["src"], m["t"]), (1, "visual", "post-1", 100.0))
        s = m["sensor"]
        self.assertEqual((s["yaw_deg"], s["pitch_deg"], s["hfov_deg"], s["lat"], s["alt_m"]), (90.0, 0.0, 60.0, 40.808, 4.0))
        self.assertAlmostEqual(s["vfov_deg"], 35.98, places=1)  # 16:9 at 60 deg horizontal
        tr = m["tracks"][0]
        self.assertEqual(tr["bbox"], [0.5, round(520 / 1080, 5), round(40 / 1920, 5), round(40 / 1080, 5)])
        self.assertEqual((tr["class"], tr["label"], tr["range_m"], tr["range_sigma_m"]), ("drone", "drone (VLM)", 120.0, 60.0))
        json.dumps(m)  # must be plain JSON

    def test_fixed_post_leaves_position_to_the_site_file(self):
        m = build_visual("post-1", 1.0, [], FRAME_W, FRAME_H, {**CAM, "cam_lat": 0.0, "cam_lon": 0.0})
        self.assertNotIn("lat", m["sensor"])
        self.assertEqual(m["tracks"], [])

    def test_only_drones_are_sent(self):
        dets = [det(1), det(2, label="airplane"), det(3, box=(-50, -50, -10, -10))]
        self.assertEqual([t["id"] for t in build_visual("p", 0, dets, FRAME_W, FRAME_H, CAM)["tracks"]], [1])
        verified = [det(1, verification={"status": "rejected", "label": "bird"}),
                    det(2, verification={"status": "pending"}),
                    det(3, verification={"status": "confirmed", "label": "drone"})]
        m = build_visual("p", 0, verified, FRAME_W, FRAME_H, {**CAM, "verify": True})
        self.assertEqual([t["id"] for t in m["tracks"]], [3])

    def test_phone_stands_in_for_the_drone(self):
        dets = [det(label="cell phone", conf=0.7), det(label="drone"), det(label="person")]
        msg = build_visual("post-1", 1.0, dets, FRAME_W, FRAME_H, {**CAM, "standin": True, "verify": True})
        self.assertEqual(len(msg["tracks"]), 1)  # only the phone, and the VLM check doesn't apply
        tr = msg["tracks"][0]
        self.assertEqual((tr["class"], tr["label"]), ("drone", base_link.STANDIN_LABEL))
        plain = build_visual("post-1", 1.0, dets, FRAME_W, FRAME_H, CAM)
        self.assertTrue(all("label" not in t or t["label"] != base_link.STANDIN_LABEL for t in plain["tracks"]))

    def test_phone_range_uses_its_long_side(self):
        # 15 cm phone filling 10% of a 60-degree view: ~1.3 m; held upright or sideways gives the same answer
        wide = base_link.phone_range((0, 0, 128, 60), 1280, 60.0)
        tall = base_link.phone_range((0, 0, 60, 128), 1280, 60.0)
        self.assertAlmostEqual(wide, tall)
        self.assertAlmostEqual(wide, 0.15 / (2 * math.tan(math.atan(0.1 * math.tan(math.radians(30))))), places=6)
        self.assertTrue(1.2 < wide < 1.4)

    def test_unknown_range_is_left_out(self):
        m = build_visual("p", 0, [det(range_m=float("inf")), det(2, range_m=None)], FRAME_W, FRAME_H, CAM)
        self.assertTrue(all("range_m" not in t for t in m["tracks"]))

    def test_cue_geometry(self):
        ahead = cue_hint(cue(lon=east_of(100)), 0.0, CAM)  # camera faces east
        self.assertAlmostEqual(ahead["rel_az_deg"], 0.0, places=1)
        self.assertAlmostEqual(ahead["dist_m"], 100.0, delta=0.5)
        self.assertTrue(ahead["in_view"])
        north = cue_hint(cue(lat=CAM["cam_lat"] + 100 / base_link.M_PER_DEG_LAT), 0.0, CAM)
        self.assertAlmostEqual(north["rel_az_deg"], -90.0, places=1)  # left of an east-facing camera
        self.assertFalse(north["in_view"])
        moving = cue_hint(cue(lon=east_of(100), vel_e=10.0), 2.0, CAM)  # 2 s later, 20 m further
        self.assertAlmostEqual(moving["dist_m"], 120.0, delta=0.5)
        no_pose = cue_hint(cue(), 0.0, {**CAM, "cam_lat": 0.0, "cam_lon": 0.0})
        self.assertNotIn("rel_az_deg", no_pose)
        self.assertIn("note", no_pose)

    def test_bad_cues_are_rejected(self):
        for raw in ('{"v":1,"type":"visual"}', '{"v":2,"type":"cue","track_id":1,"lat":0,"lon":0}', '{"v":1,"type":"cue","lat":1}'):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                parse_cue(raw)


class LinkTests(unittest.TestCase):
    def setUp(self):
        self.rx = receiver()
        self.now = [1000.0]
        self.link = BaseLink(f"127.0.0.1:{self.rx.getsockname()[1]}", "post-1", cue_port=0, clock=lambda: self.now[0])

    def tearDown(self):
        self.link.close()
        self.rx.close()

    def recv(self):
        return json.loads(self.rx.recvfrom(65535)[0])

    def test_rate_limit_and_heartbeat(self):
        self.assertIsNotNone(self.link.publish([det()], FRAME_W, FRAME_H, CAM))
        self.assertEqual(len(self.recv()["tracks"]), 1)
        self.now[0] += 0.1
        self.assertIsNone(self.link.publish([det()], FRAME_W, FRAME_H, CAM))  # 5 Hz with tracks
        self.now[0] += 0.15
        self.assertIsNotNone(self.link.publish([det()], FRAME_W, FRAME_H, CAM))
        self.recv()
        self.now[0] += 0.5
        self.assertIsNone(self.link.publish([], FRAME_W, FRAME_H, CAM))  # 1 Hz heartbeat when empty
        self.now[0] += 0.6
        self.assertEqual(self.link.publish([], FRAME_W, FRAME_H, CAM)["tracks"], [])
        self.assertEqual(self.recv()["tracks"], [])

    def test_cue_round_trip_and_expiry(self):
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx.sendto(b"not json", ("127.0.0.1", self.link.cue_port))
        tx.sendto(json.dumps(cue(lon=east_of(100))).encode(), ("127.0.0.1", self.link.cue_port))
        tx.close()
        self.assertTrue(wait_for(lambda: self.link.cues_received == 1))
        st = self.link.status(CAM)
        self.assertEqual((st["cue"]["track_id"], st["cue"]["in_view"]), (3, True))
        self.assertIsNone(st["error"])
        self.now[0] += base_link.CUE_STALE_S + 1
        self.assertIsNone(self.link.latest_cue(CAM))

    def test_c2_status_heartbeat(self):
        status = {"v": 1, "type": "status", "src": "sentinel-c2", "t": 1000.0, "esp32": "up",
                  "nodes_online": 5, "nodes_total": 6, "nodes_detecting": [2], "tracks": 1, "confirmed": 1}
        self.assertFalse(self.link.c2_status()["up"])
        tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        tx.sendto(json.dumps(status).encode(), ("127.0.0.1", self.link.cue_port))
        tx.close()
        self.assertTrue(wait_for(lambda: self.link.c2 is not None))
        st = self.link.status(CAM)
        self.assertIsNone(st["error"])  # a heartbeat is not a bad cue
        self.assertEqual(st["cues_received"], 0)
        self.assertEqual((st["c2"]["up"], st["c2"]["esp32"], st["c2"]["nodes_online"]), (True, "up", 5))
        self.now[0] += base_link.C2_TIMEOUT_S + 0.1
        self.assertFalse(self.link.c2_status()["up"])  # silent base shows as down

    def test_follows_the_base_that_finds_us(self):
        old = os.environ.pop("VT_BASE", None)
        try:
            auto = BaseLink(None, "post-1", cue_port=0)
            auto.on_message(b'{"v":1,"type":"status","src":"sentinel-c2"}', "10.0.0.5")
            self.assertEqual(auto.addr, ("10.0.0.5", base_link.VISUAL_PORT))
            auto.close()
        finally:
            if old is not None:
                os.environ["VT_BASE"] = old
        self.link.on_message(b'{"v":1,"type":"status"}', "10.0.0.5")  # explicit --base is never overridden
        self.assertEqual(self.link.addr[0], "127.0.0.1")

    def test_busy_cue_port_still_sends(self):
        other = BaseLink(f"127.0.0.1:{self.rx.getsockname()[1]}", "post-2", cue_port=self.link.cue_port)
        try:
            if other.rx is None:  # SO_REUSEADDR may or may not allow a second UDP bind
                self.assertIn("Cannot listen", other.status(CAM)["error"])
            other.publish([det()], FRAME_W, FRAME_H, CAM)
            self.assertEqual(self.recv()["src"], "post-2")
        finally:
            other.close()


class LiveLoopTests(unittest.TestCase):
    """The dashboard's live loop sends tracks and draws the base cue."""

    @classmethod
    def setUpClass(cls):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        spec = importlib.util.spec_from_file_location('link_live_engine', Path(__file__).resolve().parents[1] / 'dashboard/live_engine.py')
        cls.live = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': torch, 'ultralytics': SimpleNamespace(YOLO=object)}), \
             patch('subprocess.run', return_value=SimpleNamespace(stderr='')):
            spec.loader.exec_module(cls.live)

    def run_frame(self, engine, names=None, cls=0, seen=None):
        class End(Exception):
            pass
        image = np.zeros((1080, 1920, 3), np.uint8)
        names = names or {0: 'drone'}
        boxes = [SimpleNamespace(xyxy=[np.array([940, 500, 980, 540])], conf=.9, id=7, cls=cls)]
        seen = {} if seen is None else seen

        def track(*a, **k):
            seen['classes'] = k.get('classes')
            return [SimpleNamespace(boxes=boxes, names=names)]
        model = SimpleNamespace(predictor=None, names=names, track=track)

        class Capture:
            count = 0
            def read(self):
                self.count += 1
                if self.count > 1:
                    raise End()
                return True, image
            def get(self, *_): return 1
            def release(self): pass
        engine.viewers = 1
        engine.open_source = lambda _: (Capture(), 30, False, 1)
        def get_model(name):
            seen['model'] = name
            return model
        engine.get_model = get_model
        with self.assertRaises(End):
            engine.run()
        return engine.stats

    def test_live_loop_sends_and_shows_cue(self):
        rx = receiver()
        with patch.object(self.live, 'list_sources', return_value=['clip.mp4']):
            engine = self.live.Engine()
        engine.update(link='true', yaw='90', cam_lat=str(CAM['cam_lat']), cam_lon=str(CAM['cam_lon']), cam_alt='4')
        engine.link = BaseLink(f"127.0.0.1:{rx.getsockname()[1]}", "post-1", cue_port=0)
        try:
            tx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            tx.sendto(json.dumps(cue(lon=east_of(80))).encode(), ("127.0.0.1", engine.link.cue_port))
            tx.close()
            self.assertTrue(wait_for(lambda: engine.link.cues_received == 1))
            stats = self.run_frame(engine)
            msg = json.loads(rx.recvfrom(65535)[0])
            self.assertEqual(msg["src"], "post-1")
            self.assertEqual(msg["tracks"][0]["id"], 7)
            self.assertIsNotNone(msg["tracks"][0]["range_m"])
            self.assertTrue(stats["link"]["enabled"])
            self.assertTrue(stats["link"]["cue"]["in_view"])
            self.assertIsNotNone(engine.jpeg)
            json.dumps(stats)  # served to the browser as JSON
        finally:
            engine.link.close()
            rx.close()

    def test_phone_standin_runs_through_the_live_loop(self):
        rx = receiver()
        with patch.object(self.live, 'list_sources', return_value=['clip.mp4']):
            engine = self.live.Engine()
        engine.update(link='true', standin='true', verify='true', model='v4')
        engine.link = BaseLink(f"127.0.0.1:{rx.getsockname()[1]}", "post-1", cue_port=0)
        seen = {}
        try:
            self.run_frame(engine, names={0: 'person', 67: 'cell phone'}, cls=67, seen=seen)
            msg = json.loads(rx.recvfrom(65535)[0])
            self.assertEqual(seen['model'], 'stock')  # stock COCO model, whatever model was picked
            self.assertEqual(seen['classes'], [67])  # phones only
            tr = msg["tracks"][0]
            self.assertEqual((tr["class"], tr["label"]), ("drone", base_link.STANDIN_LABEL))
            # 40 px phone (15 cm long side) in a 1920 px, 60 deg view: about 6 m
            self.assertTrue(5 < tr["range_m"] < 7, tr["range_m"])
            self.assertIsNone(engine.verifier)  # VLM never consulted for a phone
        finally:
            engine.link.close()
            rx.close()

    def test_phone_source_reports_as_chaser_from_its_own_pose(self):
        import phone_feed
        rx = receiver()
        with patch.object(self.live, 'list_sources', return_value=['clip.mp4']):
            engine = self.live.Engine()
        engine.update(link='true', source='phone', yaw='0', cam_lat='1', cam_lon='1')  # typed pose is ignored
        engine.link = BaseLink(f"127.0.0.1:{rx.getsockname()[1]}", "post-1", cue_port=0)
        feed = phone_feed.feed()
        try:
            feed.put(b"x", {"lat": 40.8082, "lon": -73.9637, "alt_m": 3, "acc_m": 5, "heading": 135.0, "pitch": 12.0})
            self.run_frame(engine)
            msg = json.loads(rx.recvfrom(65535)[0])
            self.assertEqual(msg["src"], "chaser-1")
            self.assertEqual((msg["sensor"]["lat"], msg["sensor"]["lon"]), (40.8082, -73.9637))
            self.assertEqual((msg["sensor"]["yaw_deg"], msg["sensor"]["pitch_deg"]), (135.0, 12.0))
            # phone without a GPS fix: heartbeat still goes out, but with no position
            feed.pose, feed.pose_t = {"heading": 135.0}, 0.0
            engine.restart = True  # each run_frame is a fresh engine.run(): reopen the source
            engine.link.last_sent = None  # skip the 1 Hz heartbeat limit
            stats = self.run_frame(engine)
            msg = json.loads(rx.recvfrom(65535)[0])
            self.assertNotIn("lat", msg["sensor"])
            self.assertIn("GPS", stats["link"]["note"])
        finally:
            feed.pose, feed.pose_t = {}, 0.0
            engine.link.close()
            rx.close()

    def test_link_off_closes_the_socket(self):
        with patch.object(self.live, 'list_sources', return_value=['clip.mp4']):
            engine = self.live.Engine()
        engine.link = BaseLink("127.0.0.1:9", cue_port=0)
        self.assertEqual(engine.update_link(engine.settings, [], 640, 480, 0.0), {"enabled": False})
        self.assertIsNone(engine.link)


if __name__ == '__main__':
    unittest.main()
