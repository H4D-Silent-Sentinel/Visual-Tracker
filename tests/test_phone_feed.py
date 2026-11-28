"""Phone as chaser-1: https upload of frames + pose, and the pose the engine reports."""
import json
import ssl
import tempfile
import unittest
import urllib.request
from pathlib import Path

import phone_feed


def post(url, body, pose):
    ctx = ssl.create_default_context()
    ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE  # self-signed, as on the phone after "visit anyway"
    req = urllib.request.Request(url + "/frame", data=body, method="POST",
                                 headers={"Content-Type": "image/jpeg", "X-Pose": json.dumps(pose)})
    return json.loads(urllib.request.urlopen(req, context=ctx, timeout=5).read())


class PhoneFeedTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.feed = phone_feed.PhoneFeed()
        self.url = self.feed.start(port=0, cert_dir=Path(self.tmp.name), host="127.0.0.1")
        self.assertIsNotNone(self.url, self.feed.error)
        self.url = "https://127.0.0.1:" + self.url.rsplit(":", 1)[1]

    def tearDown(self):
        self.feed.stop()
        self.tmp.cleanup()

    def test_page_frames_and_pose_over_https(self):
        ctx = ssl.create_default_context()
        ctx.check_hostname, ctx.verify_mode = False, ssl.CERT_NONE
        page = urllib.request.urlopen(self.url + "/", context=ctx, timeout=5).read()
        self.assertIn(b"chaser-1", page)
        st = post(self.url, b"\xff\xd8jpeg", {"lat": 40.8081, "lon": -73.9639, "alt_m": 12, "acc_m": 6,
                                                "heading": 92.4, "pitch": 7.5})
        self.assertEqual(st["frames"], 1)
        seq, jpeg = self.feed.wait_frame(0, timeout=1)
        self.assertEqual((seq, jpeg), (1, b"\xff\xd8jpeg"))
        pose = self.feed.live_pose()
        self.assertEqual(pose, {"cam_lat": 40.8081, "cam_lon": -73.9639, "cam_alt": 12.0, "yaw": 92.4, "pitch": 7.5})

    def test_no_gps_fix_means_no_position(self):
        post(self.url, b"x", {"heading": 10.0, "pitch": 0.0})  # compass works, GPS not yet
        self.assertIsNone(self.feed.live_pose())
        self.assertEqual(self.feed.status()["heading"], 10.0)

    def test_stale_pose_is_dropped(self):
        self.feed.put(b"x", {"lat": 1.0, "lon": 2.0, "heading": 3.0}, now=1000.0)
        self.assertIsNotNone(self.feed.live_pose(now=1001.0))
        self.assertIsNone(self.feed.live_pose(now=1000.0 + phone_feed.POSE_FRESH_S + 1))
