import threading
import time
import unittest

import cv2
import numpy as np

from drone_verifier import DroneVerifier, crop_target, validate_verdict


class VerifierTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.image = np.random.default_rng(1).integers(0, 255, (240, 320, 3), dtype=np.uint8)
        self.calls = []
        self.reply = dict(is_drone=True, confidence=.96, p_drone=.96, calibrated=False)
        self.verifier = DroneVerifier(clock=lambda: self.now, request=self.request)
        self.addCleanup(self.verifier.close)

    def request(self, image):
        self.calls.append(image)
        return dict(self.reply)

    def update(self, tid=1, box=(40, 40, 80, 80)):
        targets = [dict(id=tid, box=list(box), conf=.8, label='drone')]
        self.verifier.update(self.image, targets)
        return targets[0]['verification']

    def wait_reply(self, status='confirmed'):
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline:
            v = self.update()
            if v['status'] == status:
                return v
            time.sleep(.005)
        self.fail('worker did not return ' + status)

    def test_new_track_waits_then_reviews_once_despite_motion_or_elapsed_time(self):
        self.assertEqual(self.update()['status'], 'pending')
        self.assertEqual(self.calls, [])
        self.now = .6
        self.update()
        self.wait_reply()
        for n in range(300):
            self.now += .1
            self.update(box=(40+n%30, 40, 80+n%30, 80))
        self.assertEqual(len(self.calls), 1)

    def test_confident_negative_is_rejected_and_uncertain_is_not(self):
        self.reply = dict(is_drone=False, confidence=.98, p_drone=.02, calibrated=True)
        self.update()
        self.now = .6
        self.update()
        self.assertFalse(self.wait_reply('rejected')['is_drone'])
        self.assertEqual(len(self.calls), 1)
        self.verifier.reset()
        self.reply = dict(is_drone=False, confidence=.6, p_drone=.4, calibrated=False)
        self.update()
        self.now += .6
        self.update()
        self.wait_reply('uncertain')

    def test_larger_view_triggers_update_but_jitter_does_not(self):
        self.update()
        self.now = .6
        self.update()
        self.wait_reply()
        for _ in range(24):
            self.now += .1
            self.update()
        self.update(box=(40, 40, 82, 82))
        self.assertEqual(len(self.calls), 1)
        self.update(box=(40, 40, 130, 130))
        deadline = time.monotonic() + 2
        while len(self.calls) < 2 and time.monotonic() < deadline:
            time.sleep(.005)
        self.assertEqual(len(self.calls), 2)

    def test_slow_review_does_not_block_and_reset_discards_old_reply(self):
        started, release = threading.Event(), threading.Event()
        def slow(image):
            started.set()
            release.wait(2)
            return dict(self.reply)
        self.verifier.request = slow
        self.update()
        self.now = .6
        before = time.monotonic()
        self.update()
        self.assertLess(time.monotonic() - before, .15)
        self.assertTrue(started.wait(1))
        self.verifier.reset()
        self.now = .7
        self.assertEqual(self.update()['status'], 'pending')
        release.set()
        time.sleep(.04)
        self.assertEqual(self.update()['status'], 'pending')

    def test_brief_improvement_while_review_pending_is_preserved(self):
        started, release = threading.Event(), threading.Event()
        def reviewing(image):
            self.calls.append(image)
            if len(self.calls) == 1:
                started.set()
                release.wait(2)
                return dict(is_drone=False, confidence=.97, p_drone=.03, calibrated=False)
            return dict(self.reply)
        self.verifier.request = reviewing
        self.update()
        self.now = .6
        self.update()
        self.assertTrue(started.wait(1))
        for _ in range(24):
            self.now += .1
            self.update()
        self.assertEqual(self.update(box=(40, 40, 130, 130))['status'], 'pending')
        self.update()  # the clearer view has already disappeared
        release.set()
        self.wait_reply()
        self.assertEqual(len(self.calls), 2)
        self.assertEqual(self.verifier.tracks[1]['review_size'], 90)

    def test_independent_ids_and_expiry_do_not_share_verdicts(self):
        self.update()
        self.now = .6
        self.update()
        self.wait_reply()
        self.assertEqual(self.update(tid=2)['status'], 'pending')
        self.now = 10
        self.verifier.update(self.image, [])
        self.assertEqual(self.update()['status'], 'pending')

    def test_missing_id_and_tiny_image_do_not_call_vlm(self):
        self.assertEqual(self.update(tid=None)['status'], 'unverified')
        self.update(box=(40, 40, 45, 45))
        self.now = 1
        self.assertEqual(self.update(box=(40, 40, 45, 45))['status'], 'unverified')
        self.assertEqual(self.calls, [])

    def test_queue_and_cache_stay_bounded(self):
        for tid in range(100):
            self.update(tid=tid)
        self.assertLessEqual(len(self.verifier.tracks), 32)

    def test_dense_scene_keeps_admitted_tracks_instead_of_churning_cache(self):
        targets = [dict(id=i, box=[40, 40, 80, 80]) for i in range(33)]
        self.verifier.update(self.image, targets)
        admitted = set(self.verifier.tracks)
        self.now = .6
        self.verifier.update(self.image, targets)
        self.assertEqual(set(self.verifier.tracks), admitted)
        self.assertEqual(targets[-1]['verification']['status'], 'unverified')
        deadline = time.monotonic()+2
        while not self.calls and time.monotonic()<deadline:
            time.sleep(.005)
        self.assertTrue(self.calls)

    def test_failure_is_error_and_not_a_false_review(self):
        def failed(image):
            raise OSError('service offline')
        self.verifier.request = failed
        self.update()
        self.now = .6
        self.update()
        v = self.wait_reply('error')
        self.assertIn('service offline', v['error'])
        self.assertNotIn('is_drone', v)


class CropAndResponseTests(unittest.TestCase):
    def test_crop_pads_clamps_preserves_shape_and_caps_size(self):
        frame = np.zeros((800, 1200, 3), np.uint8)
        crop, size, sharp = crop_target(frame, [-10, 10, 600, 300])
        self.assertLessEqual(max(crop.shape[:2]), 384)
        self.assertGreater(crop.shape[1], crop.shape[0])
        self.assertEqual(size, 600)
        self.assertEqual(sharp, 0)
        self.assertIsNone(crop_target(frame, [40, 40, 30, 30]))
        self.assertIsNone(crop_target(frame, [0, 0, float('nan'), 50]))

    def test_verdict_requires_real_binary_and_consistent_finite_probability(self):
        tied = dict(is_drone=False, confidence=.5, p_drone=.5, calibrated=False)
        self.assertEqual(validate_verdict(tied), tied)
        for bad in [
            dict(is_drone='false', confidence=.95, p_drone=.05, calibrated=False),
            dict(is_drone=False, confidence=float('nan'), p_drone=.05, calibrated=False),
            dict(is_drone=True, confidence=.99, p_drone=.1, calibrated=False),
            dict(is_drone=False, confidence=.9, p_drone=2, calibrated=False),
        ]:
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_verdict(bad)


if __name__ == '__main__':
    unittest.main()
