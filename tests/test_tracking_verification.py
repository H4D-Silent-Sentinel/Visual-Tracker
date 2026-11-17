"""Exercise the actual live loop with a finite video and deterministic reviews."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import unittest
from unittest.mock import patch
from unittest.mock import Mock

import numpy as np


class EndVideo(Exception):
    pass


class LiveIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch = SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False),
                                backends=SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False)))
        spec = importlib.util.spec_from_file_location('test_live_engine', Path(__file__).resolve().parents[1] / 'dashboard/live_engine.py')
        cls.live = importlib.util.module_from_spec(spec)
        with patch.dict(sys.modules, {'torch': torch, 'ultralytics': SimpleNamespace(YOLO=object)}), \
             patch('subprocess.run', return_value=SimpleNamespace(stderr='')):
            spec.loader.exec_module(cls.live)

    def run_frame(self, verdicts, enabled=True):
        image = np.zeros((160, 240, 3), np.uint8)
        boxes = [SimpleNamespace(xyxy=[np.array([20+i*60, 30, 55+i*60, 65])], conf=.9-i*.1, id=i+1, cls=0)
                 for i in range(len(verdicts))]
        model = SimpleNamespace(predictor=None, track=lambda *a, **k: [SimpleNamespace(boxes=boxes, names={0:'drone'})])
        class Capture:
            count = 0
            def read(self):
                self.count += 1
                if self.count > 1:
                    raise EndVideo()
                return True, image
            def get(self, *_): return 1
            def release(self): pass
        class Reviews:
            def update(self, frame, targets):
                for target, verdict in zip(targets, verdicts):
                    target['verification'] = dict(verdict)
            def reset(self): pass
            def close(self): pass
        with patch.object(self.live, 'list_sources', return_value=['clip.mp4']):
            engine = self.live.Engine()
        engine.settings['verify'] = enabled
        engine.viewers = 1
        engine.open_source = lambda _: (Capture(), 30, False, 1)
        engine.get_model = lambda _: model
        with patch.object(self.live, 'DroneVerifier', return_value=Reviews()), self.assertRaises(EndVideo):
            engine.run()
        return engine.stats

    def test_rejected_candidate_never_counts_as_lock(self):
        stats = self.run_frame([dict(status='rejected', is_drone=False, confidence=.99, p_drone=.01, calibrated=False)])
        self.assertEqual(stats['status'], 'FALSE_REVIEW')
        self.assertIsNone(stats['primary'])
        self.assertEqual(stats['locked_pct'], 0)
        self.assertEqual(stats['trail'], [])

    def test_confirmed_secondary_target_wins_over_high_confidence_rejection(self):
        stats = self.run_frame([dict(status='rejected'), dict(status='confirmed', is_drone=True, confidence=.95, p_drone=.95, calibrated=True)])
        self.assertEqual(stats['status'], 'LOCKED')
        self.assertEqual(stats['primary']['id'], 2)
        self.assertEqual(stats['candidate']['id'], 2)
        self.assertEqual(stats['verification']['status'], 'confirmed')
        self.assertEqual(stats['locked_pct'], 100)

    def test_pending_uncertain_and_error_remain_unverified(self):
        for status, expected in [('pending','REVIEWING'), ('uncertain','UNVERIFIED'), ('error','VERIFY_ERROR')]:
            with self.subTest(status=status):
                stats = self.run_frame([dict(status=status)])
                self.assertEqual(stats['status'], expected)
                self.assertEqual(stats['locked_pct'], 0)
                self.assertIsNone(stats['primary'])

    def test_detector_only_mode_preserves_existing_lock_behavior(self):
        stats = self.run_frame([dict(status='rejected')], enabled=False)
        self.assertEqual(stats['status'], 'LOCKED')
        self.assertEqual(stats['primary']['id'], 1)
        self.assertFalse(stats['verification']['enabled'])

    def test_verification_toggle_invalidates_live_session(self):
        with patch.object(self.live, 'list_sources', return_value=['clip.mp4']):
            engine = self.live.Engine()
        engine.restart = False
        engine.update(verify='true')
        self.assertTrue(engine.settings['verify'])
        self.assertTrue(engine.restart)

    def test_sources_request_does_not_open_camera_or_load_model_without_viewers(self):
        with patch.object(self.live, 'list_sources', return_value=['camera:0']):
            engine = self.live.Engine()
        engine.open_source = Mock()
        engine.get_model = Mock()
        with patch.object(self.live.time, 'sleep', side_effect=EndVideo), self.assertRaises(EndVideo):
            engine.run()
        engine.open_source.assert_not_called()
        engine.get_model.assert_not_called()
        self.assertEqual(engine.stats['status'], 'IDLE')
        self.assertEqual(engine.stats['fps'], 0)
        self.assertEqual(engine.stats['targets'], [])
        self.assertIsNone(engine.stats['primary'])


if __name__ == '__main__':
    unittest.main()
