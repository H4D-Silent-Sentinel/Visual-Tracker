import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from PIL import Image

from dashboard import app


class CollectClassificationTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.name = 'ours_20261117_120000'
        self.key = 'ours__' + self.name
        self.ours = self.root / 'our_data'
        self.web = self.root / 'footage/web'
        for directory in (self.web, self.root / 'footage/ours', self.root / 'runs/thumbs'):
            directory.mkdir(parents=True)
        self.clip = {'name': self.name, 'group': 'ours', 'frames': 10, 'fps': 30,
                     'file': f'footage/ours/{self.name}.mp4'}
        (self.web / 'index.json').write_text(json.dumps({self.key: self.clip}))
        buffer = io.BytesIO()
        Image.new('RGB', (100, 50), 'gray').save(buffer, format='JPEG')
        self.jpeg = buffer.getvalue()
        for field, value in [('ROOT', self.root), ('OURS', self.ours), ('WEB', self.web),
                             ('THUMBS', self.root / 'runs/thumbs')]:
            self.start_patch(app, field, value)
        self.read = self.start_patch(app, 'read_frame', return_value=object())
        self.encode = self.start_patch(app.cv2, 'imencode', return_value=(True, SimpleNamespace(tobytes=lambda: self.jpeg)))
        self.start_patch(app.cv2, 'imwrite', side_effect=lambda path, *_: Path(path).write_bytes(self.jpeg) > 0)

    def start_patch(self, target, name, *args, **kwargs):
        patcher = patch.object(target, name, *args, **kwargs)
        value = patcher.start()
        self.addCleanup(patcher.stop)
        return value

    def annotation(self, **changes):
        return {'clip': self.key, 'frame': 2, 'boxes': [[.1, .2, .4, .6]],
                'object_label': 'bird', 'no_drone': False, 'split': 'test',
                'source_id': 'rooftop-session', **changes}

    def test_manual_label_routes_to_separate_ground_truth_and_legacy_api_stays_compatible(self):
        def request(route, body=None):
            handler = object.__new__(app.Handler)
            handler.path = route
            encoded = json.dumps(body).encode() if body is not None else b''
            handler.rfile = io.BytesIO(encoded)
            handler.headers = {'Content-Length': str(len(encoded))}
            handler.send_json = lambda response, status=200: response
            return handler.do_POST() if body is not None else handler.do_GET()

        result = request('/api/classification/label', self.annotation())
        self.assertTrue(result['ok'], result)
        self.assertEqual(result['annotation']['label'], 'bird')
        self.assertEqual(result['annotation']['source_id'], 'rooftop-session')
        self.read.assert_called_once_with(self.key, 2)
        self.assertFalse((self.ours / 'labels').exists())
        self.assertFalse((self.ours / 'images').exists())
        loaded = request('/api/labels?clip=' + self.key)
        self.assertEqual(loaded['labels'], {})
        self.assertEqual(loaded['classifications']['2']['label'], 'bird')
        legacy = request('/api/label', {'clip': self.key, 'frame': 3, 'boxes': [[.1, .2, .4, .6]]})
        self.assertTrue(legacy['ok'])
        loaded = request('/api/labels?clip=' + self.key)
        self.assertIn('3', loaded['labels'])
        self.assertNotIn('3', loaded['classifications'])
        self.assertFalse(request('/api/classification/label', [])['ok'])

    def test_bounds_are_checked_before_read_frame_can_clamp_the_request(self):
        for frame in (-1, 10, True, 1.5, '1', None):
            with self.subTest(frame=frame):
                self.assertFalse(app.classification_label_save(self.annotation(frame=frame))['ok'])
        self.assertFalse(app.classification_label_save(self.annotation(clip='../outside'))['ok'])
        self.read.assert_not_called()

    def test_true_label_source_split_and_explicit_background_are_required(self):
        invalid = [{'object_label': ''}, {'object_label': 'invented'}, {'source_id': ''},
                   {'source_id': 'x' * 81}, {'split': 'train'}, {'boxes': [], 'no_drone': False},
                   {'object_label': 'drone', 'no_drone': True}]
        for changes in invalid:
            with self.subTest(changes=changes):
                result = app.classification_label_save(self.annotation(**changes))
                self.assertFalse(result['ok'], result)
        self.assertEqual(app.classification_data.load_annotations(self.ours, self.key), {})
        background = app.classification_label_save(self.annotation(boxes=[], object_label=None, no_drone=True))
        self.assertTrue(background['ok'], background)
        self.assertIsNone(background['annotation']['label'])
        self.assertTrue(background['annotation']['no_drone'])
        self.assertEqual(background['annotation']['boxes'], [])
        self.assertFalse(app.classification_label_save(self.annotation(frame=3, split='development'))['ok'])

    def test_unreadable_or_unencodable_frame_cannot_create_ground_truth(self):
        self.read.return_value = None
        self.assertFalse(app.classification_label_save(self.annotation())['ok'])
        self.encode.assert_not_called()
        self.read.return_value = object()
        self.encode.return_value = (False, None)
        self.assertFalse(app.classification_label_save(self.annotation())['ok'])
        self.assertEqual(app.classification_data.load_annotations(self.ours, self.key), {})

    def test_collect_summary_counts_classifications_without_adding_drone_training_frames(self):
        self.assertTrue(app.classification_label_save(self.annotation())['ok'])
        self.start_patch(app, 'camera_list', return_value=[])
        self.start_patch(app.CAM, 'state', return_value={})
        payload = app.collect_payload()
        self.assertEqual(len(payload['class_labels']), 35)
        self.assertEqual(payload['classifications'][self.key], 1)
        self.assertEqual(payload['classification_dataset']['labels'], {'bird': 1})
        self.assertEqual(payload['classification_dataset']['test'], 1)
        self.assertEqual(payload['dataset']['frames'], 0)
        self.assertEqual(payload['recordings'][0]['classified'], 1)
        for frame in (2, 3):
            self.assertTrue(app.label_save({'clip': self.key, 'frame': frame, 'boxes': [[.1, .2, .4, .6]]})['ok'])
        payload = app.collect_payload()
        self.assertEqual(payload['labeled'][self.key], 2, 'A class label and legacy label on the same frame count once')
        self.assertEqual(payload['recordings'][0]['labeled'], 2)
        self.assertEqual(payload['coverage'][0]['labeled_frames'], 2)

    def test_export_is_a_snapshot_and_recording_delete_preserves_it(self):
        self.assertTrue(app.classification_label_save(self.annotation())['ok'])
        result = app.collection_export({'name': 'rooftop_test'})
        self.assertTrue(result['ok'], result)
        manifest = Path(result['test_manifest'])
        row = json.loads(manifest.read_text())
        image = manifest.parent / row['image']
        self.assertEqual(row['expected_label'], 'bird')
        self.assertEqual(row['source_id'], 'rooftop-session')
        self.assertEqual(image.read_bytes(), self.jpeg)

        self.assertIn('scripts/evaluate/test_classification.py', result['command'])
        self.assertFalse(app.collection_export({'name': 'rooftop_test'})['ok'])
        self.assertTrue(app.classification_label_save({'clip': self.key, 'frame': 2, 'delete': True})['ok'])
        self.assertTrue(image.exists())
        self.assertTrue(app.classification_label_save(self.annotation())['ok'])
        raw = self.root / self.clip['file']
        raw.write_bytes(b'raw recording fixture')
        deleted = app.recording_delete(self.name)
        self.assertTrue(deleted['ok'], deleted)
        self.assertFalse(raw.exists())
        self.assertFalse((self.ours / 'classification/labels' / self.key).exists())
        self.assertFalse((self.ours / 'classification/images' / self.key).exists())
        self.assertTrue(manifest.exists())
        self.assertEqual(image.read_bytes(), self.jpeg)


    def test_recording_writes_raw_frames_without_running_the_preview_model(self):
        camera = app.Camera()
        camera.overlay = True
        camera.start_rec({'source_id': 'recording-session'})
        frame = SimpleNamespace(shape=(50, 100, 3))
        capture = Mock()
        capture.isOpened.return_value = True
        capture.get.return_value = 30
        capture.read.side_effect = [(True, frame), StopIteration]
        writer = Mock()
        self.start_patch(app.cv2, 'VideoCapture', return_value=capture)
        self.start_patch(app.cv2, 'VideoWriter', return_value=writer)
        model = self.start_patch(app, 'load_model')
        with self.assertRaises(StopIteration):
            camera.run()
        writer.write.assert_called_once_with(frame)
        model.assert_not_called()
        self.assertEqual(camera.rec['frames'], 1)


if __name__ == '__main__':
    unittest.main()
