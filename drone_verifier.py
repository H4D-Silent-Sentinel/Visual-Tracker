"""Bounded, event-driven crop reviews; the tracking thread never waits for a VLM."""
import base64
import collections
import json
import math
import os
import threading
import time
import urllib.request
from urllib.parse import urlparse

import cv2


def crop_target(frame, box):
    if len(box) != 4 or not all(math.isfinite(v) for v in box):
        return None
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = box
    x1, x2 = max(0, int(x1)), min(w, math.ceil(x2))
    y1, y2 = max(0, int(y1)), min(h, math.ceil(y2))
    if x2 <= x1 or y2 <= y1:
        return None
    target = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(cv2.resize(target, (64, 64)), cv2.COLOR_BGR2GRAY)
    sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    size = max(x2 - x1, y2 - y1)
    px, py = math.ceil((x2-x1)*.25), math.ceil((y2-y1)*.25)
    crop = frame[max(0,y1-py):min(h,y2+py), max(0,x1-px):min(w,x2+px)]
    ch, cw = crop.shape[:2]
    scale = 384 / max(ch, cw)
    crop = cv2.resize(crop, (max(1, round(cw*scale)), max(1, round(ch*scale))),
                      interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
    return crop, size, sharp


def validate_verdict(d):
    if not isinstance(d, dict) or type(d.get('is_drone')) is not bool or type(d.get('calibrated')) is not bool:
        raise ValueError('Verifier must return a binary verdict and calibrated flag')
    for key in ('confidence', 'p_drone'):
        if type(d.get(key)) not in (int, float) or not math.isfinite(d[key]) or not 0 <= d[key] <= 1:
            raise ValueError('Invalid verifier probability')
    p = d['p_drone']
    if (p != .5 and d['is_drone'] != (p >= .5)) or abs(d['confidence'] - (p if d['is_drone'] else 1-p)) > .001:
        raise ValueError('Verifier probability and verdict disagree')
    return d


class DroneVerifier:
    def __init__(self, threshold=.85, url=None, *, clock=time.monotonic, request=None):
        if not .5 < threshold <= 1:
            raise ValueError('Verification confidence must be greater than 0.5 and at most 1')
        self.threshold = threshold
        self.url = url or os.environ.get('VLM_URL', 'http://127.0.0.1:8002/verify')
        parsed = urlparse(self.url)
        if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost', '::1') or parsed.username or parsed.password:
            raise ValueError('VLM_URL must be a loopback HTTP endpoint')
        self.clock, self.request = clock, request or self._request
        self.tracks = collections.OrderedDict()
        self.epoch = 0
        self.next_track = 0
        self.condition = threading.Condition()
        self.jobs = collections.deque()
        self.replies = collections.deque(maxlen=32)
        self.stopped = False
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _request(self, image):
        body = json.dumps({'image': base64.b64encode(image).decode()}).encode()
        req = urllib.request.Request(self.url, body, {'Content-Type': 'application/json'})
        # Never follow a redirect carrying a camera crop outside the local scorer.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        with urllib.request.build_opener(NoRedirect).open(req, timeout=20) as r:
            data = r.read(16385)
            if len(data) > 16384:
                raise ValueError('Verifier response is too large')
            return json.loads(data)

    def _run(self):
        while True:
            with self.condition:
                self.condition.wait_for(lambda: self.stopped or self.jobs)
                if self.stopped:
                    return
                token, image = self.jobs.popleft()
            epoch, tid, generation, revision = token
            track = self.tracks.get(tid)
            if epoch != self.epoch or not track or generation != track['generation'] or revision != track['revision']:
                continue
            try:
                result = dict(validate_verdict(self.request(image)))
                result['status'] = ('confirmed' if result['is_drone'] else 'rejected') if result['confidence'] >= self.threshold else 'uncertain'
            except Exception as e:
                result = {'status': 'error', 'error': str(e)[:180]}
            with self.condition:
                if not self.stopped:
                    self.replies.append((token, result))

    def reset(self):
        self.epoch += 1
        self.tracks.clear()
        with self.condition:
            self.jobs.clear()
            self.replies.clear()

    def close(self):
        with self.condition:
            self.stopped = True
            self.jobs.clear()
            self.condition.notify_all()

    def update(self, frame, targets):
        now = self.clock()
        visible_ids = {target.get('id') for target in targets}
        with self.condition:
            replies = list(self.replies)
            self.replies.clear()
        for (epoch, tid, generation, revision), result in replies:
            track = self.tracks.get(tid)
            if epoch == self.epoch and track and generation == track['generation'] and revision == track['revision']:
                track['verdict'] = result
                track['pending'] = False
        for tid, t in list(self.tracks.items()):
            if now - t['seen'] > 5:
                del self.tracks[tid]
        for target in targets:
            tid = target.get('id')
            if tid is None:
                target['verification'] = {'status': 'unverified', 'error': 'Waiting for a stable track ID'}
                continue
            t = self.tracks.get(tid)
            if t is None:
                if len(self.tracks) >= 32:
                    # shortcut: retain 32 candidates; raise this after profiling dense flocks on 8 GB.
                    dormant = next((key for key in self.tracks if key not in visible_ids), None)
                    if dormant is None:
                        target['verification'] = {'status': 'unverified', 'error': 'Candidate review capacity reached'}
                        continue
                    del self.tracks[dormant]
                self.next_track += 1
                t = self.tracks[tid] = dict(first=now, seen=now, generation=self.next_track, revision=0, crop=None, quality=-1,
                    size=0, sharp=0, review_size=0, review_sharp=0, reviewed=None, pending=False,
                    changed=None, verdict={'status': 'pending'})
            elif now-t['seen'] > 1:
                # A reappearing ID may now refer to another object; invalidate its old crop/reply.
                t.update(first=now, revision=t['revision']+1, crop=None, quality=-1,
                         reviewed=None, pending=False, changed=None, verdict={'status': 'pending'})
            t['seen'] = now
            self.tracks.move_to_end(tid)
            cropped = crop_target(frame, target['box'])
            if cropped is None or cropped[1] < 12:
                target['verification'] = {'status': 'unverified', 'error': 'Object is too small or outside the frame'}
                continue
            crop, size, sharp = cropped
            quality = size * max(sharp, 1)
            if quality > t['quality']:
                t.update(crop=crop, quality=quality, size=size, sharp=sharp)
            first_review = t['reviewed'] is None and now-t['first'] >= .5
            changed = t['reviewed'] is not None and (
                size >= 1.75*t['review_size'] or
                (sharp >= 2*t['review_sharp'] and sharp-t['review_sharp'] >= 30))
            if changed and (t['changed'] is None or quality > t['changed'][1]*max(t['changed'][2], 1)):
                t['changed'] = crop, size, sharp
            improved = t['changed'] is not None and now-t['reviewed'] >= 2
            if not t['pending'] and (first_review or improved):
                selected, review_size, review_sharp = (t['crop'], t['size'], t['sharp']) if first_review else t['changed']
                ok, buf = cv2.imencode('.jpg', selected, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if ok:
                    with self.condition:
                        if len(self.jobs) < 8 and not self.stopped:
                            t['revision'] += 1
                            token = self.epoch, tid, t['generation'], t['revision']
                            self.jobs.append((token, buf.tobytes()))
                            t.update(reviewed=now, review_size=review_size, review_sharp=review_sharp,
                                     pending=True, changed=None, verdict={'status': 'pending'})
                            self.condition.notify()
            target['verification'] = {'status': 'pending'} if t['changed'] is not None else dict(t['verdict'])
        return targets
