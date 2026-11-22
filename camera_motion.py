"""Estimate how the whole picture moved between two frames (camera pan, tilt, shake, zoom).

Tracks a few hundred background corners from the previous frame to this one (sparse optical flow),
skipping the drone's own box, and fits one shift + rotation + scale (a 2x3 affine) with RANSAC so
moving objects don't skew it. Used to keep the drone's trail on the drone's real path while the
camera moves. Returns None when the picture has too little texture (e.g. plain sky) to tell.
"""
import cv2
import numpy as np


class CameraMotion:
    def __init__(self, work_width=640, max_corners=300, min_points=12):
        self.work_width = work_width
        self.max_corners = max_corners
        self.min_points = min_points
        self.prev = None
        self.scale = 1.0
        self.last_points = 0

    def reset(self):
        self.prev = None

    def update(self, frame, boxes=()):
        """frame: BGR image. boxes: [x1, y1, x2, y2] in full-frame pixels to ignore (moving targets).
        Returns a 2x3 matrix mapping previous-frame pixels to this frame's pixels, or None."""
        h, w = frame.shape[:2]
        self.scale = min(1.0, self.work_width / w)
        small = cv2.resize(frame, (int(w * self.scale), int(h * self.scale))) if self.scale < 1 else frame
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        prev, self.prev = self.prev, gray
        if prev is None or prev.shape != gray.shape:
            return None
        mask = np.full(gray.shape, 255, np.uint8)
        for x1, y1, x2, y2 in boxes:
            pad = 0.25 * max(x2 - x1, y2 - y1)
            cv2.rectangle(mask, (int((x1 - pad) * self.scale), int((y1 - pad) * self.scale)),
                          (int((x2 + pad) * self.scale), int((y2 + pad) * self.scale)), 0, -1)
        p0 = cv2.goodFeaturesToTrack(prev, self.max_corners, 0.01, 8, mask=mask)
        if p0 is None or len(p0) < self.min_points:
            self.last_points = 0
            return None
        p1, ok, _ = cv2.calcOpticalFlowPyrLK(prev, gray, p0, None, winSize=(21, 21), maxLevel=3)
        good = ok.reshape(-1) == 1
        a, b = p0[good], p1[good]
        self.last_points = int(good.sum())
        if len(a) < self.min_points:
            return None
        m, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if m is None or inliers is None or inliers.sum() < self.min_points:
            return None
        m[:, 2] /= self.scale  # back to full-resolution pixels (rotation/scale part is unchanged)
        return m


def apply(m, points):
    """Move (x, y) pixel points by a 2x3 affine."""
    if m is None or not points:
        return list(points)
    p = np.asarray(points, np.float64)
    return [tuple(v) for v in (p @ m[:, :2].T + m[:, 2])]
