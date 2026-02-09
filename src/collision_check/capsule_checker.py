"""Capsule-based self-collision checker for robotic manipulators.

Models each link as a capsule (line segment + radius) and checks
minimum distance between non-adjacent link pairs. More accurate than
the sphere-based CollisionChecker for runtime safety checking.
"""

import numpy as np


def _segment_distance(
    p1: np.ndarray, p2: np.ndarray,
    p3: np.ndarray, p4: np.ndarray,
) -> float:
    """Minimum distance between line segment p1-p2 and segment p3-p4."""
    d1 = p2 - p1
    d2 = p4 - p3
    r = p1 - p3

    a = float(np.dot(d1, d1))
    e = float(np.dot(d2, d2))
    f = float(np.dot(d2, r))

    EPS = 1e-12

    if a <= EPS and e <= EPS:
        return float(np.linalg.norm(r))

    if a <= EPS:
        return float(np.linalg.norm(r - np.clip(f / e, 0, 1) * d2))

    c = float(np.dot(d1, r))

    if e <= EPS:
        s = np.clip(-c / a, 0, 1)
        return float(np.linalg.norm(p1 + s * d1 - p3))

    b = float(np.dot(d1, d2))
    denom = a * e - b * b

    s = np.clip((b * f - c * e) / denom, 0, 1) if abs(denom) > EPS else 0.0
    t = (b * s + f) / e

    if t < 0:
        t = 0.0
        s = np.clip(-c / a, 0, 1)
    elif t > 1:
        t = 1.0
        s = np.clip((b - c) / a, 0, 1)

    return float(np.linalg.norm((p1 + s * d1) - (p3 + t * d2)))


class CapsuleCollisionChecker:
    """Runtime self-collision checker using capsule approximation.

    Unlike CollisionChecker (sphere-based, for optimization constraints),
    this class models links as capsules (line segments + radii) for more
    accurate runtime safety checking.

    Pinocchio joint layout for UR5e (njoints=7):
        oMi[0] = universe, oMi[1] = shoulder_pan, oMi[2] = shoulder_lift,
        oMi[3] = elbow, oMi[4] = wrist_1, oMi[5] = wrist_2, oMi[6] = wrist_3

    Segment k connects oMi[k] to oMi[k+1]  (k = 0 .. njoints-2).
    """

    # Capsule radius per segment index [m] (conservative for UR5e)
    _RADII = [
        0.085,  # seg 0: base_link -> shoulder_pan
        0.075,  # seg 1: shoulder_pan -> shoulder_lift  (zero-length)
        0.065,  # seg 2: upper_arm (shoulder_lift -> elbow)
        0.055,  # seg 3: forearm (elbow -> wrist_1)
        0.045,  # seg 4: wrist_1 -> wrist_2
        0.045,  # seg 5: wrist_2 -> wrist_3
    ]

    # Segment pairs to exclude from collision checking.
    _EXCLUDE_PAIRS: frozenset[tuple[int, int]] = frozenset({(0, 2), (3, 5)})

    def __init__(
        self,
        model,
        data,
        safety_margin: float = 0.01,
        min_gap: int = 2,
    ):
        import pinocchio as pin

        self._pin = pin
        self.model = model
        self.data = data
        self.n_seg = model.njoints - 1

        # Build pair list: (seg_i, seg_j, clearance_threshold)
        self.pairs: list[tuple[int, int, float]] = []
        for i in range(self.n_seg):
            for j in range(i + min_gap, self.n_seg):
                if (i, j) in self._EXCLUDE_PAIRS:
                    continue
                ri = self._RADII[i] if i < len(self._RADII) else 0.04
                rj = self._RADII[j] if j < len(self._RADII) else 0.04
                self.pairs.append((i, j, ri + rj + safety_margin))

    def check(self, q: np.ndarray) -> tuple[bool, float, str]:
        """Check for self-collision at joint configuration *q*.

        Returns:
            (collision, min_clearance, detail_message)
            *collision* is True when any pair is closer than its threshold.
        """
        q = np.asarray(q, dtype=np.float64).ravel()
        self._pin.forwardKinematics(self.model, self.data, q)

        pos = [self.data.oMi[i].translation.copy()
               for i in range(self.model.njoints)]

        min_clearance = float("inf")
        collision = False
        detail = ""

        for i, j, thresh in self.pairs:
            dist = _segment_distance(pos[i], pos[i + 1], pos[j], pos[j + 1])
            if dist < min_clearance:
                min_clearance = dist

            if dist < thresh:
                collision = True
                detail = (
                    f"Self-collision: seg {i} <-> seg {j}, "
                    f"dist={dist:.4f} m < threshold={thresh:.4f} m"
                )
                return collision, min_clearance, detail

        return collision, min_clearance, detail
