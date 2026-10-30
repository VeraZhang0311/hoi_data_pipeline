"""Time interpolation and interval evidence, never binary-label interpolation."""
from __future__ import annotations
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Slerp
from .common import rotation

POSE_KEYS = ("root_pos", "root_quat", "joint_pos", "object_pos", "object_quat")

def validate(clip):
    t = np.asarray(clip["timestamps_s"], dtype=np.float64)
    if t.ndim != 1 or len(t) < 2 or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("Time must be finite, strictly increasing, at least two states")
    for key, width in zip(POSE_KEYS, (3,4,29,3,4)):
        value = np.asarray(clip[key], dtype=float)
        if value.shape != (len(t),width) or not np.isfinite(value).all():
            raise ValueError(f"Invalid {key} shape or values")
        if key.endswith("quat"): rotation(value)
    return t

def resample(clip, hz=50.0):
    t = validate(clip)
    if not np.isfinite(hz) or hz <= 0: raise ValueError("Invalid target Hz")
    count = int(np.floor((t[-1]-t[0])*hz + 1e-8)) + 1
    if count < 2: raise ValueError("Reference shorter than one control interval")
    target = t[0] + np.arange(count, dtype=float)/hz
    # Do not stretch the clock to force the original endpoint onto a 50 Hz grid.
    out = {"timestamps_s": target}
    for key in POSE_KEYS:
        if key.endswith("quat"):
            q = Slerp(t, rotation(clip[key]))(np.minimum(target, t[-1])).as_quat()
            out[key] = q[:,[3,0,1,2]]
        else:
            out[key] = interp1d(t, clip[key], axis=0)(np.minimum(target,t[-1]))
    out["timebase"] = {"target_hz": hz, "source_start_s": float(t[0]), "source_end_s": float(t[-1]),
                       "target_end_s": float(target[-1]), "tail_not_on_grid_s": max(0.,float(t[-1]-target[-1])),
                       "interpolation": "linear_position_joint_and_shortest_arc_quaternion_slerp",
                       "state_count": count, "interval_count": count-1}
    return out

def sample_edges(t):
    """Nearest-sample temporal support; ends clipped to observed duration."""
    t = np.asarray(t)
    return np.r_[t[0], (t[:-1]+t[1:])*0.5, t[-1]]

def events(t, labels):
    edges = sample_edges(t)
    result = []
    for channel in range(labels.shape[1]):
        active = labels[:,channel] == 1
        changes = np.diff(np.r_[False,active,False].astype(np.int8))
        for a,b in zip(np.flatnonzero(changes==1),np.flatnonzero(changes==-1)):
            result.append({"channel": channel, "start_s": float(edges[a]), "end_s": float(edges[b]),
                           "first_state": int(a), "last_state": int(b-1), "temporal_resolution_s": float(np.max(np.diff(t[max(0,a-1):min(len(t),b+1)])))})
    return result

def interval_evidence(source_t, source_labels, target_t):
    """Duration fractions from source sample support, including sub-control-tick events.

    Fractions are observed temporal coverage, not calibrated contact probability.
    Unknown samples do not become negatives. No event is extended across a gap.
    """
    edges = sample_edges(source_t)
    n, c = len(target_t)-1, source_labels.shape[1]
    positive = np.zeros((n,c),np.float32)
    known = np.zeros((n,c),np.float32)
    for i,(start,end) in enumerate(zip(target_t[:-1],target_t[1:])):
        a = max(0,np.searchsorted(edges,start,side="right")-1)
        b = min(len(source_t),np.searchsorted(edges,end,side="left")+1)
        overlap = np.maximum(0,np.minimum(end,edges[a+1:b+1])-np.maximum(start,edges[a:b]))/(end-start)
        positive[i] = overlap @ (source_labels[a:b] == 1)
        known[i] = overlap @ (source_labels[a:b] >= 0)
    return positive,known
