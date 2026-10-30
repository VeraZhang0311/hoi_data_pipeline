from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

BODY_NAMES = tuple("pelvis imu_in_pelvis left_hip_pitch_link left_hip_roll_link left_hip_yaw_link left_knee_link left_ankle_pitch_link left_ankle_roll_link pelvis_contour_link right_hip_pitch_link right_hip_roll_link right_hip_yaw_link right_knee_link right_ankle_pitch_link right_ankle_roll_link waist_yaw_link waist_roll_link torso_link d435_link head_link imu_in_torso left_shoulder_pitch_link left_shoulder_roll_link left_shoulder_yaw_link left_elbow_link left_wrist_roll_link left_wrist_pitch_link left_wrist_yaw_link left_rubber_hand logo_link mid360_link right_shoulder_pitch_link right_shoulder_roll_link right_shoulder_yaw_link right_elbow_link right_wrist_roll_link right_wrist_pitch_link right_wrist_yaw_link right_rubber_hand".split())

def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def jsonable(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, Path): return str(value)
    if isinstance(value, dict): return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [jsonable(v) for v in value]
    return value

def canonical_json(value):
    return json.dumps(jsonable(value), sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))

def identity(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()

def rotation(wxyz):
    q = np.asarray(wxyz, dtype=float)
    if q.shape[-1] != 4 or not np.isfinite(q).all() or np.max(np.abs(np.linalg.norm(q, axis=-1)-1)) > 1e-3:
        raise ValueError("Expected finite unit WXYZ quaternion")
    return Rotation.from_quat(q[..., [1,2,3,0]])

def transform(pos, quat):
    out = np.eye(4)
    out[:3,:3] = rotation(quat).as_matrix()
    out[:3,3] = pos
    return out

def points_transform(points, matrix):
    return np.asarray(points) @ matrix[:3,:3].T + matrix[:3,3]
