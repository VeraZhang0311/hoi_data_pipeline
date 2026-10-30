"""Read GRAIL G1 sitting motion libraries without resampling or contact labelling.

The upstream retarget writer stores robot quaternions WXYZ, whereas its rollout
exporter stores XYZW.  We resolve that version difference by checking pose_aa's
independent root axis-angle, never by guessing from quaternion magnitudes.
Object quaternions are WXYZ in both writers.  Hand points remain source-derived
annotations, not forces and not revalidated robot contacts.
Released USDs retain source dimensions: the author's replay code applies
objects.pkl's static scale once, after composing mesh-to-rigid-root transforms.
"""
from __future__ import annotations

import hashlib
import io
import math
from pathlib import Path
import pickle
from typing import Mapping
import zlib

import numpy as np

UPSTREAM_COMMIT = "da382dcac447261456314c3e3ce15401d5dc1eb0"
UPSTREAM_BASE = f"https://github.com/NVlabs/GRAIL/blob/{UPSTREAM_COMMIT}/"
CONVENTION_SOURCES = {
    "retarget": UPSTREAM_BASE + "grail/retargeting/retarget.py#L769-L850",
    "rollout_export": UPSTREAM_BASE + "grail/data_export/export_successful_rollouts.py#L262-L391",
    "object_spawn": UPSTREAM_BASE + "imports/SONIC/gear_sonic/envs/manager_env/modular_tracking_env_cfg.py#L556-L1001",
    "released_usd_scale": UPSTREAM_BASE + "grail/visualization/prepare_vis_shard.py#L236-L251",
    "released_usd_replay": UPSTREAM_BASE + "grail/visualization/batch_render_replay.py#L583-L638",
    "joint_order": UPSTREAM_BASE + "imports/SONIC/gear_sonic/envs/manager_env/robots/g1.py#L31-L126",
}

# MuJoCo/Unitree order, identical to supermimic.robot.G1_JOINT_NAMES.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
_MAX_BYTES = 256 * 1024 * 1024


class GrailFormatError(ValueError):
    """A source violates the supported, explicitly checked GRAIL contract."""


class _ArrayDescriptor:
    pass


class _ArrayType:
    """An inert replacement for numpy.ndarray when named inside a pickle."""


def _numeric_dtype(value):
    dtype = np.dtype(value)
    if dtype.hasobject or dtype.fields is not None or dtype.kind not in "biufc":
        raise GrailFormatError("Only unstructured numeric NumPy arrays are accepted")
    return dtype


def _size(shape, dtype):
    if not isinstance(shape, tuple) or len(shape) > 8:
        raise GrailFormatError("Invalid or over-dimensional numeric array")
    if any(type(x) is not int or x < 0 for x in shape):
        raise GrailFormatError("Invalid numeric array shape")
    size = math.prod(shape) * _numeric_dtype(dtype).itemsize
    if size > _MAX_BYTES:
        raise GrailFormatError("Numeric array exceeds the allocation limit")
    return size


def _reconstruct(subtype, shape, typecode):
    if subtype is not _ArrayType or shape != (0,):
        raise GrailFormatError("Unsupported ndarray reconstruction")
    return np.empty(0, dtype=np.uint8)


def _scalar(dtype, data):
    dtype = _numeric_dtype(dtype)
    if not isinstance(data, bytes) or len(data) != dtype.itemsize:
        raise GrailFormatError("Malformed NumPy scalar")
    return np.frombuffer(data, dtype=dtype)[0]


def _frombuffer(data, dtype, shape, order):
    dtype = _numeric_dtype(dtype)
    if not isinstance(data, (bytes, bytearray)) or len(data) != _size(shape, dtype):
        raise GrailFormatError("Malformed NumPy buffer")
    if order not in ("C", "F"):
        raise GrailFormatError("Invalid array order")
    return np.frombuffer(data, dtype=dtype).reshape(shape, order=order).copy()


class _NumericUnpickler(pickle._Unpickler):
    """Restricted pickle/joblib numeric reader; never imports a pickle global.

    Supports protocol 4/5 NumPy arrays and joblib's uncompressed numeric-array
    wrapper, including alignment and Fortran order. Object arrays, arbitrary
    constructors, persistent IDs and external buffers are rejected.
    """
    def find_class(self, module, name):
        if (module, name) == ("joblib.numpy_pickle", "NumpyArrayWrapper"):
            return _ArrayDescriptor
        if (module, name) == ("numpy", "ndarray"):
            return _ArrayType
        if (module, name) == ("numpy", "dtype"):
            return np.dtype
        if module in ("numpy.core.multiarray", "numpy._core.multiarray"):
            if name == "_reconstruct": return _reconstruct
            if name == "scalar": return _scalar
        if module in ("numpy.core.numeric", "numpy._core.numeric") and name == "_frombuffer":
            return _frombuffer
        raise GrailFormatError(f"Forbidden pickle global: {module}.{name}")

    def persistent_load(self, pid):
        raise GrailFormatError("Persistent pickle references are forbidden")

    def load_reduce(self):
        args = self.stack.pop()
        fn = self.stack[-1]
        if fn not in (np.dtype, _reconstruct, _scalar, _frombuffer):
            raise GrailFormatError("Forbidden pickle constructor")
        self.stack[-1] = fn(*args)

    def load_newobj(self):
        args = self.stack.pop()
        cls = self.stack.pop()
        if cls is not _ArrayDescriptor or args:
            raise GrailFormatError("Forbidden pickle NEWOBJ")
        self.stack.append(_ArrayDescriptor())

    def load_build(self):
        obj, state = self.stack[-2], self.stack[-1]
        if isinstance(obj, np.ndarray):
            if not isinstance(state, tuple) or len(state) != 5:
                raise GrailFormatError("Invalid NumPy array state")
            _, shape, dtype, fortran, raw = state
            dtype = _numeric_dtype(dtype)
            if type(fortran) is not bool or not isinstance(raw, bytes) or len(raw) != _size(shape, dtype):
                raise GrailFormatError("Invalid NumPy array state/buffer length")
        elif not isinstance(obj, (np.dtype, _ArrayDescriptor)):
            raise GrailFormatError("Forbidden pickle BUILD target")
        super().load_build()
        obj = self.stack[-1]
        if not isinstance(obj, _ArrayDescriptor):
            return
        if set(vars(obj)) - {"subclass", "shape", "order", "dtype", "allow_mmap", "numpy_array_alignment_bytes"}:
            raise GrailFormatError("Unknown joblib array wrapper fields")
        if obj.subclass is not _ArrayType or obj.order not in ("C", "F"):
            raise GrailFormatError("Unsupported joblib array subclass/order")
        dtype = _numeric_dtype(obj.dtype)
        nbytes = _size(obj.shape, dtype)
        alignment = getattr(obj, "numpy_array_alignment_bytes", None)
        if alignment is not None:
            if type(alignment) is not int or not 1 <= alignment <= 256:
                raise GrailFormatError("Invalid joblib array alignment")
            padding = self.read(1)
            # joblib writes alignment - ((position + 1) % alignment), so an
            # already aligned payload has a full alignment block of padding.
            if len(padding) != 1 or padding[0] > alignment:
                raise GrailFormatError("Invalid joblib array padding")
            if len(self.read(padding[0])) != padding[0]:
                raise GrailFormatError("Truncated joblib array padding")
        raw = self.read(nbytes)
        if len(raw) != nbytes:
            raise GrailFormatError("Truncated joblib numeric array")
        self.stack[-1] = np.frombuffer(raw, dtype=dtype).reshape(obj.shape, order=obj.order).copy()

    def reject_opcode(self):
        raise GrailFormatError("Unsupported executable/external-buffer pickle opcode")

    dispatch = pickle._Unpickler.dispatch.copy()
    dispatch[pickle.REDUCE[0]] = load_reduce
    dispatch[pickle.NEWOBJ[0]] = load_newobj
    dispatch[pickle.BUILD[0]] = load_build
    for _opcode in (pickle.INST, pickle.OBJ, pickle.NEWOBJ_EX, pickle.EXT1, pickle.EXT2,
                    pickle.EXT4, pickle.NEXT_BUFFER, pickle.READONLY_BUFFER):
        dispatch[_opcode[0]] = reject_opcode


def _load_numeric(path):
    path = Path(path).resolve(strict=True)
    if not path.is_file() or path.stat().st_size > _MAX_BYTES:
        raise GrailFormatError("Source is not a bounded regular file")
    raw = path.read_bytes()
    if len(raw) > _MAX_BYTES:
        raise GrailFormatError("Source exceeds byte limit")
    receipt = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    payload = raw
    if raw.startswith(b"\x1f\x8b") or (len(raw) >= 2 and raw[0] == 0x78 and int.from_bytes(raw[:2], "big") % 31 == 0):
        decoder = zlib.decompressobj(47)
        payload = decoder.decompress(raw, _MAX_BYTES + 1)
        if len(payload) > _MAX_BYTES or decoder.unconsumed_tail or not decoder.eof or decoder.unused_data:
            raise GrailFormatError("Oversized, concatenated or incomplete compressed pickle")
    stream = io.BytesIO(payload)
    try:
        value = _NumericUnpickler(stream).load()
    except GrailFormatError:
        raise
    except Exception as exc:
        raise GrailFormatError(f"Invalid numeric pickle/joblib: {path.name}: {exc}") from exc
    if stream.read(1):
        raise GrailFormatError("Trailing bytes after numeric pickle")
    return value, receipt


def _entry(value, name):
    if not isinstance(value, dict) or len(value) != 1:
        raise GrailFormatError(f"{name} must contain exactly one motion")
    key, data = next(iter(value.items()))
    if not isinstance(key, str) or not isinstance(data, dict):
        raise GrailFormatError(f"{name} has malformed motion key/value")
    return key, data


def _array(value, shape, name):
    array = np.asarray(value)
    if array.dtype.kind not in "iuf" or array.shape != shape or not np.isfinite(array).all():
        raise GrailFormatError(f"{name}: expected finite numeric shape {shape}, got {array.shape}")
    return np.array(array, dtype=np.float64, copy=True)


def _quat(value, n, name):
    array = _array(value, (n, 4), name)
    norm = np.linalg.norm(array, axis=1)
    if np.any(np.abs(norm - 1.0) > 1e-3):
        raise GrailFormatError(f"{name}: nonunit quaternion")
    # Preserve source values (including sign); no normalization/resampling.
    return array


def _root_quaternion(robot, n, requested):
    raw = _quat(robot["root_rot"], n, "robot.root_rot")
    pose = _array(robot["pose_aa"], (n, 30, 3), "robot.pose_aa")
    rotvec = pose[:, 0]
    angle = np.linalg.norm(rotvec, axis=1)
    reference = np.column_stack((np.cos(angle / 2), rotvec * (0.5 * np.sinc(angle[:, None] / (2 * np.pi)))))
    candidates = {"wxyz": raw, "xyzw": raw[:, [3, 0, 1, 2]]}
    errors = {k: float(np.max(np.minimum(np.linalg.norm(q-reference, axis=1),
                                           np.linalg.norm(q+reference, axis=1))))
              for k, q in candidates.items()}
    matches = [k for k, error in errors.items() if error <= 1e-4]
    if requested is not None:
        if requested not in candidates or requested not in matches:
            raise GrailFormatError(f"Requested root quaternion order disagrees with pose_aa: {errors}")
        chosen = requested
    elif len(matches) == 1:
        chosen = matches[0]
    else:
        raise GrailFormatError(f"Root quaternion convention not uniquely verified by pose_aa: {errors}")
    return candidates[chosen].copy(), {"source_order": chosen, "verification": "pose_aa_root_axis_angle", "errors": errors}


def _contacts(objects):
    result = {}
    for name in ("contact_points_left_hand", "contact_points_right_hand"):
        if name not in objects:
            continue
        values = objects[name]
        if not isinstance(values, dict):
            raise GrailFormatError(f"{name}: expected frame-indexed dictionary")
        points = {}
        for frame, value in values.items():
            if not isinstance(frame, (int, np.integer)) or int(frame) < 0:
                raise GrailFormatError(f"{name}: invalid source frame index")
            array = np.asarray(value)
            if array.ndim != 2 or array.shape[1] != 3:
                raise GrailFormatError(f"{name}: expected (num_points,3)")
            _array(array, array.shape, name)
            points[int(frame)] = array.copy()
        result[name] = points
    return {"kind": "source_hand_surface_points", "points_by_source_frame": result,
            "is_force": False, "is_retargeted_contact_ground_truth": False,
            "coordinate_frame": "source_object_contact_frame_unverified",
            "timebase": "source_annotation_frame_indices_not_resampled",
            "warning": "Copied from upstream reconstruction; do not interpret missing keys as certified no-contact or align rollout-export frames without checking source timing."}


def load_clip(robot_path, object_path=None, *, meta_path=None, object_asset_path=None,
              source_revision=None, source_group=None, split=None, root_quat_order=None):
    """Return a canonical clip; source samples and original frame count stay intact.

    ``robot_path`` can also be a record returned by ``list_clips``. Asset fields
    describe how a USD geometry reader must compose coordinates; this adapter
    does not flatten USD meshes or pretend geometry has already been audited.
    """
    if isinstance(robot_path, Mapping):
        record = robot_path
        return load_clip(record["robot_path"], object_path or record.get("object_path"),
                         meta_path=meta_path or record.get("meta_path"),
                         object_asset_path=object_asset_path or record.get("object_asset_path"),
                         source_revision=source_revision or record.get("source_revision"),
                         source_group=source_group or record.get("source_group"),
                         split=split if split is not None else record.get("split"),
                         root_quat_order=root_quat_order)
    robot_path = Path(robot_path).resolve(strict=True)
    library = robot_path.parent.parent
    object_path = Path(object_path) if object_path is not None else library / "objects" / robot_path.name
    if meta_path is None:
        candidate = library / "meta" / robot_path.name
        meta_path = candidate if candidate.is_file() else None
    if object_asset_path is None:
        candidates = [library / "object_usd" / (robot_path.stem + ext) for ext in (".usd", ".usda", ".usdc")]
        found = [p for p in candidates if p.is_file()]
        if len(found) != 1:
            raise GrailFormatError("Provide exactly one paired object USD asset")
        object_asset_path = found[0]
    rv, rr = _load_numeric(robot_path)
    ov, ore = _load_numeric(object_path)
    key, robot = _entry(rv, "robot")
    object_key, objects = _entry(ov, "objects")
    if key != object_key:
        raise GrailFormatError("Robot/object internal motion keys disagree")
    dof = np.asarray(robot["dof"])
    if dof.ndim != 2 or dof.shape[0] < 2 or dof.shape[1] != 29:
        raise GrailFormatError("Only G1 sitting motions with N>=2 and exactly 29 body DOFs are supported")
    n = len(dof)
    joint_pos = _array(dof, (n, 29), "robot.dof")
    fps = float(robot["fps"])
    object_fps = float(objects["fps"])
    if not math.isfinite(fps) or fps <= 0 or not math.isclose(fps, object_fps, rel_tol=1e-9):
        raise GrailFormatError("Robot/object FPS must agree and be finite/positive")
    root_quat, quaternion_evidence = _root_quaternion(robot, n, root_quat_order)
    meta, mr = ({}, None) if meta_path is None else _load_numeric(meta_path)
    if not isinstance(meta, dict):
        raise GrailFormatError("Metadata must be a dictionary")
    # The released-USD replay contract applies objects.scale, not the separate
    # scene-level spawn branch. All audited sitting meta files have scene_scale=1.
    # A nonunit scene scale needs a separate convention audit, not multiplication
    # guessed from floor alignment or an unrelated training configuration.
    scene_scale = float(meta.get("scene_scale", 1.0))
    if not math.isfinite(scene_scale) or scene_scale <= 0:
        raise GrailFormatError("Invalid meta.scene_scale")
    if scene_scale != 1.0:
        raise GrailFormatError("Nonunit meta.scene_scale needs an explicit released-USD composition audit")
    asset = Path(object_asset_path).resolve(strict=True)
    if asset.suffix.lower() not in (".usd", ".usda", ".usdc") or not asset.is_file():
        raise GrailFormatError("Expected an existing USD object asset")
    asset_raw = asset.read_bytes()
    if not (asset_raw.startswith(b"PXR-USDC") or asset_raw.startswith(b"#usda")):
        raise GrailFormatError("Object asset is not a recognized USD layer")
    source_scale = np.asarray(objects.get("scale", np.ones(3))).reshape(-1)
    if (source_scale.dtype.kind not in "iuf" or source_scale.shape not in ((1,), (3,))
            or not np.isfinite(source_scale).all() or np.any(source_scale <= 0)):
        raise GrailFormatError("Invalid source object scale")
    applied_scale = np.repeat(source_scale, 3) if source_scale.size == 1 else source_scale
    supports = []
    if any(k in meta for k in ("table_pos", "table_quat", "table_size")):
        required = ("table_pos", "table_quat", "table_size")
        if not all(k in meta for k in required):
            supports.append({"kind": "table", "status": "incomplete_metadata", "metadata": {k: meta[k] for k in required if k in meta}})
        else:
            supports.append({"kind": "box", "name": "table", "position": _array(meta["table_pos"], (3,), "table_pos"),
                             "quaternion_wxyz": _quat(np.asarray(meta["table_quat"])[None], 1, "table_quat")[0],
                             "size": _array(meta["table_size"], (3,), "table_size"), "provenance": "source_meta"})
    group = source_group or key
    return {
        "clip_id": robot_path.stem, "timestamps_s": np.arange(n, dtype=np.float64) / fps,
        "root_pos": _array(robot["root_trans_offset"], (n, 3), "robot.root_trans_offset"),
        "root_quat": root_quat, "joint_pos": joint_pos, "joint_names": list(G1_JOINT_NAMES),
        "object_pos": _array(objects["root_pos"], (n, 1, 3), "object.root_pos")[:, 0],
        "object_quat": _quat(_array(objects["root_quat"], (n, 1, 4), "object.root_quat")[:, 0], n, "object.root_quat"),
        "object_asset": {
            "path": str(asset), "sha256": hashlib.sha256(asset_raw).hexdigest(), "format": "usd",
            "scale": applied_scale.tolist(), "source_object_scale": source_scale.tolist(),
            "source_object_scale_applied": True,
            "scale_source": "objects.scale" if "scale" in objects else "author_replay_default_1_missing_object_scale",
            "source_meta_scene_scale": scene_scale if mr else None,
            "meta_scene_scale_applied": False,
            "geometry_frame": "rigid_root_local_meters_after_descendant_USD_composition",
            "transform_contract": "world = R(object_quat_wxyz) @ (diag(scale) @ USD_mesh_to_rigid_root_local_m) + object_pos",
            "usd_transform_policy": "compose_mesh_to_rigid_root_descendant_xforms; replace_root_pose_and_scale_with_reference_pose_and_objects_scale",
            "usd_root_policy": "default_prim_must_be_the_unique_rigid_body_root; otherwise_requires_explicit_audit",
            "requires_usd_composition": True, "geometry_verified": False,
            "warning": "Apply objects.scale once to source-size released USD geometry; do not additionally multiply meta.scene_scale or infer scale from floor contact.",
        },
        "scene_supports": supports, "source_contacts": _contacts(objects),
        "provenance": {
            "dataset": "nvidia/PhysicalAI-Robotics-Locomanipulation-GRAIL", "subset": "sitting",
            "source_revision": source_revision, "source_group": group,
            "source_group_basis": "explicit" if source_group else "internal_motion_key_unverified_for_split_leakage",
            "split": split, "source_motion_key": key,
            "files": {"robot": rr, "objects": ore, "meta": mr,
                      "object_asset": {"path": str(asset), "sha256": hashlib.sha256(asset_raw).hexdigest(), "bytes": len(asset_raw)}},
            "source_fps": fps, "source_frames": n, "resampled": False,
            "world_frame": "source_Z_up_meters", "canonical_quaternion_order": "wxyz",
            "robot_quaternion": quaternion_evidence, "object_quaternion_source_order": "wxyz",
            "joint_order": "GRAIL_MuJoCo_G1_29_identical_to_supermimic_G1_JOINT_NAMES",
            "smpl_joints_used": False, "convention_sources": dict(CONVENTION_SOURCES),
        },
    }


def list_clips(source_root, *, source_revision=None, split=None):
    """Inventory paired file paths without loading arrays or hiding missing assets."""
    root = Path(source_root).resolve(strict=True)
    if (root / "data" / "sitting" / "robot").is_dir():
        root = root / "data" / "sitting"
    if not (root / "robot").is_dir():
        raise GrailFormatError("Expected GRAIL sitting library containing robot/")
    result = []
    for robot in sorted((root / "robot").glob("*.pkl")):
        objects = root / "objects" / robot.name
        meta = root / "meta" / robot.name
        assets = [root / "object_usd" / (robot.stem + ext) for ext in (".usd", ".usda", ".usdc")]
        assets = [p for p in assets if p.is_file()]
        missing = [name for name, p in (("objects", objects), ("meta", meta)) if not p.is_file()]
        if len(assets) != 1: missing.append("unique_object_asset")
        result.append({"clip_id": robot.stem, "robot_path": str(robot), "object_path": str(objects),
                       "meta_path": str(meta) if meta.is_file() else None,
                       "object_asset_path": str(assets[0]) if len(assets) == 1 else None,
                       "source_revision": source_revision, "source_group": None, "split": split,
                       "files_complete": not missing, "missing": missing})
    return result
