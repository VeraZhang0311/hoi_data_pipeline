"""Read-only HOI-Retarget OMOMO adapter. No resampling, FK, or contact inference.

list_clips(parquet_path, *, revision, split_index=None) returns row records.
load_clip(parquet_path, clip_id=None, *, row_index=None, source_root, revision,
          split_index=None) returns canonical NumPy arrays, original contacts,
          asset evidence and provenance. Exactly one row selector is required.
"""
from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

# Copied verbatim from the verified SuperMimic supermimic/robot.py order.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint", "right_hip_pitch_joint", "right_hip_roll_joint",
    "right_hip_yaw_joint", "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint", "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint", "left_wrist_roll_joint",
    "left_wrist_pitch_joint", "left_wrist_yaw_joint", "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
FROZEN_SELECTIONS = {
    "train": "8c141eb60650b9df79b92acfeeea785f796c7f18f52bf179467562c7cc315b23",
    "eval80": "459a3925faf0f66bee258ce98635c67802ddda00d47d007ecdeb45035f576002",
}
_CONTACT_FIELDS = (
    "contact_link_names", "per_link_contact_flags", "fixed_contact_points_per_frame_in_object_frame",
    "robot_foot_ground_contact_flags", "object_contact_sequence", "object_ground_contact_sequence",
    "object_near_floor_flags", "palm_contact",
)
_META_FIELDS = (
    "clip_id", "dataset", "robot", "subject", "sequence_id", "n_actors", "n_frames", "fps",
    "object", "object_model_path", "schema_version", "qc_pass", "qc_flags",
)


def _fingerprint(path):
    s = Path(path).stat()
    return s.st_size, s.st_mtime_ns, s.st_ino


@lru_cache(maxsize=64)
def _hash_cached(path, fingerprint):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    if _fingerprint(path) != fingerprint:
        raise ValueError("source changed while hashing: " + str(path))
    return h.hexdigest()


def sha256_file(path):
    path = str(Path(path).resolve())
    return _hash_cached(path, _fingerprint(path))


def _json_evidence(path):
    path = Path(path).resolve()
    raw = path.read_bytes()
    return json.loads(raw.decode("utf-8-sig")), {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}


def frozen_ultra_split_index(train_selection_path, eval80_selection_path):
    """Conservatively protect matching OMOMO capture IDs, not certify a new split."""
    result = {}
    for role, path in (("train", train_selection_path), ("eval80", eval80_selection_path)):
        data, evidence = _json_evidence(path)
        if evidence["sha256"] != FROZEN_SELECTIONS[role]:
            raise ValueError("frozen ULTRA " + role + " selection SHA256 mismatch")
        for entry in data["cases"]:
            group = entry["source_group"]
            if group in result and result[group]["ultra_partition"] != role:
                raise ValueError("ULTRA source group appears in both train and eval80")
            result[group] = {"ultra_partition": role, "evidence": evidence}
    return result


def _split_info(group, split_index=None):
    match = (split_index or {}).get(group)
    protected = bool(match and (match.get("ultra_partition") == "eval80" or match.get("protected_eval80")))
    return {"split": "validation" if protected else "unassigned",
            "protected_eval80": protected, "ultra_source_group_match": match,
            "cross_source_identity_verified": False, "requires_split_reconciliation": True,
            "training_ready": False}


def _joint_order(names):
    if len(names) != 29 or len(set(names)) != 29 or set(names) != set(G1_JOINT_NAMES):
        raise ValueError("source joint names must name each canonical G1 joint exactly once")
    return [names.index(name) for name in G1_JOINT_NAMES]


def _array(value, shape, name):
    a = np.asarray(value, dtype=np.float64)
    if a.shape != tuple(shape) or not np.isfinite(a).all():
        raise ValueError(name + ": expected finite array " + str(tuple(shape)))
    return np.ascontiguousarray(a)


def _quat(value, n, name):
    q = _array(value, (n, 4), name)
    if np.max(np.abs(np.linalg.norm(q, axis=-1) - 1)) > 1e-3:
        raise ValueError(name + ": non-unit quaternion; adapter does not silently repair")
    return q


def _parquet(path):
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise ImportError("HOI-Retarget adapter requires pyarrow in the caller's environment") from exc
    return pq.ParquetFile(path)


def _record(row, index, path, revision, split_index):
    if row["dataset"] != "omomo" or row["robot"] != "unitree_g1" or row["schema_version"] != 2:
        raise ValueError("only schema-v2 OMOMO/unitree_g1 rows have been verified")
    group = row["subject"]
    if row["sequence_id"] != "omomo/" + group or row["clip_id"] != "omomo/unitree_g1/" + group:
        raise ValueError("unexpected source capture identity; cannot infer an OMOMO group")
    if row["n_actors"] != 1 or row["fps"] != 30.0 or row["n_frames"] < 2:
        raise ValueError("unsupported actor/timebase metadata")
    return {**row, "row_index": index, "source_group": group, "source_path": str(Path(path).resolve()),
            "source_revision": revision, **_split_info(group, split_index)}


@lru_cache(maxsize=8)
def _metadata_rows(path, fingerprint):
    pf = _parquet(path)
    rows = []
    for batch in pf.iter_batches(batch_size=256, columns=list(_META_FIELDS)):
        rows.extend(batch.to_pylist())
    if _fingerprint(path) != fingerprint:
        raise ValueError("source changed during metadata read")
    return rows


def list_clips(parquet_path, *, revision, split_index=None):
    if not isinstance(revision, str) or not revision:
        raise ValueError("source revision must be supplied explicitly")
    path = str(Path(parquet_path).resolve())
    records = [_record(copy.deepcopy(row), i, path, revision, split_index)
               for i, row in enumerate(_metadata_rows(path, _fingerprint(path)))]
    if len({r["clip_id"] for r in records}) != len(records):
        raise ValueError("duplicate clip_id in source parquet")
    return records


def _within(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if not path.is_relative_to(root):
        raise ValueError("asset path escapes source snapshot")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def _triple(text, default):
    a = np.asarray([float(v) for v in text.split()] if text is not None else default, dtype=np.float64)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError("invalid URDF origin/scale")
    return a.tolist()


def _object_asset(row, source_root):
    root = Path(source_root).resolve()
    relative = Path(row["object_model_path"])
    if relative.is_absolute():
        raise ValueError("object_model_path must be snapshot-relative")
    urdf = _within(root / relative, root)
    tree = ET.parse(urdf).getroot()
    if len(tree.findall("link")) != 1 or tree.findall("joint"):
        raise ValueError("only verified single-link rigid object URDFs are supported")
    meshes = []
    for role in ("visual", "collision"):
        for element in tree.findall("./link/" + role):
            mesh = element.find("./geometry/mesh")
            if mesh is None:
                raise ValueError("object geometry is not a mesh; explicit adapter required")
            filename = mesh.get("filename", "")
            if not filename or "://" in filename or Path(filename).is_absolute():
                raise ValueError("unsupported mesh URI")
            path = _within(urdf.parent / filename, root)
            origin = element.find("origin")
            scale = _triple(mesh.get("scale"), [1, 1, 1])
            if min(scale) <= 0:
                raise ValueError("mesh scale must be positive")
            meshes.append({"path": str(path), "sha256": sha256_file(path), "role": role,
                           "scale": scale, "origin_xyz": _triple(None if origin is None else origin.get("xyz"), [0, 0, 0]),
                           "origin_rpy": _triple(None if origin is None else origin.get("rpy"), [0, 0, 0])})
    if not meshes:
        raise ValueError("object URDF has no mesh geometry")
    scale, aug = float(row["object_scale"]), float(row["object_aug_scale"])
    if not np.isfinite([scale, aug]).all() or scale <= 0 or aug != 1.0:
        raise ValueError("unknown object scale/augmentation composition; explicit evidence required")
    if row["scale_object_mesh"] is not True or row["scale_object_trajectory"] is not True:
        raise ValueError("only verified G1 mesh/trajectory scale flags are supported")
    return {"asset_id": "omomo/" + row["object"], "source_model_path": row["object_model_path"],
            "urdf_path": str(urdf), "urdf_sha256": sha256_file(urdf), "meshes": meshes,
            "source_scale": scale, "source_aug_scale": aug, "scale_object_mesh": True,
            "scale_object_trajectory": True, "trajectory_scaling_already_applied": True,
            "pose_origin": "recorded_object_URDF_link_frame", "pose_frame_alignment": "source_URDF_binding_no_recenter",
            "transform_application": "not_applied_by_adapter; scale source mesh once and preserve URDF origins",
            "robot_geometry_policy": "not_used; downstream robot geometry must be the explicitly bound v1.2 asset"}


def load_clip(parquet_path, clip_id=None, *, row_index=None, source_root, revision, split_index=None):
    if (clip_id is None) == (row_index is None):
        raise ValueError("provide exactly one of clip_id and row_index")
    records = list_clips(parquet_path, revision=revision, split_index=split_index)
    if clip_id is not None:
        found = [r for r in records if r["clip_id"] == clip_id]
        if len(found) != 1:
            raise KeyError("clip_id not uniquely present: " + str(clip_id))
        record = found[0]
    else:
        if isinstance(row_index, bool) or not isinstance(row_index, int) or not 0 <= row_index < len(records):
            raise IndexError("row_index out of range")
        record = records[row_index]
    index = record["row_index"]
    before = _fingerprint(parquet_path)
    digest = sha256_file(parquet_path)
    pf = _parquet(parquet_path)
    offset, selected = 0, None
    columns = list(_LOAD_FIELDS)
    for group in range(pf.num_row_groups):
        count = pf.metadata.row_group(group).num_rows
        if offset <= index < offset + count:
            for batch in pf.iter_batches(batch_size=16, row_groups=[group], columns=columns):
                if offset <= index < offset + len(batch):
                    selected = batch.slice(index - offset, 1).to_pylist()[0]
                    break
                offset += len(batch)
            break
        offset += count
    if selected is None or selected["clip_id"] != record["clip_id"] or _fingerprint(parquet_path) != before:
        raise ValueError("source row changed during read")
    return _canonical_clip(selected, record, digest, source_root)


def _canonical_clip(row, record, digest, source_root):
    index = record["row_index"]
    n = row["n_frames"]
    names = list(row["dof_names"])
    order = _joint_order(names)
    contacts = {k: row[k] for k in _CONTACT_FIELDS if k in row}
    links = contacts.get("contact_link_names", [])
    flags = np.asarray(contacts.get("per_link_contact_flags"))
    points = np.asarray(contacts.get("fixed_contact_points_per_frame_in_object_frame"), dtype=np.float64)
    if len(set(links)) != 4 or flags.shape != (n, 4) or flags.dtype.kind != "b" or points.shape != (n, 4, 3):
        raise ValueError("malformed source link contact annotations")
    if not np.isfinite(points[flags]).all() or np.isinf(points).any():
        raise ValueError("non-finite active contact point")
    return {"clip_id": record["clip_id"], "timestamps_s": np.arange(n, dtype=np.float64) / row["fps"],
            "root_pos": _array(row["root_pos"], (n, 3), "root_pos"),
            "root_quat": _quat(row["root_rot"], n, "root_rot"),
            "joint_pos": _array(row["dof_pos"], (n, 29), "dof_pos")[:, order],
            "joint_names": list(G1_JOINT_NAMES),
            "object_pos": _array(row["object_pos"], (n, 3), "object_pos"),
            "object_quat": _quat(row["object_rot"], n, "object_rot"),
            "object_asset": _object_asset(row, source_root),
            "source_contacts": {"values": contacts, "semantics": "source_annotations_not_native_force_or_impulse_truth"},
            "provenance": {**record, "source_sha256": digest, "source_schema": "hoi_retarget_v2",
                           "source_row_index": index, "source_joint_names": names, "target_to_source_joint_indices": order,
                           "quaternion_order": "wxyz", "position_units": "m", "world_frame": "source_world_no_recenter",
                           "source_fps": row["fps"], "timebase_status": "source_declared_30hz",
                           "source_duration_s": row["duration_s"], "resampled": False,
                           "robot_geometry_for_labels": "must_bind_verified_v1.2_geometry_downstream"}}


# Preview bytes and unused velocity/realsize arrays are never read by the adapter.
_LOAD_FIELDS = tuple(dict.fromkeys(_META_FIELDS + (
    "dof_names", "duration_s", "root_pos", "root_rot", "dof_pos", "object_pos", "object_rot",
    "object_scale", "object_aug_scale", "scale_object_mesh", "scale_object_trajectory",
) + _CONTACT_FIELDS))


def iter_clips(parquet_path, *, source_root, revision, split_index=None,
               clip_ids=None, row_indices=None, batch_size=16, on_error="raise"):
    """Stream each selected row once; use this API for bulk annotation.

    Do not loop over load_clip for a full corpus. This iterator decodes each
    relevant row group once, in bounded batches, without reading preview bytes.
    Source hashes use a stat-keyed per-process cache; mutation aborts iteration.
    A filter is optional and clip_ids/row_indices are mutually exclusive.
    on_error="yield" emits an explicit _adapter_error record for a malformed
    selected row and continues; file mutation/stream corruption still aborts.
    """
    if not isinstance(revision, str) or not revision:
        raise ValueError("source revision must be supplied explicitly")
    if on_error not in ("raise", "yield"):
        raise ValueError("on_error must be raise or yield")
    if clip_ids is not None and row_indices is not None:
        raise ValueError("provide only one clip filter")
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("batch_size must be a positive integer")
    requested_ids = None if clip_ids is None else set(clip_ids)
    if requested_ids is not None and (isinstance(clip_ids, (str, bytes)) or any(not isinstance(x, str) for x in requested_ids)):
        raise ValueError("clip_ids must be a collection of strings")
    requested_rows = None if row_indices is None else set(row_indices)
    pf = _parquet(parquet_path)
    if requested_rows is not None and any(type(x) is not int or not 0 <= x < pf.metadata.num_rows for x in requested_rows):
        raise IndexError("invalid row_indices")
    before = _fingerprint(parquet_path)
    digest = sha256_file(parquet_path)
    seen, matched, index = set(), set(), 0
    for batch in pf.iter_batches(batch_size=batch_size, columns=list(_LOAD_FIELDS)):
        for row in batch.to_pylist():
            if row["clip_id"] in seen:
                raise ValueError("duplicate clip_id in source parquet")
            seen.add(row["clip_id"])
            row_index = index
            wanted = (requested_ids is None or row["clip_id"] in requested_ids) and (requested_rows is None or row_index in requested_rows)
            index += 1
            if wanted:
                if _fingerprint(parquet_path) != before:
                    raise ValueError("source changed while iterating")
                matched.add(row["clip_id"])
                try:
                    record = _record({k: row[k] for k in _META_FIELDS}, row_index, parquet_path, revision, split_index)
                    clip = _canonical_clip(row, record, digest, source_root)
                except Exception as exc:
                    if on_error == "raise":
                        raise
                    yield {"clip_id": row["clip_id"], "row_index": row_index,
                           "_adapter_error": {"type": type(exc).__name__, "message": str(exc)}}
                else:
                    yield clip
    if _fingerprint(parquet_path) != before:
        raise ValueError("source changed while iterating")
    if requested_ids is not None and matched != requested_ids:
        raise KeyError("requested clip_ids absent from source: " + str(sorted(requested_ids - matched)))
