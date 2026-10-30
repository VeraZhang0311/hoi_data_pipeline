"""ULTRA frozen-selection reader; original 60 Hz protocol grid, no resampling.

list_clips(selection_path, *, source_root=None, profile='canonical') -> records.
load_clip(selection_path, clip_id, *, bindings_path, assets_path,
          timebase='canonical_60hz', source_root=None, profile='canonical') -> dict.
The .pt tensor does not contain joint names or asset transforms: both explicit
bindings and the original object recipe are required. Custom fixtures remain
unassigned; only byte-pinned canonical selections receive known membership.
"""
from __future__ import annotations

import copy
import hashlib
import io
from functools import lru_cache
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np

from .hoi_retarget import G1_JOINT_NAMES, FROZEN_SELECTIONS, _json_evidence, _joint_order, _quat, _fingerprint


@lru_cache(maxsize=16)
def _cached_json(path, fingerprint):
    value = _json_evidence(path)
    if _fingerprint(path) != fingerprint:
        raise ValueError("JSON evidence changed while reading")
    return value


def _read_json(path):
    path = str(Path(path).resolve())
    return _cached_json(path, _fingerprint(path))


@lru_cache(maxsize=8)
def _selection_catalog(selection_path, fingerprint, source_root, profile):
    data, selection = _cached_json(selection_path, fingerprint)
    matches = [role for role, digest in FROZEN_SELECTIONS.items() if digest == selection["sha256"]]
    if profile not in ("canonical", "custom"):
        raise ValueError("profile must be canonical or custom")
    if profile == "canonical" and len(matches) != 1:
        raise ValueError("selection does not match either frozen ULTRA selection SHA256")
    role = matches[0] if profile == "canonical" else "unassigned"
    cases = data.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("selection must contain nonempty cases")
    if source_root is not None and len({str(PurePosixPath(c["motion"].replace("\\", "/")).parent) for c in cases}) != 1:
        raise ValueError("flat source_root remap requires one common original source directory")
    records = []
    for i, case in enumerate(cases):
        for key in ("motion", "motion_sha256", "source_group", "object", "object_asset"):
            if not isinstance(case.get(key), str) or not case[key]:
                raise ValueError("selection entry missing " + key)
        source = PurePosixPath(case["motion"].replace("\\", "/"))
        if not (source.is_absolute() or PureWindowsPath(case["motion"]).is_absolute()) or ".." in source.parts or source.suffix != ".pt":
            raise ValueError("motion must be an absolute .pt path; no traversal")
        if "_".join(source.stem.split("_")[:3]) != case["source_group"]:
            raise ValueError("source_group disagrees with original ULTRA filename")
        if len(case["motion_sha256"]) != 64 or any(c not in "0123456789abcdef" for c in case["motion_sha256"]):
            raise ValueError("selection requires exact lowercase source SHA256")
        resolved = Path(case["motion"]) if source_root is None else Path(source_root).resolve() / source.name
        records.append({**case, "clip_id": source.stem, "row_index": i, "source_path": str(resolved),
                        "selection_evidence": selection, "source_revision": "OMOMO_retarget_aug_v4",
                        "split": "train" if role == "train" else "validation" if role == "eval80" else "unassigned",
                        "protected_eval80": role == "eval80", "split_identity_verified": role != "unassigned",
                        "training_ready": False})
    if len({r["clip_id"] for r in records}) != len(records):
        raise ValueError("duplicate source motion in selection")
    aliases = {}
    for i, record in enumerate(records):
        for key in (record["clip_id"], record.get("id"), record["motion"]):
            if key is None:
                continue
            if key in aliases and aliases[key] != i:
                raise ValueError("ambiguous clip alias in selection")
            aliases[key] = i
    return tuple(records), aliases


def _catalog(selection_path, source_root, profile):
    path = str(Path(selection_path).resolve())
    root = None if source_root is None else str(Path(source_root).resolve())
    return _selection_catalog(path, _fingerprint(path), root, profile)


def list_clips(selection_path, *, source_root=None, profile="canonical"):
    return copy.deepcopy(list(_catalog(selection_path, source_root, profile)[0]))


def _object_asset(assets_path, asset_id):
    recipe, evidence = _read_json(assets_path)
    entries = [entry for entry in recipe.get("assets", []) if entry.get("object_asset") == asset_id]
    if len(entries) != 1:
        raise ValueError("object asset must have exactly one explicit recipe entry: " + asset_id)
    entry = entries[0]
    scale = np.asarray(entry.get("exact_scale"), dtype=np.float64)
    if scale.ndim == 0:
        scale = np.repeat(scale, 3)
    transform = np.asarray(entry.get("mesh_to_object"), dtype=np.float64)
    if scale.shape != (3,) or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError("asset recipe requires positive exact_scale; never infer it from asset name")
    if transform.shape != (4, 4) or not np.isfinite(transform).all() or not np.allclose(transform[3], [0, 0, 0, 1]):
        raise ValueError("asset recipe requires explicit homogeneous mesh_to_object")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7, rtol=0) or np.linalg.det(rotation) < 0:
        raise ValueError("mesh_to_object must contain a proper rotation")
    if not entry.get("collision_meshes"):
        raise ValueError("asset recipe requires original collision mesh evidence")
    for mesh in entry["collision_meshes"]:
        if mesh.get("frame", "mesh") not in ("mesh", "object_local"):
            raise ValueError("unsupported collision mesh frame")
    return {"asset_id": asset_id, "recipe_evidence": copy.deepcopy(evidence), "recipe_entry": copy.deepcopy(entry),
            "exact_scale": scale.tolist(), "mesh_to_object": transform.tolist(),
            "pose_origin": "recorded_object_pose_origin", "pose_frame_alignment": "existing_frozen_ULTRA_recipe_binding",
            "transform_application": "not_applied_by_adapter; object_local meshes already baked, mesh-frame meshes need recipe transform",
            "robot_geometry_policy": "recipe hands ignored; downstream robot geometry must use the explicitly bound v1.2 asset"}


def load_clip(selection_path, clip_id, *, bindings_path, assets_path, timebase=None,
              source_root=None, profile="canonical"):
    if timebase != "canonical_60hz":
        raise ValueError("explicit timebase='canonical_60hz' required; generation clock is not verified")
    records, aliases = _catalog(selection_path, source_root, profile)
    if clip_id not in aliases:
        raise KeyError("clip must match exactly one entry in the frozen selection")
    record = copy.deepcopy(records[aliases[clip_id]])
    binding_document, binding_evidence = _read_json(bindings_path)
    bindings = binding_document.get("bindings", binding_document)
    source_names = copy.deepcopy(bindings.get("source_joint_names"))
    if not isinstance(source_names, list) or not bindings.get("joint_order_evidence"):
        raise ValueError("explicit source joint order and its provenance are required; the tensor has no names")
    order = _joint_order(source_names)
    if "target_joint_names" in bindings and bindings["target_joint_names"] != list(G1_JOINT_NAMES):
        raise ValueError("binding target joint order is not the canonical SuperMimic G1 order")
    path = Path(record["source_path"])
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != record["motion_sha256"]:
        raise ValueError("ULTRA motion SHA256 differs from selection")
    try:
        import torch
    except ImportError as exc:
        raise ImportError("ULTRA adapter requires torch to load its weights-only source tensor") from exc
    frames = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    if not isinstance(frames, torch.Tensor) or frames.layout != torch.strided or frames.is_quantized or not frames.is_floating_point():
        raise TypeError("ULTRA source must be a plain dense floating Tensor[N,630]")
    if frames.ndim != 2 or frames.shape[1] != 630 or len(frames) < 2 or not bool(torch.isfinite(frames).all()):
        raise ValueError("ULTRA source requires finite Tensor[N,630] with at least two states")
    a = frames.to(dtype=torch.float64).numpy()
    n = len(a)
    for start, stop in ((3, 7), (74, 78), (201, 357)):
        q = a[:, start:stop].reshape(-1, 4)
        if np.max(np.abs(np.linalg.norm(q, axis=-1) - 1)) > 1e-3:
            raise ValueError("ULTRA source has non-unit xyzw quaternion")
    if not np.isin(a[:, 591:630], [0, 1]).all():
        raise ValueError("ULTRA source body contact flags are not binary")
    if "source_frames" in record and record["source_frames"] != n:
        raise ValueError("ULTRA frame count differs from frozen selection")
    return {"clip_id": record["clip_id"], "timestamps_s": np.arange(n, dtype=np.float64) / 60.0,
            "root_pos": a[:, :3].copy(), "root_quat": _quat(a[:, [6, 3, 4, 5]], n, "root_quat"),
            "joint_pos": a[:, 13:42][:, order].copy(), "joint_names": list(G1_JOINT_NAMES),
            "object_pos": a[:, 71:74].copy(), "object_quat": _quat(a[:, [77, 74, 75, 76]], n, "object_quat"),
            "object_asset": _object_asset(assets_path, record["object_asset"]),
            "source_contacts": {"values": {"body_contact_flags": a[:, 591:630].copy()},
                                "body_order": "original_39_body_order_unmodified",
                                "semantics": "source_body_flags_not_object_pair_force_or_impulse_truth"},
            "provenance": {**record, "source_sha256": digest, "source_schema": "ultra_tensor_N_630",
                           "source_joint_names": source_names, "target_to_source_joint_indices": order,
                           "joint_order_evidence": copy.deepcopy(bindings["joint_order_evidence"]), "binding_document": copy.deepcopy(binding_evidence),
                           "source_quaternion_order": "xyzw", "quaternion_order": "wxyz", "position_units": "m",
                           "world_frame": "source_world_z_up_no_recenter", "source_fps": 60.0,
                           "timebase_status": "canonical_60hz_protocol_not_generation_verified", "resampled": False,
                           "robot_geometry_for_labels": "must_bind_verified_v1.2_geometry_downstream"}}
