"""Read-only adapter for the pinned OmniContact G1 reference release.

World poses, frame count and the original time base are preserved. This module
does not resample, cut, repair, or infer physical contacts. Carry-box geometry is
resolved only through explicit author resize/center metadata and verified box
topology; the other object bindings remain review items.
"""
from __future__ import annotations

import csv
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import numpy as np

REVISION = "068b83049df50653327b06ee41bbbaf1b9bd593e"
TRAINING_REVISION = "19cde38364a5b48e47348b270c895eabe2ba3ad9"
VIEWER_SHA256 = "50f0a007893af5b6c6d2f67abd50f5cab60da490cf01353d820a8d0b695762a6"
URDF_SHA256 = "85d7a98f15e10bd43eae40fac67f08c547239d6962ac1a1ca11f130bd571aa06"
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
# Official viewer lines 48-83 and 859-870; output is released URDF revolute order.
ISAAC_TO_G1 = (0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8,
               11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28)
CONTACT_CHANNELS = ("left_ankle", "right_ankle", "left_wrist", "right_wrist")


class OmniContactFormatError(ValueError):
    """Input lacks a verified layout or violates the source schema."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@lru_cache(maxsize=8)
def _release_layout(root: str) -> dict:
    """Require pinned evidence before interpreting unnamed joint/body columns."""
    base = Path(root)
    viewer = base / "visualize_npz_bvh_minimal.py"
    urdf = base / "assets/g1/g1_29dof.urdf"
    if not viewer.is_file() or not urdf.is_file():
        raise OmniContactFormatError("Unnamed layout requires the pinned official viewer and G1 URDF")
    if _sha256(viewer) != VIEWER_SHA256 or _sha256(urdf) != URDF_SHA256:
        raise OmniContactFormatError("Official layout evidence hash mismatch; review the new release")
    tree = ET.parse(urdf).getroot()
    children = {joint.find("child").get("link") for joint in tree.findall("joint")}
    roots = [link.get("name") for link in tree.findall("link") if link.get("name") not in children]
    names = [joint.get("name") for joint in tree.findall("joint") if joint.get("type") == "revolute"]
    if roots != ["pelvis"] or names != list(G1_JOINT_NAMES):
        raise OmniContactFormatError("Released URDF does not establish pelvis/SDK joint order")
    return {"viewer_sha256": VIEWER_SHA256, "urdf_sha256": URDF_SHA256,
            "root_evidence": "official viewer body_pos_w[:,0], body_quat_w[:,0]; URDF root pelvis",
            "joint_evidence": "official viewer ISAAC_SIM_TO_G1_JOINT_MAPPING -> URDF revolute order"}


def _source_root(path: Path) -> Path:
    for parent in path.parents:
        if (parent / "npz_clips").is_dir():
            return parent
    raise OmniContactFormatError("Pass source_root pointing at the released dataset directory")


def _csv_rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def list_clips(source_root, manifest_path=None, *, split=None) -> list[dict]:
    """List available continuous clips; metadata-only, no downloads or NPZ reads.

    Optional manifest is the earlier audit's JSONL. Otherwise source groups and
    splits come from official clip_info/raw_mocap_manifest/splits metadata.
    A unique filename suffix alias is recorded as provisional, never verified.
    """
    root = Path(source_root).resolve()
    prior = {}
    if manifest_path is not None:
        with Path(manifest_path).open(encoding="utf-8-sig") as stream:
            prior = {row["path"]: row for line in stream if (row := json.loads(line))}
    raw = {Path(row["npz"]).name: row for row in _csv_rows(root / "metadata/raw_mocap_manifest.csv")}
    splits = {row["capture_id"]: row["split"] for row in _csv_rows(root / "metadata/splits.csv")}
    indices = {}
    for index_path in sorted((root / "npz_clips").glob("*/*/clip_info.json")):
        content = json.loads(index_path.read_text(encoding="utf-8-sig"))
        if content.get("frame_indexing") != "0-based, inclusive":
            raise OmniContactFormatError(f"Unknown frame indexing: {index_path}")
        task = "/".join(index_path.relative_to(root).parts[1:3])
        indices[task] = {row["clip"]: row for row in content["clips"]}
    records = []
    for path in sorted((root / "npz_clips").rglob("*.npz")):
        relative = path.relative_to(root).as_posix()
        task = "/".join(path.relative_to(root).parts[1:3])
        record = dict(prior.get(relative, {}))
        index = indices.get(task, {}).get(path.name)
        mapping = "exact_basename" if index else "missing"
        if index is None:
            suffix = "_".join(path.stem.split("_")[-3:])
            candidates = [v for k, v in indices.get(task, {}).items()
                          if "_".join(Path(k).stem.split("_")[-3:]) == suffix]
            if len(candidates) == 1:
                index = candidates[0]
                mapping = "candidate_suffix_alias_requires_verification"
        capture = raw.get(index["source"], {}).get("capture_id") if index else None
        record.update(dataset="omnicontact", revision=REVISION, clip_id=relative,
                      path=str(path), relative_path=relative, source_root=str(root), task=task)
        if index:
            record.update(source_group_id=capture, capture_id=capture,
                          split=splits.get(capture), mapping_status=mapping,
                          source_group_verified=mapping == "exact_basename",
                          source_start_frame=index["start_frame"],
                          source_end_frame_inclusive=index["end_frame"],
                          source_path="npz/" + task + "/" + index["source"])
        else:
            record.setdefault("mapping_status", mapping)
            record.setdefault("source_group_verified", False)
        if split is None or record.get("split") == split:
            records.append(record)
    return records


def _array(data, key, shape, *, dtype=np.float64):
    if key not in data:
        raise OmniContactFormatError(f"Missing {key}")
    value = np.array(data[key], dtype=dtype, copy=True)
    if value.shape != shape or not np.isfinite(value).all():
        raise OmniContactFormatError(f"Invalid {key}: expected {shape}, got {value.shape}")
    return value


def _quat(value, label):
    if np.max(np.abs(np.linalg.norm(value, axis=-1) - 1.0)) > 1e-3:
        raise OmniContactFormatError(f"{label} contains non-unit quaternions; refusing silent normalization")
    return value


def _quat_multiply(a, b):
    aw, ax, ay, az = np.moveaxis(a, -1, 0)
    bw, bx, by, bz = np.moveaxis(b, -1, 0)
    return np.stack((aw*bw-ax*bx-ay*by-az*bz, aw*bx+ax*bw+ay*bz-az*by,
                     aw*by-ax*bz+ay*bw+az*bx, aw*bz+ax*by-ay*bx+az*bw), axis=-1)


@lru_cache(maxsize=8)
def _obj_cuboid_bounds(path: str):
    """Verify that the original OBJ is exactly the eight corners of a box."""
    with Path(path).open(encoding="utf-8") as stream:
        lines = stream.readlines()
    vertices = np.array([[float(v) for v in line.split()[1:4]]
                         for line in lines if line.startswith("v ")], dtype=np.float64)
    faces = [[int(value.split("/")[0])-1 for value in line.split()[1:]]
             for line in lines if line.startswith("f ")]
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
        raise OmniContactFormatError("Invalid original object vertices")
    low, high = vertices.min(axis=0), vertices.max(axis=0)
    if len(vertices) != 8 or len(np.unique(vertices, axis=0)) != 8 or np.any(high <= low):
        raise OmniContactFormatError("Original object is not an eight-corner cuboid")
    if not np.all(np.isclose(vertices, low, atol=1e-6) | np.isclose(vertices, high, atol=1e-6)):
        raise OmniContactFormatError("Original object vertices do not form an axis-aligned cuboid")
    planes = set()
    for face in faces:
        if len(face) != 4 or len(set(face)) != 4 or min(face) < 0 or max(face) >= 8:
            raise OmniContactFormatError("Original cuboid does not have six complete quad faces")
        points = vertices[face]
        face_planes = [(axis, side) for axis in range(3) for side, bound in enumerate((low, high))
                       if np.allclose(points[:, axis], bound[axis], atol=1e-6, rtol=0)]
        if len(face_planes) != 1:
            raise OmniContactFormatError("Original cuboid face is not a complete boundary plane")
        planes.add(face_planes[0])
    if len(faces) != 6 or len(planes) != 6:
        raise OmniContactFormatError("Original cuboid boundary is incomplete")
    return low, high


def _bind_carry_cuboid(data, asset: dict, relative: str) -> None:
    """Resolve author-augmented carry geometry without fitting contact labels.

    The raw OBJ is an offset cuboid. Author metadata specifies the FINAL box
    dimensions after recenter/resize; the filename supplies an independent check.
    Later palm-center passes explicitly change only the arms and retain this box.
    The supplied object quaternion already includes the XML geometry rotation.
    """
    if "/box/case1_carry/" not in "/" + relative or not asset["mesh_path"]:
        return
    required = ("carry_box_selected_size_cm", "carry_box_source_size_cm", "carry_box_offset_local_m",
                "carry_modification_method", "carry_modification_scope", "object_geom_quat_wxyz",
                "object_xml_geom_quat_mode", "object_quat_input_wxyz")
    if any(key not in data for key in required):
        asset["binding_review_reason"] = "Missing explicit carry recenter/resize metadata"
        return
    method = str(np.asarray(data["carry_modification_method"]).item())
    accepted_methods = {
        "box_recenter_resize_random_float_hand_ik_all_frames_v2_wristwide_2cm",
        "wristwide_palm_center_arm_only_ik_all_frames_v1",
    }
    if method not in accepted_methods or str(data["carry_modification_scope"].item()) != "all_frames":
        asset["binding_review_reason"] = "Unrecognized author augmentation lineage"
        return
    if method == "wristwide_palm_center_arm_only_ik_all_frames_v1":
        if "carry_newcontact_modified_joint_groups" not in data or str(data["carry_newcontact_modified_joint_groups"].item()) != "contacting_shoulder_elbow_wrist_only":
            asset["binding_review_reason"] = "Later augmentation is not explicitly arm-only"
            return
    size = np.asarray(data["carry_box_selected_size_cm"], dtype=np.float64).reshape(-1) / 100.0
    match = re.search(r"carry_box_(\d+)-(\d+)-(\d+)_", Path(relative).name)
    named_size = np.array([float(v) / 100 for v in match.groups()]) if match else None
    if size.shape != (3,) or not np.isfinite(size).all() or np.any(size <= 0) or named_size is None or not np.allclose(size, named_size, atol=1e-6, rtol=0):
        asset["binding_review_reason"] = "Author dimensions and training asset filename disagree"
        return
    if str(data["object_xml_geom_quat_mode"].item()) != "post":
        asset["binding_review_reason"] = "Unknown object geometry quaternion composition"
        return
    input_q = np.asarray(data["object_quat_input_wxyz"], dtype=np.float64)
    geom_q = np.asarray(data["object_geom_quat_wxyz"], dtype=np.float64)
    actual_q = np.asarray(data["object_quat_w"], dtype=np.float64)
    if input_q.shape != actual_q.shape or geom_q.shape != (4,):
        asset["binding_review_reason"] = "Malformed quaternion composition evidence"
        return
    expected_q = _quat_multiply(input_q, geom_q)
    quaternion_error = np.minimum(np.linalg.norm(actual_q-expected_q, axis=-1),
                                  np.linalg.norm(actual_q+expected_q, axis=-1))
    if not np.isfinite(quaternion_error).all() or np.max(quaternion_error) > 1e-5:
        asset["binding_review_reason"] = "Stored object orientation differs from documented post-composition"
        return
    try:
        low, high = _obj_cuboid_bounds(asset["mesh_path"])
    except OmniContactFormatError as exc:
        asset["binding_review_reason"] = str(exc)
        return
    asset.update(
        kind="cuboid", size_m=size, geometry_ready=True,
        binding_status="resolved_author_metadata_centered_cuboid",
        local_position_m=np.zeros(3), local_quaternion_wxyz=np.array([1., 0., 0., 0.]),
        source_obj_center_units=(low + high) / 2,
        source_obj_scale_to_m_per_axis=size / (high-low),
        source_scale_applied_again=False, source_geom_quaternion_applied_again=False,
        reason="Exact affine recenter/resize of the released eight-corner OBJ using author final dimensions; preserve supplied centered object pose. This resolves annotation geometry, not simulator material/inertia equivalence.",
        binding_evidence={
            "dimension_field": "carry_box_selected_size_cm", "filename_dimensions_match": True,
            "author_method": method, "scope": "all_frames",
            "original_mesh_is_eight_corner_six_face_cuboid": True,
            "object_quaternion_postcomposition_max_error": float(np.max(quaternion_error)),
            "contact_labels_used_to_fit_geometry": False,
            "source_obj_transform": "(original_vertices - source_obj_center_units) * source_obj_scale_to_m_per_axis",
            "training_asset_mapping": "Ingrid789/OmniContact assets/object_spawner.py: box_<filename_dimensions>; revision " + TRAINING_REVISION,
        },
    )


def _object_asset(data, root: Path, relative: str) -> dict:
    metadata = {key: np.asarray(data[key]).tolist() for key in data.files
                if (key.startswith("object_") and np.asarray(data[key]).size <= 16
                    and key not in ("object_pos_w", "object_quat_w"))
                or (key.startswith("carry_") and np.asarray(data[key]).size <= 100)
                or key == "scale"}
    name = str(np.asarray(data["object_contact_name"]).item()) if "object_contact_name" in data else None
    candidates = {"Box_H_1": "Box_H_1.obj", "Box_I_1": "Box_I_1.obj", "Box_K_1": "Box_K_1.obj"}
    inferred_name = False
    if name is None and "/soccer/" in "/" + relative:
        name = "ScoccerBall_A_1"
        inferred_name = True
    mesh_relative = candidates.get(name)
    if name in ("ScoccerBall_A_1", "SoccerBall_A_1", "soccerball_a_1"):
        mesh_relative = "ScoccerBall_A_1.obj"
    mesh = root / "assets/objects" / mesh_relative if mesh_relative else None
    present = mesh is not None and mesh.is_file()
    size = data["carry_box_selected_size_cm"].astype(np.float64) / 100 if "carry_box_selected_size_cm" in data else None
    size_origin = "carry_box_selected_size_cm" if size is not None else None
    if size is None:
        match = re.search(r"(?:carry|push|slide)_box_(\d+)-(\d+)-(\d+)_", Path(relative).name)
        if match:
            size = np.array([float(v) / 100 for v in match.groups()])
            size_origin = "filename; official training asset_label convention, binding unverified"
    asset = {
        "partner_id": "object", "kind": "mesh", "source_object_name": name,
        "mesh_path": str(mesh) if present else None,
        "mesh_sha256": _sha256(mesh) if present else None,
        "mesh_unit_scale_m": 0.01 if present else None,
        "source_scale": np.array(data["scale"], copy=True) if "scale" in data else None,
        "object_mesh_scale": np.array(data["object_mesh_scale"], copy=True) if "object_mesh_scale" in data else None,
        "declared_size_m": size, "declared_size_source": size_origin,
        "raw_metadata": metadata, "name_inferred_from_task": inferred_name,
        "binding_status": "unverified_augmented_geometry_and_object_frame" if present else "missing_mesh_binding",
        "geometry_ready": False,
        "reason": "Released meshes are offset and in cm. Per-clip augmentation, geom quaternion and center compensation must be resolved before geometry labels; do not apply the compensation twice.",
    }
    _bind_carry_cuboid(data, asset, relative)
    return asset


def _supports(data, count: int) -> list[dict]:
    result = []
    for partner in ("table1", "table2"):
        key = partner + "_pos_w"
        if key not in data:
            continue
        position = _array(data, key, (count, 3))
        parked = bool(np.all(position[:, 2] < -5.0))
        official_uses = bool(np.any(np.abs(position) > 1e-6) and not parked) if partner == "table1" else None
        result.append({
            "partner_id": partner, "kind": "cuboid", "size_m": np.array([0.3, 0.25, 0.01]),
            "position": position, "quaternion": np.tile([1., 0., 0., 0.], (count, 1)),
            "kinematic": True, "source_field": key, "source_present": True,
            "parked_below_scene": parked, "official_table1_uses": official_uses,
            "binding_status": "official_recipe_found_release_binding_unverified",
            "recipe": {"repository": "Ingrid789/OmniContact", "revision": TRAINING_REVISION,
                       "path": "source/omnicontact/omnicontact/tasks/omnicontact/env_cfg.py",
                       "sha256": "d737b55175c0a539d0862698a527dbf00b09b95e3e111b175b45249591b69c6c",
                       "lines": [72, 121]},
        })
    return result


def load_clip(path_or_record, *, source_root=None, manifest_path=None, expected_joint_names=None) -> dict:
    """Load one uncut clip into world-frame pelvis/SDK-order canonical arrays.

    Unnamed inputs require the exact official layout evidence. Explicit
    joint_order_names/body_names can establish their own permutation/root index.
    Quaternions are WXYZ as specified by the release; invalid norms are errors.
    All returned arrays are copies. Only verified author carry-box geometry is
    annotation-ready; other object bindings retain explicit review status.
    """
    record = dict(path_or_record) if isinstance(path_or_record, dict) else {}
    path = Path(record.get("path", path_or_record)).expanduser()
    if source_root is None:
        source_root = record.get("source_root") or _source_root(path.resolve())
    root = Path(source_root).resolve()
    path = (root / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_relative_to(root):
        raise OmniContactFormatError("Clip must reside inside source_root")
    relative = path.relative_to(root).as_posix()
    if not record:
        record = next((r for r in list_clips(root, manifest_path) if r["relative_path"] == relative), {})
    expected = tuple(expected_joint_names) if expected_joint_names is not None else G1_JOINT_NAMES
    if expected != G1_JOINT_NAMES:
        raise OmniContactFormatError("Canonical joint names differ from verified SuperMimic SDK order")
    review = []
    evidence = {}
    with np.load(path, allow_pickle=False) as data:
        if "joint_pos" not in data or data["joint_pos"].ndim != 2:
            raise OmniContactFormatError("joint_pos must be [N,29]")
        count = len(data["joint_pos"])
        if count < 2:
            raise OmniContactFormatError("A continuous reference needs at least two frames")
        joints = _array(data, "joint_pos", (count, 29))
        if "joint_order_names" in data:
            names = [str(v) for v in data["joint_order_names"].reshape(-1)]
            if len(names) != 29 or set(names) != set(expected):
                raise OmniContactFormatError("joint_order_names must name each of the 29 canonical joints once")
            joints = joints[:, [names.index(name) for name in expected]]
            evidence["joint_evidence"] = "NPZ joint_order_names"
        else:
            evidence.update(_release_layout(str(root)))
            joints = joints[:, ISAAC_TO_G1]
        if "body_names" in data:
            names = [str(v) for v in data["body_names"].reshape(-1)]
            if names.count("pelvis") != 1 or len(names) != data["body_pos_w"].shape[1]:
                raise OmniContactFormatError("body_names must identify exactly one pelvis")
            root_index = names.index("pelvis")
            evidence["root_evidence"] = "NPZ body_names pelvis"
        else:
            evidence.update(_release_layout(str(root)))
            root_index = 0
        body_count = data["body_pos_w"].shape[1]
        body_pos = _array(data, "body_pos_w", (count, body_count, 3))
        body_quat = _array(data, "body_quat_w", (count, body_count, 4))
        root_pos = body_pos[:, root_index].copy()
        root_quat = _quat(body_quat[:, root_index].copy(), "pelvis")
        obj_pos = _array(data, "object_pos_w", (count, 3))
        obj_quat = _quat(_array(data, "object_quat_w", (count, 4)), "object")
        if "fps" not in data or np.asarray(data["fps"]).size != 1:
            raise OmniContactFormatError("A single explicit fps is required")
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        if not np.isfinite(fps) or fps <= 0:
            raise OmniContactFormatError("fps must be finite and positive")
        if "source_start_frame" in record:
            expected_count = record["source_end_frame_inclusive"] - record["source_start_frame"] + 1
            if count != expected_count:
                raise OmniContactFormatError("NPZ length disagrees with official inclusive source interval")
        contacts = None
        if "contact_info" in data:
            values = _array(data, "contact_info", (count, 4, 1), dtype=data["contact_info"].dtype)
            if not np.isin(values, [0, 1]).all():
                raise OmniContactFormatError("contact_info must be binary")
            contacts = {"values": values[:, :, 0], "channel_names": list(CONTACT_CHANNELS),
                        "partner_ids": ["object"] * 4, "source_field": "contact_info",
                        "source_shape": [count, 4, 1], "type": "reference_intent_heuristic",
                        "is_force_or_impulse": False, "missing_body_parts": "unknown_not_negative"}
        else:
            review.append("missing_source_contact_info")
        asset = _object_asset(data, root, relative)
        supports = _supports(data, count)
    jumps = {}
    for label, positions, threshold in (("pelvis", root_pos, 0.5), ("object", obj_pos, 0.75)):
        distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        frames = np.flatnonzero(distances > threshold) + 1
        if frames.size:
            jumps[label] = {"clip_frames": frames.tolist(), "jump_m": distances[frames - 1].tolist(),
                            "threshold_m_per_frame": threshold}
    if jumps:
        review.append("spatial_discontinuity_requires_review_not_a_confirmed_cut")
    if not record.get("source_group_verified", False):
        review.append("source_group_mapping_unverified")
    if not asset["geometry_ready"]:
        review.append("object_geometry_binding_unverified")
    provenance = {key: record.get(key) for key in (
        "source_group_id", "capture_id", "split", "source_path", "source_start_frame",
        "source_end_frame_inclusive", "mapping_status", "source_group_verified", "task")}
    provenance.update(dataset="omnicontact", revision=REVISION, source_file=str(path), relative_path=relative,
                      sha256=_sha256(path), root_body="pelvis", root_body_index=root_index,
                      coordinate_frame="source_absolute_world", position_units="m", quaternion_order="WXYZ",
                      source_fps=fps, source_frames=count, timestamp_origin="clip_first_frame",
                      resampled=False, cut=False, repaired=False, layout_evidence=evidence,
                      review_flags=review, spatial_discontinuities=jumps)
    return {"clip_id": relative, "timestamps_s": np.arange(count, dtype=np.float64) / fps,
            "root_pos": root_pos, "root_quat": root_quat, "joint_pos": joints,
            "joint_names": list(expected), "object_pos": obj_pos, "object_quat": obj_quat,
            "object_asset": asset, "scene_supports": supports,
            "provenance": provenance, "source_contacts": contacts}
