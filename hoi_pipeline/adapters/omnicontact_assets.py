"""Explicit, hash-bound OmniContact training assets; never fit geometry to contact."""
from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import re

import numpy as np

SCHEMA = "omnicontact_author_assets_v1"
DATASET_REVISION = "068b83049df50653327b06ee41bbbaf1b9bd593e"
TRAINING_REVISION = "19cde38364a5b48e47348b270c895eabe2ba3ad9"
AUTHOR_CODE = {
    "assets/object_spawner.py": "bd37f9c16bf7f31217a5e9181f6fb33af1340e2d39a5aebfc6cb99b9759574b8",
    "source/omnicontact/omnicontact/utils/motion_labels.py": "13e58f73756b1a36795c5b2bad06b4694d2bb28387f2eb2f3b1d79fa9a7e5206",
    "source/omnicontact/omnicontact/tasks/omnicontact/mdp/commands.py": "f1e356a97dc7580eb5c81b1c119b4c8a15c1a1bfdeafdeb08613582f0b270d04",
}


def asset_label(filename):
    """Exact pinned training filename convention, with push material prefix removed."""
    suffix = r"(?:\d+_)*\d{3}"
    match = re.fullmatch(rf"(?:carry|push|slide)_box_(\d+(?:-\d+){{2}})_{suffix}\.npz", filename, re.I)
    if match:
        return "box_" + match.group(1)
    match = re.fullmatch(rf"(?:(?:relocate|pkick)_soccer|kick_ball)_(\d+)_{suffix}\.npz", filename, re.I)
    if match:
        return "soccer_" + match.group(1)
    raise ValueError("Filename has no verified author training asset mapping: " + filename)


@lru_cache(maxsize=512)
def _verified_bytes(path, fingerprint, expected):
    if _fingerprint(Path(path).stat()) != fingerprint:
        raise ValueError("Author binding file changed before verification: " + path)
    raw = Path(path).read_bytes()
    if _fingerprint(Path(path).stat()) != fingerprint:
        raise ValueError("Author binding file changed during verification: " + path)
    if len(raw) != fingerprint[0] or hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("Author binding dependency SHA mismatch: " + path)
    return raw


def _fingerprint(stat):
    return (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino, stat.st_dev)


def verify_file(item):
    path = Path(item["path"]).resolve(strict=True)
    stat = path.stat()
    if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
        raise ValueError("Invalid author binding file identity")
    if "size" in item and item["size"] != stat.st_size:
        raise ValueError("Author binding dependency size mismatch: " + str(path))
    return _verified_bytes(str(path), _fingerprint(stat), item["sha256"])


def _lookup(binding, clip_id):
    if binding.get("schema") != SCHEMA or binding.get("clip_id") != clip_id:
        raise ValueError("Author asset binding schema/clip identity differs")
    catalog_file = {"path": binding["catalog_path"], "sha256": binding["catalog_sha256"]}
    catalog = json.loads(verify_file(catalog_file))
    if (catalog.get("schema") != SCHEMA or catalog.get("dataset_revision") != DATASET_REVISION
            or catalog.get("training_revision") != TRAINING_REVISION):
        raise ValueError("Author asset catalog revision/schema differs")
    code = catalog.get("author_code", {})
    if set(code) != set(AUTHOR_CODE):
        raise ValueError("Author training pose/asset code evidence is incomplete")
    for name, expected in AUTHOR_CODE.items():
        if code[name]["sha256"] != expected:
            raise ValueError("Author training code revision differs: " + name)
        verify_file(code[name])
    label = asset_label(Path(clip_id).name)
    if label != binding.get("asset_label"):
        raise ValueError("Bound asset label differs from author filename mapping")
    entry = catalog["assets"][label]
    geometry = entry["geometry"]
    if (entry.get("status") != "verified" or geometry.get("basis") != "author_USD_collision_in_rigid_body_frame"
            or geometry.get("root_pose_replaced_by_recorded_pose") is not True
            or not geometry.get("root_path") or not geometry.get("shapes")):
        raise ValueError("Author asset has no verified rigid-body collision geometry")
    files = entry.get("dependencies", [])
    if not files or not any(str(f["path"]).endswith((".usd", ".usda", ".usdc")) for f in files):
        raise ValueError("Author asset USD dependencies are missing")
    if not any(str(f["path"]).endswith(".urdf") for f in files):
        raise ValueError("Author asset independent URDF evidence is missing")
    for item in files:
        verify_file(item)
    return deepcopy(entry), [catalog_file, *code.values(), *files]


def binding_files(record):
    """Freeze must include the entire explicit binding, not just its JSON pointer."""
    binding = record.get("author_geometry_binding")
    if binding is None:
        return []
    _, dependencies = _lookup(binding, record["clip_id"])
    source = next((item for item in record.get("_input_files", [])
                   if Path(item["path"]).resolve() == Path(record["path"]).resolve()), None)
    if source is None or source["sha256"] != binding.get("source_motion_sha256"):
        raise ValueError("Author binding motion SHA differs from frozen source record")
    return dependencies


def bind_asset(asset, data, record, relative, motion_sha256):
    binding = record["author_geometry_binding"]
    if binding.get("source_motion_sha256") != motion_sha256:
        raise ValueError("Author binding motion SHA mismatch")
    entry, _ = _lookup(binding, relative)
    flags = []
    label = binding["asset_label"]
    if label.startswith("box_"):
        named_size = np.asarray(label[4:].split("-"), dtype=float) / 100
        declared = entry["urdf_geometry"]
        if declared.get("kind") != "cube" or not np.allclose(declared["size"], named_size, atol=1e-6, rtol=0):
            raise ValueError("Author box filename and independent URDF dimensions disagree")
        for field in ("object_size_m_xyz", "object_grounding_box_size_m"):
            if field in data:
                value = np.asarray(data[field], dtype=float).reshape(-1)
                if value.shape != (3,) or not np.isfinite(value).all():
                    raise ValueError("Malformed source dimension metadata: " + field)
                if not np.allclose(value, named_size, atol=1e-6, rtol=0):
                    flags.append("author_asset_differs_from_source_" + field)
    # The fixed MotionCommand consumes object_pos_w/quat_w directly; scale,
    # offsets, grounding, and XML quaternion composition are history, not new ops.
    asset.update(kind="author_collision", geometry_ready=True,
                 binding_status="verified_author_training_asset", author_geometry=entry["geometry"],
                 author_binding=deepcopy(binding), author_asset_files=entry["dependencies"],
                 author_urdf_geometry=entry["urdf_geometry"], binding_review_flags=flags,
                 source_scale_applied_again=False, source_center_applied_again=False,
                 source_geom_quaternion_applied_again=False,
                 reason="Pinned author filename-to-asset and MotionCommand root-state contract; recorded world pose unchanged. Historical metadata conflicts remain review flags, not pose corrections.")
    return flags
