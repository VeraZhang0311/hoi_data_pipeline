"""Build an explicit OmniContact catalog from pinned, independently audited assets.

Reopens the actual USD and URDF; audit JSON is evidence, not executable geometry.
Only the reviewed centered boxes and convex soccer surfaces are admitted.
"""
from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from .adapters.omnicontact_assets import AUTHOR_CODE, DATASET_REVISION, TRAINING_REVISION, SCHEMA, verify_file
from .common import sha256
from .pipeline import atomic_json

AUDIT_SHA256 = "1cdbfbd825374227399ef1114bb1433d19386753839543e1b984661f04d81e5c"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_identity(path):
    path = Path(path).resolve(strict=True)
    return {"path": str(path), "sha256": sha256(path), "size": path.stat().st_size}


def checked_surface(vertices, faces):
    """Preserve the authored mesh only when it already describes a closed convex solid."""
    import trimesh
    vertices = np.asarray(vertices, dtype=float)
    faces = np.asarray(faces, dtype=np.int64)
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all(), "Invalid vertices")
    require(faces.ndim == 2 and faces.shape[1] == 3 and len(faces) > 0
            and faces.min() >= 0 and faces.max() < len(vertices), "Invalid faces")
    mesh = trimesh.Trimesh(vertices, faces, process=False)
    require(np.all(mesh.area_faces > 1e-14), "Degenerate mesh face")
    require(mesh.is_watertight and mesh.is_winding_consistent and mesh.volume > 0, "Surface is not an outward closed solid")
    # Every vertex must lie behind every outward face, within float export noise.
    violation = float(np.max(mesh.face_normals @ vertices.T - np.einsum("ij,ij->i", mesh.face_normals, mesh.triangles[:, 0])[:, None]))
    require(violation <= 1e-6, "Author mesh is not convex; do not silently fit a hull")
    return {"kind": "mesh", "vertices": vertices.tolist(), "faces": faces.tolist()}, violation


def extract_asset(root, entry):
    from pxr import Usd, UsdGeom, UsdPhysics, UsdUtils
    label = entry["label"]
    files = []
    allowed = set()
    for item in entry["files"]:
        path = (root / item["relative_path"]).resolve(strict=True)
        require(path.is_relative_to(root), "Asset dependency escapes asset root")
        identity = {"path": str(path), "sha256": item["sha256"], "size": item["size_bytes"]}
        verify_file(identity)
        files.append(identity)
        allowed.add(path)
    usd = (root / entry["selected_entry_usd"]).resolve(strict=True)
    require(usd in allowed, "USD entry missing from hash manifest")
    layers, external, unresolved = UsdUtils.ComputeAllDependencies(str(usd))
    require(not unresolved and not external, "USD has unresolved or unsupported external dependencies")
    require(all(Path(layer.realPath).resolve() in allowed for layer in layers if layer.realPath), "USD dependency is not hash-bound")
    stage = Usd.Stage.Open(str(usd))
    require(stage and UsdGeom.GetStageMetersPerUnit(stage) == 1 and str(UsdGeom.GetStageUpAxis(stage)) == "Z", "Unexpected USD axes/units")
    prims = list(Usd.PrimRange.Stage(stage, Usd.TraverseInstanceProxies()))
    bodies = [p for p in prims if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    require(len(bodies) == 1 and str(bodies[0].GetPath()) == entry["rigid_body_root"] == "/object", "Unexpected rigid root")
    cache = UsdGeom.XformCache()
    body_world = cache.GetLocalToWorldTransform(bodies[0])
    require(np.allclose(body_world, np.eye(4), atol=1e-9, rtol=0), "Unreviewed authored rigid root transform")
    colliders = [p for p in prims if p.HasAPI(UsdPhysics.CollisionAPI) and UsdPhysics.CollisionAPI(p).GetCollisionEnabledAttr().Get()]
    require(len(colliders) == 1, "Unexpected collider count")
    collider = colliders[0]
    geometries = [p for p in Usd.PrimRange(collider, Usd.TraverseInstanceProxies()) if p.GetTypeName() in ("Cube", "Mesh", "Sphere", "Capsule", "Cylinder")]
    require(len(geometries) == 1, "Unexpected collider subtree geometry")
    prim = geometries[0]
    matrix = np.asarray(cache.GetLocalToWorldTransform(prim) * body_world.GetInverse())
    urdfs = [Path(x["path"]) for x in files if x["path"].endswith(".urdf")]
    require(len(urdfs) == 1, "Independent URDF is ambiguous")
    xml = ET.parse(urdfs[0]).getroot()
    collisions = xml.findall("link/collision")
    require(len(collisions) == 1 and xml.find("link").get("name") == "object", "Unexpected URDF collision frame")
    origin = collisions[0].find("origin")
    xyz = np.fromstring(origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" ")
    rpy = np.fromstring(origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" ")
    geom = list(collisions[0].find("geometry"))
    require(len(geom) == 1, "Unexpected URDF geometry count")
    g = geom[0]
    geometry = {"basis": "author_USD_collision_in_rigid_body_frame", "root_path": "/object",
                "root_pose_replaced_by_recorded_pose": True, "shapes": []}
    if label.startswith("box_"):
        require(prim.IsA(UsdGeom.Cube) and g.tag == "box", "Expected box geometry")
        size = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) * np.diag(matrix)[:3]
        require(np.all(size > 0) and np.allclose(matrix, np.diag(np.diag(matrix)), atol=1e-9, rtol=0), "Unreviewed cube rotation/offset")
        urdf_size = np.fromstring(g.get("size"), sep=" ")
        named_size = np.asarray(label[4:].split("-"), dtype=float) / 100
        require(np.allclose(xyz, 0) and np.allclose(rpy, 0) and np.allclose(size, urdf_size, atol=1e-6, rtol=0)
                and np.allclose(size, named_size, atol=1e-6, rtol=0), "Box USD/URDF/name disagree")
        geometry["shapes"] = [{"kind": "cube", "size": size.tolist()}]
        urdf_geometry = {"kind": "cube", "size": urdf_size.tolist()}
    else:
        require(label in ("soccer_20", "soccer_22") and prim.IsA(UsdGeom.Mesh) and g.tag == "mesh", "Unreviewed mesh asset")
        raw_schemas = list(collider.GetMetadata("apiSchemas").GetAppliedItems())
        includes = [str(p) for p in collider.GetRelationship("collection:collisionmeshes:includes").GetTargets()]
        require("PhysxMeshMergeCollisionAPI" in raw_schemas and includes == [str(collider.GetPath())]
                and str(UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()) == "convexHull", "Unreviewed mesh merge/collision semantics")
        mesh = UsdGeom.Mesh(prim)
        require(str(mesh.GetOrientationAttr().Get()) == "rightHanded" and str(mesh.GetSubdivisionSchemeAttr().Get()) == "none", "Unreviewed mesh surface semantics")
        points = np.asarray(mesh.GetPointsAttr().Get(), dtype=float)
        vertices = points @ matrix[:3, :3] + matrix[3, :3]
        counts = list(mesh.GetFaceVertexCountsAttr().Get()); indices = list(mesh.GetFaceVertexIndicesAttr().Get())
        faces = []; cursor = 0
        for count in counts:
            require(count >= 3, "Invalid polygon")
            face = indices[cursor:cursor + count]; cursor += count
            require(len(face) == count, "Truncated polygon")
            faces.extend([[face[0], face[i], face[i + 1]] for i in range(1, count - 1)])
        require(cursor == len(indices), "Extra face indices")
        surface, violation = checked_surface(vertices, faces)
        obj = (urdfs[0].parent / g.get("filename")).resolve(strict=True)
        require(obj in allowed, "URDF mesh is not hash-bound")
        obj_vertices = np.asarray([[float(x) for x in line.split()[1:4]] for line in obj.read_text().splitlines() if line.startswith("v ")])
        scale = np.fromstring(g.get("scale", "1 1 1"), sep=" ")
        urdf_points = (obj_vertices * scale) @ Rotation.from_euler("xyz", rpy).as_matrix().T + xyz
        require(len(urdf_points) == len(vertices) and max(cKDTree(vertices).query(urdf_points)[0].max(), cKDTree(urdf_points).query(vertices)[0].max()) < 1e-6, "USD and URDF mesh surfaces disagree")
        geometry.update(shapes=[surface], collision_approximation="convexHull",
                        surface_representation="authored closed convex triangle surface; original vertices preserved",
                        maximum_convexity_violation_m=violation, physx_cooked_equivalence_verified=False,
                        limitation="Native PhysX cooked vertex reduction and contact/rest offsets are not reconstructed.")
        urdf_geometry = {"kind": "mesh", "mesh_sha256": sha256(obj), "scale": scale.tolist(), "origin_xyz": xyz.tolist(), "origin_rpy": rpy.tolist()}
    return {"status": "verified", "geometry": geometry, "urdf_geometry": urdf_geometry, "dependencies": files,
            "source_folder_url": entry["folder_url"]}


def build_catalog(asset_root, author_code_root):
    root = Path(asset_root).resolve(strict=True)
    audit_file = {"path": str(root / "geometry_audit.json"), "sha256": AUDIT_SHA256}
    audit = json.loads(verify_file(audit_file))
    code = {name: {"path": str((Path(author_code_root) / name).resolve(strict=True)), "sha256": digest} for name, digest in AUTHOR_CODE.items()}
    for item in code.values():
        verify_file(item)
    assets = {}
    for label, entry in sorted(audit["assets"].items()):
        require(label == entry["label"], "Audit label mismatch")
        assets[label] = extract_asset(root, entry)
        assets[label]["dependencies"].append(audit_file)
    require(len(assets) == 57, "Incomplete reviewed asset set")
    require(assets["soccer_20"]["geometry"] == assets["soccer_22"]["geometry"], "Soccer aliases no longer have identical geometry")
    return {"schema": SCHEMA, "dataset_revision": DATASET_REVISION, "training_revision": TRAINING_REVISION,
            "author_code": code, "assets": assets, "independent_audit": audit_file}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", required=True)
    parser.add_argument("--author-code-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    require(not Path(args.output).exists(), "Catalog output already exists")
    atomic_json(Path(args.output), build_catalog(args.asset_root, args.author_code_root))
