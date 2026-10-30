"""Extract composed USD surfaces in their recorded rigid-body frame.

No simulator cooking, asset modification, or replacement foot decomposition.
Mesh collision convexHull uses the existing authored hull points only.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from .common import BODY_NAMES, sha256, points_transform

def _matrix(value): return np.asarray(value, dtype=float).T

def _triangles(counts, indices):
    faces=[]; offset=0
    for n in counts:
        if n != 3:
            # Fan triangulation is invalid for concave USD polygons. Fail closed.
            raise ValueError("USD surface must contain explicitly triangulated faces")
        faces.append(indices[offset:offset+n]); offset+=n
    if offset != len(indices): raise ValueError("USD face topology inconsistent")
    return np.asarray(faces, dtype=np.int32)

def _surface(prim, relative, collision=False):
    from pxr import UsdGeom, UsdPhysics
    kind=prim.GetTypeName()
    matrix=_matrix(relative)
    spec={"name":str(prim.GetPath()),"transform":matrix.tolist()}
    if kind == "Mesh":
        mesh=UsdGeom.Mesh(prim)
        if mesh.GetPointsAttr().ValueMightBeTimeVarying(): raise ValueError("Animated mesh unsupported")
        vertices=np.asarray(mesh.GetPointsAttr().Get(),dtype=float)
        approximation=str(UsdPhysics.MeshCollisionAPI(prim).GetApproximationAttr().Get() or "none")
        vertices=points_transform(vertices,matrix)
        if collision and approximation == "convexHull":
            faces=ConvexHull(vertices).simplices
        elif collision and approximation not in ("none","meshSimplification"):
            raise ValueError(f"Unexported native collision approximation {approximation}: {prim.GetPath()}")
        else:
            if collision and approximation == "meshSimplification":
                raise ValueError("Native simplified mesh must be exported before binding")
            faces=_triangles(mesh.GetFaceVertexCountsAttr().Get(),mesh.GetFaceVertexIndicesAttr().Get())
        spec.update(kind="mesh",vertices=vertices.tolist(),faces=faces.tolist(),approximation=approximation)
        spec.pop("transform")
    elif kind in ("Sphere","Cube","Capsule","Cylinder"):
        shape=getattr(UsdGeom,kind)(prim)
        # FCL supports exact analytic primitives; preserve orientation and scale.
        scale=np.linalg.norm(matrix[:3,:3],axis=0)
        rot=matrix[:3,:3]/scale
        if not np.allclose(rot.T@rot,np.eye(3),atol=1e-7) or np.linalg.det(rot)<0:
            raise ValueError("Sheared/reflected primitive unsupported")
        matrix[:3,:3]=rot;spec["transform"]=matrix.tolist()
        spec["kind"]=kind.lower()
        if kind=="Cube": spec["size"]=(float(shape.GetSizeAttr().Get())*scale).tolist()
        else:
            if not np.allclose(scale,scale[0]): raise ValueError("Nonuniform round primitive scale")
            spec["radius"]=float(shape.GetRadiusAttr().Get())*scale[0]
            if kind != "Sphere":
                spec["height"]=float(shape.GetHeightAttr().Get())*scale[0]
                spec["axis"]=str(shape.GetAxisAttr().Get())
    else: raise ValueError("Unsupported active collision shape "+kind)
    return spec

def stage_dependencies(stage):
    files=[]
    for layer in stage.GetUsedLayers():
        path=layer.realPath
        if path: files.append({"path":path,"sha256":sha256(path)})
    return sorted(files,key=lambda x:x["path"])

def robot_geometry(path, fk, *, coordinate_contract=None):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage=Usd.Stage.Open(str(path))
    if not stage: raise ValueError("Cannot open robot USD")
    source_metadata={"meters_per_unit":UsdGeom.GetStageMetersPerUnit(stage),"up_axis":str(UsdGeom.GetStageUpAxis(stage)),
                     "authored_units":stage.HasAuthoredMetadata("metersPerUnit"),"authored_up_axis":stage.HasAuthoredMetadata("upAxis")}
    if coordinate_contract!={"meters_per_unit":1.0,"up_axis":"Z"}:
        raise ValueError("Explicit v1.2 parent simulation stage meter/Z coordinate contract required")
    if source_metadata["authored_units"] and not np.isclose(source_metadata["meters_per_unit"],1.): raise ValueError("Authored robot units disagree")
    if source_metadata["authored_up_axis"] and source_metadata["up_axis"]!="Z": raise ValueError("Authored robot up-axis disagrees")
    # The v1.2 override layers omit stage metadata. USD's fallback cm/Y is not
    # a geometry transform. Native spawning references these into a metre/Z
    # simulation stage. Reproduce that composition without editing the asset.
    source_default=stage.GetDefaultPrim().GetPath()
    stage=Usd.Stage.CreateInMemory();UsdGeom.SetStageMetersPerUnit(stage,1.);UsdGeom.SetStageUpAxis(stage,UsdGeom.Tokens.z)
    host=UsdGeom.Xform.Define(stage,"/Robot").GetPrim();host.GetReferences().AddReference(str(path),source_default);stage.SetDefaultPrim(host)
    cache=UsdGeom.XformCache()
    named={}
    for prim in Usd.PrimRange(host,Usd.TraverseInstanceProxies()):
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() in BODY_NAMES and prim.GetPath().pathString.startswith("/Robot/Geometry/"):
            if prim.GetName() in named: raise ValueError("Duplicate geometric body frame")
            named[prim.GetName()]=prim
    if "pelvis" not in named: raise ValueError("No canonical pelvis geometric frame")
    rest=fk(np.zeros((1,3)),np.array([[1.,0,0,0]]),np.zeros((1,29)))
    errors={}
    for name,prim in named.items():
        rel,_=cache.ComputeRelativeTransform(prim,named["pelvis"])
        error=float(np.max(np.abs(_matrix(rel)-rest[name][0])))
        errors[name]=error
        if error>2e-4: raise ValueError(f"USD/URDF body frames disagree for {name}: {error}")
    shapes=[];disabled=[]
    for prim in Usd.PrimRange(host,Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI): continue
        if UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get() is False:
            disabled.append(str(prim.GetPath()));continue
        ancestor=prim
        while ancestor and (ancestor.GetName() not in named or ancestor!=named[ancestor.GetName()]): ancestor=ancestor.GetParent()
        if not ancestor: raise ValueError("Active robot collision lacks canonical link "+str(prim.GetPath()))
        if ancestor != named[ancestor.GetName()]: raise ValueError("Ambiguous robot body frame")
        relative,_=cache.ComputeRelativeTransform(prim,ancestor)
        spec=_surface(prim,relative,True);spec["body"]=ancestor.GetName();shapes.append(spec)
    if not shapes: raise ValueError("Robot collision export empty")
    for hand in ("left_rubber_hand","right_rubber_hand"):
        if not any(s["body"]==hand for s in shapes): raise ValueError("Rubber hand collision missing")
    foot_shapes=[s for s in shapes if "HOIUltraFootCollision" in s["name"]]
    if len(foot_shapes)!=10: raise ValueError("Frozen v1.2 expects five hulls per foot")
    if any("Sphere" in s["name"] and s["body"].endswith("ankle_roll_link") for s in shapes):
        raise ValueError("Old foot spheres unexpectedly active")
    return {"shapes":shapes,"body_names":list(BODY_NAMES),"dependencies":stage_dependencies(stage),
            "frame_max_errors":errors,"disabled_collision_paths":disabled,"basis":"v1.2_composed_active_collision_surfaces",
            "source_stage_metadata":source_metadata,"parent_stage_coordinate_contract":coordinate_contract}

def object_geometry(path, scale):
    from pxr import Usd, UsdGeom, UsdPhysics
    stage=Usd.Stage.Open(str(path))
    if not stage: raise ValueError("Cannot open object USD")
    root=stage.GetDefaultPrim()
    if not root: raise ValueError("USD lacks default rigid root")
    root_ops=[str(op.GetOpName()) for op in UsdGeom.Xformable(root).GetOrderedXformOps()]
    if any(op not in ("xformOp:translate","xformOp:orient","xformOp:scale") for op in root_ops):
        raise ValueError("Object root has extra transforms not overwritten by released replay: "+str(root_ops))
    rigid=[p for p in stage.Traverse() if p.HasAPI(UsdPhysics.RigidBodyAPI)]
    if len(rigid)!=1 or rigid[0]!=root: raise ValueError("Object must have one default-prim rigid body")
    if str(UsdGeom.GetStageUpAxis(stage))!="Z" or not np.isclose(UsdGeom.GetStageMetersPerUnit(stage),1.):
        raise ValueError("Explicit source conversion required for non-meter or non-Z-up USD")
    cache=UsdGeom.XformCache();shapes=[]
    # Label reference surface, NOT the approximate cooked chair convex hull.
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if prim.GetTypeName() != "Mesh": continue
        if UsdGeom.Imageable(prim).ComputeVisibility()=="invisible": continue
        rel,reset=cache.ComputeRelativeTransform(prim,root)
        if reset: raise ValueError("Object descendant resets rigid-root transform")
        spec=_surface(prim,rel,False)
        spec["vertices"]=(np.asarray(spec["vertices"])*np.asarray(scale)).tolist()
        shapes.append(spec)
    if not shapes: raise ValueError("No visible reference object surface")
    return {"shapes":shapes,"dependencies":stage_dependencies(stage),"basis":"source_visible_surface_not_native_cooked_collision",
            "root_path":str(root.GetPath()),"root_xform_ops":root_ops,"scale":list(np.broadcast_to(scale,(3,))),"asset_root_transform_replaced_by_recorded_pose":True}
