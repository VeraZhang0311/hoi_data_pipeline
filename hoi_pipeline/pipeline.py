from __future__ import annotations
import json
import os
from pathlib import Path
import tempfile
import time
from collections import OrderedDict
import numpy as np
from scipy.spatial.transform import Rotation
from . import SCHEMA
from .common import BODY_NAMES, identity, canonical_json, sha256, transform, points_transform
from .kinematics import RobotFK
from .timebase import validate, resample, events, interval_evidence
from .geometry import Shape, group_gap, classify

def object_recipe(asset):
    if "recipe_entry" in asset:
        entry=asset["recipe_entry"];shapes=[]
        for mesh in entry["collision_meshes"]:
            vertices=np.asarray(mesh["vertices"],dtype=float)
            if mesh.get("frame","mesh") != "object_local":
                vertices=points_transform(vertices*np.asarray(entry["exact_scale"]),np.asarray(entry["mesh_to_object"]))
            shapes.append({"kind":"mesh","vertices":vertices.tolist(),"faces":mesh["faces"]})
        return {"shapes":shapes,"basis":"ULTRA_frozen_native_object_local_collision_surface","source":asset["recipe_evidence"]}
    if "meshes" in asset:
        import trimesh
        shapes=[]
        meshes=[m for m in asset["meshes"] if m["role"]=="visual"] or asset["meshes"]
        for item in meshes:
            if sha256(item["path"]) != item["sha256"]: raise ValueError("Object mesh changed")
            mesh=trimesh.load(item["path"],force="mesh",process=False)
            local=np.eye(4);local[:3,:3]=Rotation.from_euler("xyz",item["origin_rpy"]).as_matrix();local[:3,3]=item["origin_xyz"]
            vertices=points_transform(np.asarray(mesh.vertices)*np.asarray(item["scale"]),local)*asset["source_scale"]
            shapes.append({"kind":"mesh","vertices":vertices.tolist(),"faces":np.asarray(mesh.faces).tolist()})
        return {"shapes":shapes,"basis":"source_URDF_visual_surface_scaled_once","source":asset}
    if asset.get("geometry_ready") and asset.get("kind")=="cuboid":
        return {"shapes":[{"kind":"cube","size":np.asarray(asset["size_m"]).tolist()}],"basis":"verified_source_cuboid","source":asset}
    if "usd_path" in asset or "path" in asset and str(asset["path"]).endswith((".usd",".usda",".usdc")):
        from .usd_geometry import object_geometry
        return object_geometry(asset.get("usd_path",asset.get("path")),asset["scale"])
    raise ValueError("Object geometry binding requires review: "+str(asset.get("binding_status",list(asset))))

def discontinuities(clip):
    t=validate(clip);dt=np.diff(t);issues=[]
    if dt.max()>dt.min()*1.5: issues.append("irregular_source_time_grid")
    for key,limit in (("root_pos",8.),("object_pos",25.)):
        speed=np.linalg.norm(np.diff(clip[key],axis=0),axis=1)/dt
        if np.any(speed>limit): issues.append(f"{key}_discontinuity_candidate")
    return issues

class Labeler:
    def __init__(self,robot_recipe,urdf_path,joint_names,on=.02,off=.04,hz=50.):
        if not (0<=on<off): raise ValueError("Bad geometric thresholds")
        self.recipe=robot_recipe;self.fk=RobotFK(urdf_path,joint_names)
        self.robot={name:[] for name in BODY_NAMES}
        for spec in robot_recipe["shapes"]: self.robot[spec["body"]].append(Shape(spec))
        self.on,self.off,self.hz=on,off,hz;self.cache=OrderedDict()

    def _measure(self,poses,obj):
        fk=self.fk(poses["root_pos"],poses["root_quat"],poses["joint_pos"])
        n=len(poses["timestamps_s"]);c=len(BODY_NAMES)+1
        distance=np.full((n,c),np.nan,np.float32);exact=np.zeros((n,c),bool);intersection=np.zeros((n,c),bool)
        for t in range(n):
            obj_pose=transform(poses["object_pos"][t],poses["object_quat"][t])
            for shape in obj: shape.place(obj_pose)
            for i,name in enumerate(BODY_NAMES):
                body=self.robot[name]
                for shape in body: shape.place(fk[name][t])
                distance[t,i],exact[t,i],intersection[t,i]=group_gap(body,obj,self.off)
            z=min(s.min_z() for s in obj)
            distance[t,-1]=max(0,z);exact[t,-1]=True;intersection[t,-1]=z < -1e-6
        labels=classify(distance,exact,intersection,self.on,self.off)
        return {"distance_m":distance,"distance_exact":exact,"intersection":intersection,"labels":labels,"valid":labels>=0}

    def label(self,clip):
        validate(clip)
        issues=sorted(set(discontinuities(clip)+list(clip.get("provenance",{}).get("review_flags",[]))))
        if "irregular_source_time_grid" in issues:
            raise ValueError("Source gaps require explicit interval segmentation; no labels interpolated across missing time")
        if tuple(clip["joint_names"])!=self.fk.names: raise ValueError("Joint order mismatch")
        asset_key=identity(clip["object_asset"])
        if asset_key not in self.cache:
            recipe=object_recipe(clip["object_asset"])
            self.cache[asset_key]=(recipe,[Shape(s) for s in recipe["shapes"]])
            while len(self.cache)>8: self.cache.popitem(last=False)
        self.cache.move_to_end(asset_key)
        recipe,obj=self.cache[asset_key]
        target=resample(clip,self.hz)
        native=self._measure(clip,obj)
        if len(target["timestamps_s"])==len(clip["timestamps_s"]) and np.allclose(target["timestamps_s"],clip["timestamps_s"],rtol=0,atol=1e-10):
            canonical=native
        else: canonical=self._measure(target,obj)
        positive,known=interval_evidence(clip["timestamps_s"],native["labels"],target["timestamps_s"])
        arrays={k:target[k] for k in ("timestamps_s","root_pos","root_quat","joint_pos","object_pos","object_quat")}
        arrays.update({"target_"+k:v for k,v in canonical.items()})
        arrays.update({"source_"+k:v for k,v in native.items()})
        arrays["source_timestamps_s"]=clip["timestamps_s"]
        arrays["interval_source_positive_fraction"]=positive;arrays["interval_source_known_fraction"]=known
        arrays["interval_source_any_contact"]=positive>0
        active=[bool(self.robot[name]) for name in BODY_NAMES]+[True]
        source_events=events(clip["timestamps_s"],native["labels"])
        target_events=events(target["timestamps_s"],canonical["labels"])
        metadata={"schema":SCHEMA,"clip_id":clip["clip_id"],"provenance":clip["provenance"],
                  "timebase":target["timebase"],"channels":[x+"__object" for x in BODY_NAMES]+["object__ground"],
                  "active_geometry_channels":active,"robot_geometry_sha256":identity(self.recipe),
                  "object_geometry_sha256":identity(recipe),"object_geometry":recipe,
                  "object_asset":clip["object_asset"],"thresholds_m":{"on":self.on,"off":self.off},
                  "semantics":{"1":"geometric_proximity","0":"geometrically_separated","-1":"unknown_or_no_geometry",
                               "not_force_ground_truth":True,"intersections":"unknown","distance_exact_false":"conservative_AABB_lower_bound",
                               "interval_fractions":"nearest_source_sample_time_support_not_contact_probability",
                               "ground_plane_z_m":0.},
                  "source_events":source_events,"target_events":target_events,"review_flags":issues,
                  "training_ready":False,"source_contacts":clip.get("source_contacts"),"scene_supports":clip.get("scene_supports",[]),
                  "statistics":{"source_frames":len(clip["timestamps_s"]),"target_frames":len(target["timestamps_s"]),
                                "source_event_count":len(source_events),"target_event_count":len(target_events),
                                "positive_frames_by_channel":(canonical["labels"]==1).sum(0),
                                "unknown_frames_by_channel":(canonical["labels"]<0).sum(0),
                                "intersecting_frames_by_channel":canonical["intersection"].sum(0)}}
        # Store large geometry once per asset outside per-clip metadata in run().
        return arrays,metadata

def _extract_arrays(value,arrays,path="metadata"):
    if isinstance(value,np.ndarray):
        arrays[path]=value;return {"npz_key":path}
    if isinstance(value,np.generic): return _extract_arrays(value.item(),arrays,path)
    if isinstance(value,dict): return {str(k):_extract_arrays(v,arrays,path+"/"+str(k)) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        try: candidate=np.asarray(value)
        except (ValueError,TypeError): candidate=None
        if candidate is not None and candidate.dtype.kind in "biuf" and candidate.ndim>0:
            arrays[path]=candidate;return {"npz_key":path}
        return [_extract_arrays(v,arrays,path+"/"+str(i)) for i,v in enumerate(value)]
    if isinstance(value,Path): return str(value)
    if isinstance(value,float) and not np.isfinite(value):
        arrays[path]=np.asarray(value);return {"npz_key":path,"source_nonfinite_annotation":True}
    return value

def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+f".{os.getpid()}.tmp")
    temp.write_text(canonical_json(value),encoding="utf8");os.replace(temp,path)

def save_clip(directory,arrays,metadata):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False)
    arrays=dict(arrays);metadata=_extract_arrays(metadata,arrays)
    temporary=directory/"labels.partial.npz";final=directory/"labels.npz"
    np.savez_compressed(temporary,**arrays);os.replace(temporary,final)
    metadata["array_file"]={"path":"labels.npz","sha256":sha256(final)}
    atomic_json(directory/"metadata.json",metadata)
    atomic_json(directory/"SUCCESS.json",{"schema":SCHEMA,"metadata_sha256":sha256(directory/"metadata.json"),"array_sha256":sha256(final)})
