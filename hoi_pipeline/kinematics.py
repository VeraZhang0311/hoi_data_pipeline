from __future__ import annotations
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from .common import BODY_NAMES, rotation

class RobotFK:
    def __init__(self, urdf_path, joint_names):
        tree = ET.parse(urdf_path).getroot()
        self.names = tuple(joint_names)
        self.links = [x.attrib["name"] for x in tree.findall("link")]
        if not set(BODY_NAMES).issubset(self.links): raise ValueError("URDF lacks canonical G1 links")
        self.joints = []
        children = set()
        for j in tree.findall("joint"):
            parent,child = j.find("parent").attrib["link"],j.find("child").attrib["link"]
            origin = j.find("origin")
            xyz = np.fromstring(origin.get("xyz","0 0 0") if origin is not None else "0 0 0",sep=" ")
            rpy = np.fromstring(origin.get("rpy","0 0 0") if origin is not None else "0 0 0",sep=" ")
            mat = np.eye(4); mat[:3,:3] = Rotation.from_euler("xyz",rpy).as_matrix(); mat[:3,3]=xyz
            axis = j.find("axis"); axis = np.fromstring(axis.get("xyz","1 0 0") if axis is not None else "1 0 0",sep=" ")
            kind = j.get("type")
            if kind not in ("fixed","revolute","continuous"): raise ValueError("Unsupported joint "+kind)
            if xyz.shape!=(3,) or rpy.shape!=(3,) or not np.isfinite(np.r_[xyz,rpy]).all(): raise ValueError("Invalid joint origin")
            if kind!="fixed" and (axis.shape!=(3,) or not np.isfinite(axis).all() or not np.isclose(np.linalg.norm(axis),1.,atol=1e-7)):
                raise ValueError("Movable joint axis must be a unit vector")
            if child in children: raise ValueError("Multiple parents for body "+child)
            index = self.names.index(j.get("name")) if kind != "fixed" else None
            self.joints.append((parent,child,mat,axis,index)); children.add(child)
        roots=set(self.links)-children
        if roots != {"pelvis"}: raise ValueError(f"Unexpected robot root {roots}")
        movable=[j[4] for j in self.joints if j[4] is not None]
        if sorted(movable) != list(range(29)): raise ValueError("Incomplete or duplicate joint mapping")

    def __call__(self, root_pos, root_quat, joints):
        n=len(joints)
        base=np.tile(np.eye(4),(n,1,1)); base[:,:3,:3]=rotation(root_quat).as_matrix();base[:,:3,3]=root_pos
        result={"pelvis":base}; pending=list(self.joints)
        while pending:
            remaining=[]
            for parent,child,origin,axis,index in pending:
                if parent not in result: remaining.append((parent,child,origin,axis,index));continue
                local=np.tile(origin,(n,1,1))
                if index is not None:
                    local[:,:3,:3]=local[:,:3,:3] @ Rotation.from_rotvec(joints[:,index,None]*axis).as_matrix()
                result[child]=result[parent] @ local
            if len(remaining)==len(pending): raise ValueError("Cyclic or disconnected URDF")
            pending=remaining
        return result
