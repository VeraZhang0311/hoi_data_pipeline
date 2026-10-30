"""Exact analytic/triangle surface queries near contact, conservative broad phase."""
from __future__ import annotations
import numpy as np
from scipy.spatial.transform import Rotation
from .common import points_transform

class Shape:
    def __init__(self,spec):
        import fcl
        self.spec=spec;self.local=np.asarray(spec.get("transform",np.eye(4)),dtype=float)
        kind=spec["kind"];self.mesh=None
        if kind=="mesh":
            import trimesh
            vertices=np.asarray(spec["vertices"],dtype=np.float64);faces=np.asarray(spec["faces"],dtype=np.int32)
            if vertices.ndim!=2 or vertices.shape[1]!=3 or faces.ndim!=2 or faces.shape[1]!=3 or not np.isfinite(vertices).all():
                raise ValueError("Invalid triangle mesh")
            if len(vertices)<3 or len(faces)==0 or np.any(faces<0) or np.any(faces>=len(vertices)):
                raise ValueError("Empty mesh or out-of-bounds triangle indices")
            model=fcl.BVHModel();model.beginModel(len(vertices),len(faces));model.addSubModel(vertices,faces);model.endModel()
            self.geom=model; self.vertices=vertices
            self.mesh=trimesh.Trimesh(vertices,faces,process=False)
        elif kind=="cube":
            size=np.asarray(spec["size"]);self.geom=fcl.Box(*size);self.vertices=_corners(-size/2,size/2)
        elif kind in ("sphere","capsule","cylinder"):
            r=spec["radius"]
            if kind=="sphere": self.geom=fcl.Sphere(r);ext=np.repeat(r,3)
            else:
                h=spec["height"];self.geom=(fcl.Capsule if kind=="capsule" else fcl.Cylinder)(r,h)
                ext=np.array([r,r,h/2+(r if kind=="capsule" else 0)])
                axis=spec.get("axis","Z")
                orient={"Z":np.eye(3),"X":Rotation.from_euler("y",np.pi/2).as_matrix(),"Y":Rotation.from_euler("x",-np.pi/2).as_matrix()}[axis]
                self.local[:3,:3]=self.local[:3,:3]@orient
            self.vertices=_corners(-ext,ext)
        else: raise ValueError("Unknown shape "+kind)
        self.obj=fcl.CollisionObject(self.geom)
        if self.mesh is not None:
            # Actual surface vertices, one per connected face component. An AABB
            # centre need not belong to a concave/disconnected solid at all.
            import trimesh
            components=trimesh.graph.connected_components(self.mesh.face_adjacency,nodes=np.arange(len(self.mesh.faces)),min_len=1)
            self.probes=np.asarray([self.mesh.vertices[self.mesh.faces[c[0],0]] for c in components])
        else: self.probes=np.zeros((1,3))
        self.place(np.eye(4))

    def place(self,body):
        import fcl
        self.world=body@self.local
        self.obj.setTransform(fcl.Transform(self.world[:3,:3],self.world[:3,3]))
        self.world_vertices=points_transform(self.vertices,self.world)
        self.lo=self.world_vertices.min(0);self.hi=self.world_vertices.max(0)

    def contains_world(self,point):
        p=(np.asarray(point)-self.world[:3,3])@self.world[:3,:3]
        kind=self.spec["kind"]
        if kind=="mesh":
            return bool(self.mesh.is_watertight and self.mesh.contains(p[None])[0])
        if kind=="cube": return bool(np.all(np.abs(p)<np.asarray(self.spec["size"])/2-1e-7))
        r=self.spec["radius"]
        if kind=="sphere": return bool(np.linalg.norm(p)<r-1e-7)
        h=self.spec["height"]/2
        if kind=="cylinder": return bool(np.linalg.norm(p[:2])<r-1e-7 and abs(p[2])<h-1e-7)
        return bool(np.linalg.norm(p-np.array([0,0,np.clip(p[2],-h,h)]))<r-1e-7)

    def min_z(self):
        kind=self.spec["kind"]
        if kind=="mesh": return float(self.world_vertices[:,2].min())
        if kind=="sphere": extent=self.spec["radius"]
        elif kind=="cube": extent=np.abs(self.world[2,:3])@(np.asarray(self.spec["size"])/2)
        else:
            z=abs(self.world[2,2]);r=self.spec["radius"];h=self.spec["height"]/2
            extent=h*z+(r if kind=="capsule" else r*np.sqrt(max(0,1-z*z)))
        return float(self.world[2,3]-extent)

def _corners(lo,hi):
    import itertools
    return np.array(list(itertools.product(*zip(lo,hi))),dtype=float)

def gap(a,b,query_threshold):
    import fcl
    lower=float(np.linalg.norm(np.maximum(0,np.maximum(a.lo-b.hi,b.lo-a.hi))))
    if lower>query_threshold: return lower,False,False
    collision=fcl.collide(a.obj,b.obj,fcl.CollisionRequest(),fcl.CollisionResult())>0
    if not collision:
        # Triangle BVHs are surfaces: test solid containment separately where defined.
        for source,target in ((a,b),(b,a)):
            if collision: break
            for point in points_transform(source.probes,source.world):
                if np.all(point>target.lo) and np.all(point<target.hi) and target.contains_world(point):
                    collision=True;break
    if collision: return 0.,True,True
    distance=float(fcl.distance(a.obj,b.obj,fcl.DistanceRequest(),fcl.DistanceResult()))
    if not np.isfinite(distance) or distance<0: raise ValueError("FCL returned invalid separation")
    return distance,True,False

def group_gap(a,b,off):
    values=[gap(x,y,off) for x in a for y in b]
    if not values: return np.nan,False,False
    if any(v[2] for v in values): return 0.,True,True
    index=int(np.argmin([v[0] for v in values]));result=values[index]
    # An exact minimum below off beats every broad-phase lower bound (> off).
    return result

def classify(distance,exact,intersect,on=.02,off=.04):
    if not (0<=on<off): raise ValueError("Expected 0 <= on < off")
    d=np.asarray(distance); labels=np.full(d.shape,-1,np.int8)
    labels[np.isfinite(d)&~intersect&(d>=off)]=0
    labels[np.isfinite(d)&~intersect&exact&(d<=on)]=1
    return labels
