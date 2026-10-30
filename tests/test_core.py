import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
from hoi_pipeline.timebase import resample,interval_evidence,events
from hoi_pipeline.geometry import Shape,gap,classify
from hoi_pipeline.pipeline import save_clip

def clip(hz=90,n=91):
    t=np.arange(n)/hz;q=np.tile([1.,0,0,0],(n,1))
    return dict(timestamps_s=t,root_pos=np.c_[t,t*0,t*0],root_quat=q,joint_pos=np.tile(t[:,None],(1,29)),object_pos=np.c_[t*2,t*0,t*0],object_quat=q)

class TimeTests(unittest.TestCase):
    def test_real_duration_and_state_interval_contract(self):
        out=resample(clip(90,100))
        self.assertEqual(len(out["timestamps_s"]),56)
        np.testing.assert_allclose(np.diff(out["timestamps_s"]),.02,rtol=0,atol=1e-15)
        np.testing.assert_allclose(out["root_pos"][:,0],out["timestamps_s"])
        self.assertAlmostEqual(out["timebase"]["tail_not_on_grid_s"],0.)
        tail=resample(clip(30,2))
        self.assertEqual(len(tail["timestamps_s"]),2)
        self.assertAlmostEqual(tail["timebase"]["tail_not_on_grid_s"],1/30-.02)
    def test_quaternion_shortest_arc_and_sign(self):
        c=clip(1,2);c["root_quat"]=Rotation.from_euler("z",np.array([[170.],[-170.]]),degrees=True).as_quat()[:,[3,0,1,2]]
        c["root_quat"][1]*=-1
        out=resample(c)
        self.assertAlmostEqual(abs(out["root_quat"][25,3]),1.)
    def test_short_event_preserved_in_interval(self):
        t=np.arange(10)/90;labels=np.zeros((10,1),np.int8);labels[1]=1
        target=np.arange(5)/50
        positive,known=interval_evidence(t,labels,target)
        self.assertGreater(positive[0,0],0)
        self.assertAlmostEqual(float(positive[:,0].sum()*.02),1/90,places=7)
        np.testing.assert_allclose(known,1)
        self.assertEqual(len(events(t,labels)),1)
    def test_unknown_is_not_negative(self):
        labels=np.array([[-1],[1],[-1]],np.int8)
        positive,known=interval_evidence(np.array([0,.02,.04]),labels,np.array([0,.02,.04]))
        np.testing.assert_allclose(positive,.5);np.testing.assert_allclose(known,.5)
    def test_threshold_unknown_intersection_and_lower_bound(self):
        result=classify(np.array([.01,.01,.03,.05,np.nan]),np.array([1,1,1,0,0],bool),np.array([0,1,0,0,0],bool))
        np.testing.assert_array_equal(result,[1,-1,-1,0,-1])
    def test_source_nan_saved_in_npz(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)/"clip"
            save_clip(out,{"a":np.arange(2)},{"source_contacts":{"palm_contact":float("nan"),"points":{0:np.array([1.,2.,3.])}}})
            self.assertTrue((out/"SUCCESS.json").exists())
            with np.load(out/"labels.npz",allow_pickle=False) as a:
                self.assertTrue(np.isnan(a["metadata/source_contacts/palm_contact"]))
                np.testing.assert_array_equal(a["metadata/source_contacts/points/0"],[1,2,3])
            with self.assertRaises(FileExistsError): save_clip(out,{}, {})

@unittest.skipUnless(importlib.util.find_spec("fcl"),"FCL CPU dependency not installed")
class GeometryTests(unittest.TestCase):
    def test_sphere_surface_distance_and_overlap(self):
        a=Shape({"kind":"sphere","radius":.1});b=Shape({"kind":"cube","size":[.2,.2,.2]})
        mat=np.eye(4);mat[0,3]=.215;a.place(mat)
        d,exact,inside=gap(a,b,.04)
        self.assertAlmostEqual(d,.015,places=6);self.assertTrue(exact);self.assertFalse(inside)
        mat[0,3]=.195;a.place(mat)
        self.assertTrue(gap(a,b,.04)[2])
    def test_broadphase_is_lower_bound(self):
        a=Shape({"kind":"sphere","radius":.1});b=Shape({"kind":"sphere","radius":.1})
        mat=np.eye(4);mat[:3,3]=[2,2,0];a.place(mat)
        d,exact,inside=gap(a,b,.04)
        self.assertFalse(exact);self.assertFalse(inside);self.assertLessEqual(d,np.sqrt(8)-.2)
    def test_complete_containment_rejected(self):
        import trimesh
        mesh=trimesh.creation.box([1,1,1])
        big=Shape({"kind":"mesh","vertices":mesh.vertices.tolist(),"faces":mesh.faces.tolist()})
        small=Shape({"kind":"sphere","radius":.1})
        self.assertTrue(gap(big,small,.04)[2])
    def test_ground_capsule_rotation(self):
        s=Shape({"kind":"capsule","radius":.1,"height":1.,"axis":"X"})
        self.assertAlmostEqual(s.min_z(),-.1)

if __name__=="__main__": unittest.main()
