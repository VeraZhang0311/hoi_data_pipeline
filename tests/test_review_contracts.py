"""Independent analytic contracts; review reproducers below are not hidden fixes."""
import tempfile
from pathlib import Path
import unittest
import xml.etree.ElementTree as ET
import numpy as np
from scipy.spatial.transform import Rotation
from hoi_pipeline.common import BODY_NAMES
from hoi_pipeline.timebase import resample, interval_evidence, events
from hoi_pipeline.kinematics import RobotFK
from hoi_pipeline.adapters.hoi_retarget import G1_JOINT_NAMES


def clip(t):
    n = len(t)
    return {'timestamps_s':np.asarray(t), 'root_pos':np.zeros((n,3)), 'root_quat':np.tile([1.,0,0,0],(n,1)),
            'joint_pos':np.zeros((n,29)), 'object_pos':np.zeros((n,3)), 'object_quat':np.tile([1.,0,0,0],(n,1))}


def make_chain(path, first_axis='0 0 1', duplicate_child=False):
    root = ET.Element('robot', name='review_fixture')
    for name in BODY_NAMES:
        ET.SubElement(root,'link',name=name)
    for i,child in enumerate(BODY_NAMES[1:]):
        joint = ET.SubElement(root,'joint',name=G1_JOINT_NAMES[i] if i < 29 else 'fixed_'+str(i), type='revolute' if i < 29 else 'fixed')
        ET.SubElement(joint,'parent',link=BODY_NAMES[i]); ET.SubElement(joint,'child',link=child)
        ET.SubElement(joint,'origin',xyz='0.5 0.25 -0.1' if i==0 else '0 0 0',rpy='0.2 -0.1 0.3' if i==0 else '0 0 0')
        ET.SubElement(joint,'axis',xyz=first_axis if i==0 else '0 0 1')
    if duplicate_child:
        joint=ET.SubElement(root,'joint',name='invalid_duplicate_child',type='fixed')
        ET.SubElement(joint,'parent',link='pelvis'); ET.SubElement(joint,'child',link=BODY_NAMES[1])
    ET.ElementTree(root).write(path)


class TimebaseContracts(unittest.TestCase):
    def test_90hz_to_50hz_has_t_plus_one_without_stretching(self):
        out=resample(clip(np.arange(5)/90))
        np.testing.assert_allclose(out['timestamps_s'],[0,.02,.04],rtol=0,atol=1e-14)
        self.assertEqual(out['timebase']['state_count'],3)
        self.assertEqual(out['timebase']['interval_count'],2)
        self.assertAlmostEqual(out['timebase']['tail_not_on_grid_s'],4/90-.04)

    def test_short_contact_survives_interval_projection_and_unknowns_stay_unknown(self):
        t=np.arange(5)/200
        labels=np.array([[-1],[-1],[1],[-1],[-1]],dtype=np.int8)
        positive,known=interval_evidence(t,labels,np.array([0,.02]))
        np.testing.assert_allclose(positive,[[.25]],atol=1e-7)
        np.testing.assert_allclose(known,[[.25]],atol=1e-7)
        event=events(t,labels)[0]
        self.assertAlmostEqual(event['start_s'],.0075)
        self.assertAlmostEqual(event['end_s'],.0125)

    def test_quaternion_interpolation_uses_wxyz_and_short_arc(self):
        source=clip([0.,.04])
        q=Rotation.from_euler('z',[[170],[-170]],degrees=True).as_quat()
        source['root_quat']=q[:,[3,0,1,2]]
        out=resample(source)
        middle=Rotation.from_quat(out['root_quat'][1,[1,2,3,0]]).as_matrix()
        np.testing.assert_allclose(middle,Rotation.from_euler('z',180,degrees=True).as_matrix(),atol=1e-12)


class FKContracts(unittest.TestCase):
    def test_origin_then_local_joint_then_root_composition(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'robot.urdf'; make_chain(p)
            fk=RobotFK(p,G1_JOINT_NAMES)
            joints=np.zeros((1,29)); joints[0,0]=.7
            root_R=Rotation.from_euler('z',np.pi/2)
            root_q=root_R.as_quat()[[3,0,1,2]]
            poses=fk(np.array([[1.,2.,3.]]),root_q[None],joints)
            expected_R=root_R.as_matrix() @ Rotation.from_euler('xyz',[.2,-.1,.3]).as_matrix() @ Rotation.from_euler('z',.7).as_matrix()
            expected_p=np.array([1.,2.,3.])+root_R.apply([.5,.25,-.1])
            np.testing.assert_allclose(poses[BODY_NAMES[1]][0,:3,:3],expected_R,atol=1e-12)
            np.testing.assert_allclose(poses[BODY_NAMES[1]][0,:3,3],expected_p,atol=1e-12)


if __name__=='__main__':
    unittest.main()
