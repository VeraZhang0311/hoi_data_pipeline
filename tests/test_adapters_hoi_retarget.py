"""Adapter boundary tests: coordinate/order identity, provenance, and rejection."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from hoi_pipeline.adapters import hoi_retarget as h
from hoi_pipeline.adapters import ultra


class RetargetAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root / 'mesh.obj').write_text('v 0 0 0\nv 1 0 0\nv 0 1 0\nf 1 2 3\n')
        (self.root / 'box.urdf').write_text('<robot name="box"><link name="base"><visual><origin xyz="0.1 0.2 0.3"/><geometry><mesh filename="mesh.obj" scale="1 2 3"/></geometry></visual></link></robot>')
        self.path = self.root / 'motions.parquet'
        n = 3
        self.row = {'clip_id':'omomo/unitree_g1/sub1_largebox_000', 'dataset':'omomo', 'robot':'unitree_g1',
                    'subject':'sub1_largebox_000', 'sequence_id':'omomo/sub1_largebox_000', 'n_actors':1,
                    'n_frames':n, 'fps':30.0, 'duration_s':n/30, 'object':'largebox', 'object_model_path':'box.urdf',
                    'schema_version':2, 'qc_pass':True, 'qc_flags':[], 'dof_names':list(reversed(h.G1_JOINT_NAMES)),
                    'object_scale':0.83, 'object_aug_scale':1.0, 'scale_object_mesh':True, 'scale_object_trajectory':True,
                    'root_pos':[[i, 2, 3] for i in range(n)], 'root_rot':[[1, 0, 0, 0]]*n,
                    'dof_pos':[list(range(28, -1, -1))]*n, 'object_pos':[[4, 5, 6]]*n, 'object_rot':[[1, 0, 0, 0]]*n,
                    'contact_link_names':['left_palm_pad','right_palm_pad','left_ankle_roll_link','right_ankle_roll_link'],
                    'per_link_contact_flags':[[False]*4]*n,
                    'fixed_contact_points_per_frame_in_object_frame':[[[float('nan')]*3]*4]*n,
                    'robot_foot_ground_contact_flags':[[True, True]]*n, 'object_contact_sequence':[[0]]*n,
                    'object_ground_contact_sequence':[[1]]*n, 'object_near_floor_flags':[True]*n, 'palm_contact':float('nan')}
        self.write()

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        pq.write_table(pa.Table.from_pylist([self.row]), self.path)

    def load(self, **kwargs):
        return h.load_clip(self.path, row_index=0, source_root=self.root, revision='fixture', **kwargs)

    def test_no_resampling_named_reorder_and_asset_evidence(self):
        clip = self.load()
        np.testing.assert_array_equal(clip['timestamps_s'], np.arange(3)/30)
        np.testing.assert_array_equal(clip['joint_pos'], np.tile(np.arange(29), (3, 1)))
        np.testing.assert_array_equal(clip['root_pos'], self.row['root_pos'])
        np.testing.assert_array_equal(clip['root_quat'], self.row['root_rot'])
        self.assertEqual(clip['timestamps_s'].dtype, np.float64)
        self.assertEqual(clip['object_asset']['meshes'][0]['scale'], [1., 2., 3.])
        self.assertEqual(clip['object_asset']['meshes'][0]['origin_xyz'], [.1, .2, .3])
        self.assertEqual(clip['object_asset']['source_scale'], .83)
        self.assertEqual(clip['provenance']['source_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(clip['provenance']['split'], 'unassigned')
        self.assertFalse(clip['provenance']['training_ready'])
        self.assertEqual(np.asarray(clip['source_contacts']['values']['object_contact_sequence']).shape, (3, 1))

    def test_streaming_matches_single_row_and_skips_preview_column(self):
        from unittest import mock
        rows = []
        for i in range(5):
            row = dict(self.row)
            group = f"sub1_largebox_{i:03d}"
            row.update(clip_id="omomo/unitree_g1/"+group, subject=group, sequence_id="omomo/"+group, video=b"unused")
            rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), self.path, row_group_size=3)
        real = h._parquet(self.path)
        class Traced:
            metadata = real.metadata
            def iter_batches(self, **kwargs):
                self.call_count = getattr(self, "call_count", 0) + 1
                assert "video" not in kwargs["columns"]
                return real.iter_batches(**kwargs)
        traced = Traced()
        with mock.patch.object(h, "_parquet", return_value=traced):
            streamed = list(h.iter_clips(self.path, source_root=self.root, revision="fixture", batch_size=2))
        self.assertEqual(traced.call_count, 1)
        self.assertEqual(len(streamed), 5)
        for i, clip in enumerate(streamed):
            single = h.load_clip(self.path, row_index=i, source_root=self.root, revision="fixture")
            np.testing.assert_array_equal(clip["joint_pos"], single["joint_pos"])
            self.assertEqual(clip["provenance"]["source_sha256"], single["provenance"]["source_sha256"])
        filtered = list(h.iter_clips(self.path, source_root=self.root, revision="fixture", row_indices=[1,4]))
        self.assertEqual([c["provenance"]["source_row_index"] for c in filtered], [1,4])

    def test_streaming_row_error_does_not_poison_later_rows(self):
        import copy
        rows=[]
        for i in range(3):
            row=copy.deepcopy(self.row)
            group=f"sub1_largebox_{i:03d}"
            row.update(clip_id="omomo/unitree_g1/"+group, subject=group, sequence_id="omomo/"+group)
            if i==1: row["root_rot"][0]=[2,0,0,0]
            rows.append(row)
        pq.write_table(pa.Table.from_pylist(rows), self.path)
        result=list(h.iter_clips(self.path,source_root=self.root,revision="fixture",on_error="yield"))
        self.assertEqual(len(result),3)
        self.assertNotIn("_adapter_error",result[0])
        self.assertEqual(result[1]["row_index"],1)
        self.assertEqual(result[1]["_adapter_error"]["type"],"ValueError")
        self.assertNotIn("_adapter_error",result[2])
        self.assertTrue(result[2]["clip_id"].endswith("002"))
        with self.assertRaisesRegex(ValueError,"non-unit"):
            list(h.iter_clips(self.path,source_root=self.root,revision="fixture"))

    def test_protected_group_never_becomes_training(self):
        split = {'sub1_largebox_000': {'ultra_partition':'eval80'}}
        clip = self.load(split_index=split)
        self.assertTrue(clip['provenance']['protected_eval80'])
        self.assertEqual(clip['provenance']['split'], 'validation')
        train = self.load(split_index={'sub1_largebox_000': {'ultra_partition':'train'}})
        self.assertEqual(train['provenance']['split'], 'unassigned')

    def test_requires_unambiguous_selector(self):
        with self.assertRaises(ValueError):
            h.load_clip(self.path, source_root=self.root, revision='fixture')
        with self.assertRaises(ValueError):
            h.load_clip(self.path, self.row['clip_id'], row_index=0, source_root=self.root, revision='fixture')

    def test_bad_joint_identity_rejected(self):
        self.row['dof_names'][0] = self.row['dof_names'][1]
        self.write()
        with self.assertRaisesRegex(ValueError, 'joint names'):
            self.load()

    def test_nonunit_quaternion_rejected_without_repair(self):
        self.row['root_rot'][0] = [2, 0, 0, 0]
        self.write()
        with self.assertRaisesRegex(ValueError, 'non-unit'):
            self.load()

    def test_active_contact_requires_finite_point(self):
        self.row['per_link_contact_flags'] = [[True, False, False, False]]*3
        self.write()
        with self.assertRaisesRegex(ValueError, 'active contact'):
            self.load()

    def test_asset_escape_rejected(self):
        self.row['object_model_path'] = '../box.urdf'
        self.write()
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.load()

    def test_unknown_schema_and_augmentation_rejected(self):
        self.row['schema_version'] = 3
        self.write()
        with self.assertRaisesRegex(ValueError, 'schema-v2'):
            self.load()
        self.row['schema_version'] = 2
        self.row['object_aug_scale'] = .8
        self.write()
        with self.assertRaisesRegex(ValueError, 'augmentation'):
            self.load()


class UltraAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / 'sub1_largebox_000_100_100_100.pt'
        a = torch.zeros((3, 630), dtype=torch.float32)
        a[:, 6] = 1; a[:, 77] = 1
        a[:, 201:357] = torch.tensor([0., 0., 0., 1.]).repeat(39)
        a[:, 13:42] = torch.arange(28, -1, -1, dtype=torch.float32)
        a[:, :3] = torch.tensor([1., 2., 3.]); a[:, 71:74] = torch.tensor([4., 5., 6.])
        torch.save(a, self.path)
        case = {'motion':str(self.path), 'motion_sha256':hashlib.sha256(self.path.read_bytes()).hexdigest(),
                'source_group':'sub1_largebox_000', 'object':'largebox', 'object_asset':'largebox_100_100_100'}
        self.selection = self.root / 'selection.json'; self.selection.write_text(json.dumps({'cases':[case]}))
        self.binding = self.root / 'binding.json'; self.binding.write_text(json.dumps({
            'source_joint_names':list(reversed(h.G1_JOINT_NAMES)), 'joint_order_evidence':{'kind':'synthetic_fixture'}}))
        self.assets = self.root / 'assets.json'; self.assets.write_text(json.dumps({'assets':[{
            'object_asset':'largebox_100_100_100','exact_scale':[1,1,1], 'mesh_to_object':np.eye(4).tolist(),
            'collision_meshes':[{'vertices':[[0,0,0],[1,0,0],[0,1,0]],'faces':[[0,1,2]],'frame':'object_local'}]}]}))

    def tearDown(self):
        self.temp.cleanup()

    def load(self, **kwargs):
        return ultra.load_clip(self.selection, self.path.stem, bindings_path=self.binding, assets_path=self.assets,
                               profile='custom', **kwargs)

    def test_clock_and_xyzw_conversion_and_reorder(self):
        out = self.load(timebase='canonical_60hz')
        np.testing.assert_array_equal(out['root_quat'], [[1,0,0,0]]*3)
        np.testing.assert_array_equal(out['object_quat'], [[1,0,0,0]]*3)
        np.testing.assert_array_equal(out['joint_pos'], np.tile(np.arange(29), (3,1)))
        np.testing.assert_array_equal(out['timestamps_s'], np.arange(3)/60)
        self.assertEqual(out['source_contacts']['values']['body_contact_flags'].shape, (3,39))
        self.assertEqual(out['provenance']['split'], 'unassigned')
        self.assertIn('not_generation_verified', out['provenance']['timebase_status'])

    def test_timebase_and_canonical_identity_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'explicit timebase'):
            self.load()
        with self.assertRaisesRegex(ValueError, 'frozen ULTRA'):
            ultra.list_clips(self.selection)

    def test_hash_mismatch_rejected_before_tensor_load(self):
        with self.path.open('ab') as stream:
            stream.write(b'x')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            self.load(timebase='canonical_60hz')

    def test_cached_evidence_is_reused_and_invalidated_on_change(self):
        from unittest import mock
        with mock.patch.object(ultra, "_json_evidence", wraps=ultra._json_evidence) as reader:
            self.load(timebase="canonical_60hz")
            initial = reader.call_count
            self.load(timebase="canonical_60hz")
            self.assertEqual(reader.call_count, initial)
            self.binding.write_text(json.dumps({"source_joint_names":list(h.G1_JOINT_NAMES)}))
            with self.assertRaisesRegex(ValueError, "joint order"):
                self.load(timebase="canonical_60hz")
            self.assertEqual(reader.call_count, initial + 1)

    def test_returned_provenance_cannot_mutate_cached_source_evidence(self):
        first=self.load(timebase="canonical_60hz")
        first["provenance"]["source_joint_names"][0]="mutated"
        first["provenance"]["joint_order_evidence"]["kind"]="mutated"
        first["object_asset"]["recipe_evidence"]["sha256"]="mutated"
        second=self.load(timebase="canonical_60hz")
        self.assertEqual(second["provenance"]["source_joint_names"],list(reversed(h.G1_JOINT_NAMES)))
        self.assertEqual(second["provenance"]["joint_order_evidence"]["kind"],"synthetic_fixture")
        self.assertNotEqual(second["object_asset"]["recipe_evidence"]["sha256"],"mutated")

    def test_missing_joint_evidence_rejected(self):
        self.binding.write_text(json.dumps({'source_joint_names':list(h.G1_JOINT_NAMES)}))
        with self.assertRaisesRegex(ValueError, 'joint order'):
            self.load(timebase='canonical_60hz')


if __name__ == '__main__':
    unittest.main()
