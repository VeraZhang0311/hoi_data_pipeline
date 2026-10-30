import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from hoi_pipeline.adapters.omnicontact import (
    G1_JOINT_NAMES, ISAAC_TO_G1, OmniContactFormatError, list_clips, load_clip,
)


class OmniContactAdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.folder = self.root / "npz_clips/box/case1_carry"
        self.folder.mkdir(parents=True)
        self.path = self.folder / "carry_box_30-30-30_001_1234567890_000.npz"
        self.values = {
            "fps": np.array([90]),
            "joint_pos": np.tile(np.arange(29, dtype=np.float32), (4, 1)),
            "joint_order_names": np.array(list(reversed(G1_JOINT_NAMES))),
            "body_names": np.array(["torso_link", "pelvis"]),
            "body_pos_w": np.array([[[4, 5, 6], [7, 8, 9]]] * 4, dtype=float),
            "body_quat_w": np.tile([1., 0., 0., 0.], (4, 2, 1)),
            "object_pos_w": np.array([[2., 3., 4.]] * 4),
            "object_quat_w": np.tile([1., 0., 0., 0.], (4, 1)),
            "contact_info": np.tile(np.array([0, 1, 1, 0], dtype=np.uint8)[None, :, None], (4, 1, 1)),
            "table1_pos_w": np.tile([.5, .6, .7], (4, 1)),
            "table2_pos_w": np.tile([0., 0., -10.], (4, 1)),
        }
        self.write()
        index = {"frame_indexing": "0-based, inclusive", "clips": [{
            "clip": self.path.name, "source": "source.npz", "start_frame": 10, "end_frame": 13}]}
        (self.folder / "clip_info.json").write_text(json.dumps(index))
        (self.root / "metadata").mkdir()
        (self.root / "metadata/raw_mocap_manifest.csv").write_text("npz,capture_id\nsource.npz,capture_a\n")
        (self.root / "metadata/splits.csv").write_text("capture_id,split\ncapture_a,train\n")

    def tearDown(self):
        self.temp.cleanup()

    def write(self):
        np.savez(self.path, **self.values)

    def test_named_layout_world_pose_time_and_provenance(self):
        clip = load_clip(self.path, source_root=self.root)
        np.testing.assert_array_equal(clip["joint_pos"][0], np.arange(29)[::-1])
        np.testing.assert_array_equal(clip["root_pos"], np.tile([7, 8, 9], (4, 1)))
        np.testing.assert_allclose(clip["timestamps_s"], np.arange(4) / 90)
        self.assertEqual(clip["timestamps_s"].dtype, np.float64)
        self.assertEqual(clip["provenance"]["root_body_index"], 1)
        self.assertEqual(clip["provenance"]["source_group_id"], "capture_a")
        self.assertEqual(clip["provenance"]["split"], "train")
        self.assertFalse(clip["provenance"]["cut"])
        self.assertEqual(len(clip["provenance"]["sha256"]), 64)

    def test_contacts_and_partner_identity_are_not_forces(self):
        clip = load_clip(list_clips(self.root)[0])
        self.assertFalse(clip["source_contacts"]["is_force_or_impulse"])
        self.assertEqual(clip["source_contacts"]["values"].shape, (4, 4))
        self.assertEqual([s["partner_id"] for s in clip["scene_supports"]], ["table1", "table2"])
        self.assertTrue(clip["scene_supports"][1]["parked_below_scene"])
        self.assertFalse(clip["object_asset"]["geometry_ready"])
        np.testing.assert_array_equal(clip["scene_supports"][0]["size_m"], [.3, .25, .01])

    def test_discontinuity_flagged_without_repair_or_cut(self):
        self.values["object_pos_w"][2:, 0] += 2
        self.write()
        clip = load_clip(self.path, source_root=self.root)
        self.assertEqual(clip["provenance"]["spatial_discontinuities"]["object"]["clip_frames"], [2])
        self.assertEqual(len(clip["timestamps_s"]), 4)
        np.testing.assert_array_equal(clip["object_pos"], self.values["object_pos_w"])

    def test_invalid_quaternion_is_not_normalized(self):
        self.values["object_quat_w"][1] *= 2
        self.write()
        with self.assertRaisesRegex(OmniContactFormatError, "non-unit"):
            load_clip(self.path, source_root=self.root)

    def test_missing_unnamed_layout_evidence_rejected(self):
        del self.values["joint_order_names"]
        self.write()
        with self.assertRaisesRegex(OmniContactFormatError, "pinned official viewer"):
            load_clip(self.path, source_root=self.root)

    def test_misaligned_lengths_rejected_without_truncation(self):
        self.values["object_pos_w"] = self.values["object_pos_w"][:3]
        self.write()
        with self.assertRaisesRegex(OmniContactFormatError, "object_pos_w"):
            load_clip(self.path, source_root=self.root)

    def test_missing_contact_is_unknown_not_zero(self):
        del self.values["contact_info"]
        self.write()
        clip = load_clip(self.path, source_root=self.root)
        self.assertIsNone(clip["source_contacts"])
        self.assertIn("missing_source_contact_info", clip["provenance"]["review_flags"])

    def test_author_carry_binding_uses_final_dimensions_without_contact_fit(self):
        mesh = self.root / "assets/objects/Box_H_1.obj"
        mesh.parent.mkdir(parents=True)
        mesh.write_text("v -15 -15 -15\nv 15 -15 -15\nv -15 15 -15\nv 15 15 -15\n"
                        "v -15 -15 15\nv 15 -15 15\nv -15 15 15\nv 15 15 15\n"
                        "f 1 2 4 3\nf 5 6 8 7\nf 1 2 6 5\nf 3 4 8 7\nf 1 3 7 5\nf 2 4 8 6\n")
        self.values.update(
            object_contact_name=np.array("Box_H_1"),
            carry_box_selected_size_cm=np.array([30, 30, 30]),
            carry_box_source_size_cm=np.array([30, 30, 30]),
            carry_box_offset_local_m=np.zeros(3),
            carry_modification_method=np.array("box_recenter_resize_random_float_hand_ik_all_frames_v2_wristwide_2cm"),
            carry_modification_scope=np.array("all_frames"),
            object_geom_quat_wxyz=np.array([1., 0., 0., 0.]),
            object_xml_geom_quat_mode=np.array("post"),
            object_quat_input_wxyz=self.values["object_quat_w"].copy(),
            scale=np.array([0.7]),
        )
        self.write()
        first = load_clip(self.path, source_root=self.root)
        self.assertTrue(first["object_asset"]["geometry_ready"])
        np.testing.assert_array_equal(first["object_asset"]["size_m"], [.3, .3, .3])
        self.values["contact_info"][:] = 0
        self.write()
        second = load_clip(self.path, source_root=self.root)
        np.testing.assert_array_equal(first["object_asset"]["size_m"], second["object_asset"]["size_m"])
        np.testing.assert_array_equal(second["object_pos"], self.values["object_pos_w"])
        self.values["carry_box_selected_size_cm"][0] = 40
        self.write()
        third = load_clip(self.path, source_root=self.root)
        self.assertFalse(third["object_asset"]["geometry_ready"])
        self.assertIn("disagree", third["object_asset"]["binding_review_reason"])

    @unittest.skipUnless(os.environ.get("OMNICONTACT_SOURCE_ROOT"),
                         "Set OMNICONTACT_SOURCE_ROOT for immutable real-data integration check")
    def test_real_release_sample(self):
        root = Path(os.environ["OMNICONTACT_SOURCE_ROOT"])
        records = list_clips(root)
        self.assertGreater(len(records), 0)
        clip = load_clip(records[0])
        with np.load(records[0]["path"], allow_pickle=False) as original:
            np.testing.assert_array_equal(clip["joint_pos"], original["joint_pos"][:, ISAAC_TO_G1])
            np.testing.assert_array_equal(clip["root_pos"], original["body_pos_w"][:, 0])
            np.testing.assert_array_equal(clip["root_quat"], original["body_quat_w"][:, 0])
            np.testing.assert_array_equal(clip["object_pos"], original["object_pos_w"])
            np.testing.assert_array_equal(clip["object_quat"], original["object_quat_w"])
            self.assertEqual(len(clip["timestamps_s"]), len(original["joint_pos"]))
        self.assertEqual(clip["joint_names"], list(G1_JOINT_NAMES))
        self.assertTrue(clip["object_asset"]["geometry_ready"])
        self.assertEqual(clip["object_asset"]["kind"], "cuboid")
        self.assertFalse(clip["object_asset"]["source_scale_applied_again"])
        self.assertFalse(clip["object_asset"]["binding_evidence"]["contact_labels_used_to_fit_geometry"])


if __name__ == "__main__":
    unittest.main()
