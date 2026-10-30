import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hoi_pipeline.adapters import omnicontact_assets as assets
from hoi_pipeline.adapters.omnicontact import load_clip, G1_JOINT_NAMES
from hoi_pipeline.pipeline import object_recipe


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class AuthorAssetBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.relative = "npz_clips/box/case2_push/push/push_box_37-42-56_001_1234567_000.npz"
        self.path = self.root / self.relative
        self.path.parent.mkdir(parents=True)
        self.data = {
            "fps": np.array(90), "joint_pos": np.zeros((4, 29)),
            "joint_order_names": np.array(G1_JOINT_NAMES), "body_names": np.array(["pelvis"]),
            "body_pos_w": np.zeros((4, 1, 3)), "body_quat_w": np.tile([1., 0, 0, 0], (4, 1, 1)),
            "object_pos_w": np.tile([2., 3, .5], (4, 1)),
            "object_quat_w": np.tile([.5, .5, .5, .5], (4, 1)),
            "contact_info": np.zeros((4, 4, 1), dtype=np.uint8),
            "object_size_m_xyz": np.array([.37, .42, .56]), "scale": np.array([.7, .6, .8]),
            "object_viewer_center_offset_local": np.array([.1, .2, .3]),
            "object_geom_quat_wxyz": np.array([0., 1, 0, 0]),
        }
        np.savez(self.path, **self.data)
        self.usd = self.root / "object.usd"
        self.urdf = self.root / "object.urdf"
        self.code = self.root / "author_code.py"
        for p in (self.usd, self.urdf, self.code): p.write_text(p.name)
        self.mock_code = patch.object(assets, "AUTHOR_CODE", {"test/code.py": sha(self.code)})
        self.mock_code.start()
        self.addCleanup(self.mock_code.stop)
        local = np.eye(4); local[:3, 3] = [.01, .02, .03]
        self.geometry = {"basis": "author_USD_collision_in_rigid_body_frame", "root_path": "/Object/body",
                         "root_pose_replaced_by_recorded_pose": True,
                         "shapes": [{"kind": "cube", "size": [.37, .42, .56], "transform": local.tolist()}]}
        self.catalog = {"schema": assets.SCHEMA, "dataset_revision": assets.DATASET_REVISION,
                        "training_revision": assets.TRAINING_REVISION,
                        "author_code": {"test/code.py": {"path": str(self.code), "sha256": sha(self.code)}},
                        "assets": {"box_37-42-56": {"status": "verified", "geometry": self.geometry,
                                   "urdf_geometry": {"kind": "cube", "size": [.37, .42, .56]},
                                   "dependencies": [{"path": str(p), "sha256": sha(p)} for p in (self.usd, self.urdf)]}}}
        self.catalog_path = self.root / "catalog.json"
        self.record = {"path": str(self.path), "clip_id": self.relative, "source_group_verified": True,
                       "_input_files": [{"path": str(self.path), "sha256": sha(self.path)}]}
        self.record["author_geometry_binding"] = {"schema": assets.SCHEMA, "clip_id": self.relative,
            "asset_label": "box_37-42-56", "source_motion_sha256": sha(self.path),
            "catalog_path": str(self.catalog_path)}
        self.save_catalog()

    def save_catalog(self):
        self.catalog_path.write_text(json.dumps(self.catalog))
        self.record["author_geometry_binding"]["catalog_sha256"] = sha(self.catalog_path)

    def load(self):
        return load_clip(self.record, source_root=self.root)

    def test_author_shape_and_pose_unchanged_despite_legacy_transforms(self):
        clip = self.load()
        np.testing.assert_array_equal(clip["object_pos"], self.data["object_pos_w"])
        np.testing.assert_array_equal(clip["object_quat"], self.data["object_quat_w"])
        self.assertEqual(object_recipe(clip["object_asset"]), self.geometry)
        self.assertFalse(clip["object_asset"]["source_scale_applied_again"])
        self.assertFalse(clip["object_asset"]["source_center_applied_again"])

    def test_explicit_geometry_preserves_sphere_radius_and_local_transform(self):
        geometry = dict(self.geometry, shapes=[{"kind": "sphere", "radius": .11, "transform": np.eye(4).tolist()}])
        asset = {"kind": "author_collision", "geometry_ready": True,
                 "binding_status": "verified_author_training_asset", "author_geometry": geometry}
        self.assertEqual(object_recipe(asset)["shapes"][0]["radius"], .11)
        self.assertEqual(object_recipe(self.load()["object_asset"])["shapes"][0]["transform"], self.geometry["shapes"][0]["transform"])

    def test_dependencies_enter_freeze_and_motion_identity_is_mandatory(self):
        deps = assets.binding_files(self.record)
        self.assertEqual({Path(x["path"]).name for x in deps}, {"catalog.json", "author_code.py", "object.usd", "object.urdf"})
        self.record["_input_files"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "motion SHA"):
            assets.binding_files(self.record)

    def test_changed_motion_catalog_and_cached_dependencies_rejected(self):
        self.load()  # populate verification cache
        with self.path.open("ab") as stream: stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "motion SHA"):
            self.load()
        np.savez(self.path, **self.data)
        self.catalog_path.write_text(self.catalog_path.read_text() + " ")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.load()
        self.save_catalog()
        self.usd.write_text("changed USD dependency")
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.load()

    def test_wrong_clip_label_revision_and_missing_code_rejected(self):
        binding = self.record["author_geometry_binding"]
        for key, value in (("clip_id", "different.npz"), ("asset_label", "soccer_22")):
            with self.subTest(key=key):
                old = binding[key]; binding[key] = value
                with self.assertRaises(ValueError): self.load()
                binding[key] = old
        self.catalog["training_revision"] = "new_unreviewed_release"; self.save_catalog()
        with self.assertRaisesRegex(ValueError, "revision/schema"):
            self.load()
        self.catalog["training_revision"] = assets.TRAINING_REVISION
        self.catalog["author_code"] = {}; self.save_catalog()
        with self.assertRaisesRegex(ValueError, "code evidence"):
            self.load()

    def test_metadata_conflict_is_preserved_without_fitting_or_repair(self):
        self.data["object_grounding_box_size_m"] = np.array([.37, .50, .56])
        np.savez(self.path, **self.data)
        self.record["author_geometry_binding"]["source_motion_sha256"] = sha(self.path)
        clip = self.load()
        self.assertIn("author_asset_differs_from_source_object_grounding_box_size_m", clip["provenance"]["review_flags"])
        self.assertFalse(clip["provenance"]["repaired"])
        np.testing.assert_array_equal(clip["object_pos"], self.data["object_pos_w"])

    def test_lookup_returns_independent_geometry_not_mutable_cache_state(self):
        first = self.load(); first["object_asset"]["author_geometry"]["shapes"][0]["size"][0] = 999
        self.assertEqual(self.load()["object_asset"]["author_geometry"]["shapes"][0]["size"][0], .37)

    def test_same_size_replacement_with_preserved_mtime_invalidates_cache(self):
        self.load()
        stat = self.usd.stat()
        replacement = self.root / "replacement"
        replacement.write_bytes(b"x" * stat.st_size)
        os.utime(replacement, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        os.replace(replacement, self.usd)
        with self.assertRaisesRegex(ValueError, "SHA mismatch"):
            self.load()

    def test_full_filename_mapping_refuses_size_substring_guesses(self):
        for filename, label in (("kick_ball_22_123_000.npz", "soccer_22"),
                                ("relocate_soccer_20_123_000.npz", "soccer_20"),
                                ("slide_box_24-24-24_123_000.npz", "box_24-24-24")):
            self.assertEqual(assets.asset_label(filename), label)
        for filename in ("prefix_push_box_37-42-56_000.npz", "push_box_37-42-56_bad.npz"):
            with self.assertRaises(ValueError): assets.asset_label(filename)


if __name__ == "__main__": unittest.main()
