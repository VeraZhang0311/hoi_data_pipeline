import hashlib
import os
from pathlib import Path
import pickle
import tempfile
import unittest
import zlib

import numpy as np

from hoi_pipeline.adapters.grail import G1_JOINT_NAMES, GrailFormatError, _load_numeric, list_clips, load_clip


def numeric_joblib_fixture(array):
    """Minimal author-format numeric wrapper + aligned raw Fortran payload."""
    def word(text):
        encoded = text.encode("utf-8")
        return pickle.SHORT_BINUNICODE + bytes([len(encoded)]) + encoded
    payload = pickle.PROTO + b"\x04"
    payload += word("joblib.numpy_pickle") + word("NumpyArrayWrapper") + pickle.STACK_GLOBAL
    payload += pickle.EMPTY_TUPLE + pickle.NEWOBJ + pickle.EMPTY_DICT + pickle.MARK
    payload += word("subclass") + word("numpy") + word("ndarray") + pickle.STACK_GLOBAL
    payload += word("shape") + pickle.BININT1 + bytes([array.shape[0]]) + pickle.BININT1 + bytes([array.shape[1]]) + pickle.TUPLE2
    payload += word("order") + word("F")
    payload += word("dtype") + word("numpy") + word("dtype") + pickle.STACK_GLOBAL + word("<f4") + pickle.TUPLE1 + pickle.REDUCE
    payload += word("allow_mmap") + pickle.NEWTRUE
    payload += word("numpy_array_alignment_bytes") + pickle.BININT1 + bytes([16])
    payload += pickle.SETITEMS + pickle.BUILD
    padding = (16 - (len(payload) + 1) % 16) % 16
    return payload + bytes([padding]) + b"\xff" * padding + array.astype("<f4").tobytes(order="F") + pickle.STOP


def fixtures(root, *, order="xyzw", protocol=4, compressed=False):
    for name in ("robot", "objects", "meta", "object_usd"):
        (root / name).mkdir(parents=True, exist_ok=True)
    n = 4
    aa = np.zeros((n, 30, 3), dtype=np.float32)
    aa[:, 0, 2] = np.array([0.0, 0.2, 0.4, 0.6])
    q = np.zeros((n, 4), dtype=np.float32)
    q[:, 0] = np.cos(aa[:, 0, 2] / 2)
    q[:, 3] = np.sin(aa[:, 0, 2] / 2)
    robot = {"generation_original_key": {
        "dof": np.arange(n * 29, dtype=np.float32).reshape(n, 29) / 100,
        "root_trans_offset": np.array([[i, 0, 0.8] for i in range(n)], dtype=np.float32),
        "root_rot": q[:, [1, 2, 3, 0]] if order == "xyzw" else q,
        "pose_aa": aa, "smpl_joints": np.zeros((n, 24, 3), dtype=np.float32), "fps": 25.0,
    }}
    objects = {"generation_original_key": {
        "root_pos": np.ones((n, 1, 3), dtype=np.float32),
        "root_quat": np.tile([1., 0, 0, 0], (n, 1, 1)), "fps": 25.,
        "scale": np.full((3, 1), 0.8, dtype=np.float32),
        "contact_points_left_hand": {2: np.array([[0.1, 0.2, 0.3]], dtype=np.float32)},
        "contact_points_right_hand": {2: np.empty((0, 3), dtype=np.float32)},
    }}
    meta = {"object_name": "object", "scene_scale": 1.0, "smpl_height": 1.66}
    for folder, data in (("robot", robot), ("objects", objects), ("meta", meta)):
        raw = pickle.dumps(data, protocol=protocol)
        (root / folder / "sitting__chair__000.pkl").write_bytes(zlib.compress(raw) if compressed else raw)
    (root / "object_usd" / "sitting__chair__000.usda").write_text(
        '#usda 1.0\n( metersPerUnit = 1\n upAxis = "Z" )\ndef Xform "Chair" {}\n', encoding="utf-8")
    return robot, objects, meta


class GrailAdapterTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def load(self, **kwargs):
        return load_clip(self.root / "robot" / "sitting__chair__000.pkl", **kwargs)

    def rewrite(self, folder, data):
        (self.root / folder / "sitting__chair__000.pkl").write_bytes(pickle.dumps(data, protocol=4))

    def test_canonical_pose_time_and_exact_source_hashes(self):
        robot, objects, _ = fixtures(self.root)
        clip = self.load(source_revision="a" * 40, source_group="shared_original", split="train")
        np.testing.assert_array_equal(clip["timestamps_s"], np.arange(4, dtype=np.float64) / 25)
        self.assertEqual(clip["timestamps_s"].dtype, np.float64)
        np.testing.assert_array_equal(clip["joint_pos"], robot["generation_original_key"]["dof"])
        self.assertEqual(clip["joint_names"], list(G1_JOINT_NAMES))
        np.testing.assert_array_equal(clip["object_quat"], objects["generation_original_key"]["root_quat"][:, 0])
        self.assertEqual(clip["provenance"]["robot_quaternion"]["source_order"], "xyzw")
        self.assertEqual(clip["provenance"]["source_group"], "shared_original")
        self.assertFalse(clip["provenance"]["resampled"])
        self.assertFalse(clip["provenance"]["smpl_joints_used"])
        raw = (self.root / "robot" / "sitting__chair__000.pkl").read_bytes()
        self.assertEqual(clip["provenance"]["files"]["robot"]["sha256"], hashlib.sha256(raw).hexdigest())

    def test_both_author_quaternion_formats_are_verified(self):
        for order in ("xyzw", "wxyz"):
            with self.subTest(order=order):
                fixtures(self.root, order=order)
                clip = self.load()
                self.assertEqual(clip["provenance"]["robot_quaternion"]["source_order"], order)
                np.testing.assert_allclose(clip["root_quat"][0], [1, 0, 0, 0], atol=1e-7)
        with self.assertRaisesRegex(GrailFormatError, "disagrees"):
            self.load(root_quat_order="xyzw")

    def test_quaternion_mismatch_is_not_silently_guessed(self):
        robot, _, _ = fixtures(self.root)
        robot["generation_original_key"]["pose_aa"][:, 0, 0] = 1.2
        self.rewrite("robot", robot)
        with self.assertRaisesRegex(GrailFormatError, "not uniquely verified"):
            self.load()

    def test_released_usd_uses_source_object_scale_once(self):
        fixtures(self.root)
        clip = self.load()
        np.testing.assert_allclose(clip["object_asset"]["scale"], [0.8] * 3)
        np.testing.assert_allclose(clip["object_asset"]["source_object_scale"], [0.8] * 3)
        self.assertTrue(clip["object_asset"]["source_object_scale_applied"])
        self.assertEqual(clip["object_asset"]["scale_source"], "objects.scale")
        self.assertEqual(clip["object_asset"]["source_meta_scene_scale"], 1.0)
        self.assertFalse(clip["object_asset"]["meta_scene_scale_applied"])
        self.assertFalse(clip["object_asset"]["geometry_verified"])
        self.assertTrue(clip["object_asset"]["requires_usd_composition"])

    def test_scalar_and_nonuniform_object_scale_are_preserved(self):
        for value, expected in ((0.7, [0.7] * 3), ([0.7, 0.8, 0.9], [0.7, 0.8, 0.9])):
            with self.subTest(value=value):
                _, objects, _ = fixtures(self.root)
                objects["generation_original_key"]["scale"] = value
                self.rewrite("objects", objects)
                np.testing.assert_array_equal(self.load()["object_asset"]["scale"], expected)

    def test_invalid_scale_and_unaudited_scene_scale_fail_closed(self):
        for value in (0, -1, [1, 2], [1, np.nan, 1], [1, np.inf, 1], ["x"]):
            with self.subTest(value=value):
                _, objects, _ = fixtures(self.root)
                objects["generation_original_key"]["scale"] = value
                self.rewrite("objects", objects)
                with self.assertRaisesRegex(GrailFormatError, "Invalid source object scale"):
                    self.load()
        _, _, meta = fixtures(self.root)
        meta["scene_scale"] = 1.25
        self.rewrite("meta", meta)
        with self.assertRaisesRegex(GrailFormatError, "explicit released-USD composition audit"):
            self.load()

    def test_source_hand_points_remain_annotations(self):
        fixtures(self.root)
        source = self.load()["source_contacts"]
        self.assertFalse(source["is_force"])
        self.assertFalse(source["is_retargeted_contact_ground_truth"])
        self.assertEqual(set(source["points_by_source_frame"]["contact_points_left_hand"]), {2})
        self.assertEqual(source["points_by_source_frame"]["contact_points_right_hand"][2].shape, (0, 3))

    def test_pairing_timebase_and_shape_fail_closed(self):
        for mutation in ("key", "fps", "length", "nonfinite"):
            with self.subTest(mutation=mutation):
                _, objects, _ = fixtures(self.root)
                if mutation == "key": objects = {"wrong_key": objects["generation_original_key"]}
                elif mutation == "fps": objects["generation_original_key"]["fps"] = 30
                elif mutation == "length": objects["generation_original_key"]["root_pos"] = np.zeros((3, 1, 3))
                else: objects["generation_original_key"]["root_pos"][0, 0, 0] = np.nan
                self.rewrite("objects", objects)
                with self.assertRaises(GrailFormatError): self.load()

    def test_protocol5_and_zlib_pickle(self):
        for protocol in (4, 5):
            for compressed in (False, True):
                with self.subTest(protocol=protocol, compressed=compressed):
                    fixtures(self.root, protocol=protocol, compressed=compressed)
                    self.assertEqual(self.load()["joint_pos"].shape, (4, 29))

    def test_actual_joblib_numeric_wrapper_alignment_and_fortran_order(self):
        array = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
        target = self.root / "joblib.pkl"
        raw = numeric_joblib_fixture(array)
        for compressed in (False, True):
            target.write_bytes(zlib.compress(raw) if compressed else raw)
            value, receipt = _load_numeric(target)
            np.testing.assert_array_equal(value, array)
            self.assertEqual(receipt["sha256"], hashlib.sha256(target.read_bytes()).hexdigest())
        target.write_bytes(raw[:-4])
        with self.assertRaises(GrailFormatError):
            _load_numeric(target)

    def test_ambiguous_quaternion_requires_explicit_verified_order(self):
        robot, _, _ = fixtures(self.root)
        robot["generation_original_key"]["root_rot"][:] = 0.5
        robot["generation_original_key"]["pose_aa"][:, 0] = (2 * np.pi / 3) / np.sqrt(3)
        self.rewrite("robot", robot)
        with self.assertRaisesRegex(GrailFormatError, "not uniquely verified"):
            self.load()
        self.assertEqual(self.load(root_quat_order="wxyz")["provenance"]["robot_quaternion"]["source_order"], "wxyz")

    def test_inventory_keeps_missing_pairs_visible(self):
        fixtures(self.root)
        records = list_clips(self.root, source_revision="revision")
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0]["files_complete"])
        self.assertEqual(load_clip(records[0])["provenance"]["source_revision"], "revision")
        (self.root / "objects" / "sitting__chair__001.pkl").write_bytes(b"orphan")
        (self.root / "robot" / "sitting__chair__002.pkl").write_bytes(b"pending")
        records = list_clips(self.root)
        self.assertFalse(records[1]["files_complete"])
        self.assertIn("objects", records[1]["missing"])

    def test_untrusted_pickle_cannot_execute_globals(self):
        class Attack:
            def __reduce__(self):
                return (os.system, ("DO_NOT_EXECUTE",))
        target = self.root / "attack.pkl"
        target.write_bytes(pickle.dumps(Attack()))
        with self.assertRaisesRegex(GrailFormatError, "Forbidden pickle global"):
            _load_numeric(target)

    def test_object_dtype_and_trailing_bytes_are_rejected(self):
        target = self.root / "bad.pkl"
        target.write_bytes(pickle.dumps(np.array(["unsafe"], dtype=object), protocol=4))
        with self.assertRaises(GrailFormatError): _load_numeric(target)
        target.write_bytes(pickle.dumps({"ok": 1}) + b"trailing")
        with self.assertRaisesRegex(GrailFormatError, "Trailing bytes"):
            _load_numeric(target)


if __name__ == "__main__":
    unittest.main()
