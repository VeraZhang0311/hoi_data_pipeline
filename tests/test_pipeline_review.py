"""CPU-only batch/output contracts; missing geometry produces real NaN gaps.

The fake shape replaces FCL construction only. Labeler._measure, group_gap,
classification, 90->50Hz projection and NPZ/JSON output run their real code.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from hoi_pipeline import SCHEMA
from hoi_pipeline import cli, pipeline
from hoi_pipeline.adapters.hoi_retarget import G1_JOINT_NAMES
from hoi_pipeline.common import BODY_NAMES, sha256


class _FK:
    def __init__(self, path, names):
        self.names = tuple(names)

    def __call__(self, root_pos, root_quat, joints):
        return {name: np.tile(np.eye(4), (len(root_pos), 1, 1)) for name in BODY_NAMES}


class _Shape:
    def __init__(self, spec):
        self.spec = spec
        self.world = np.eye(4)

    def place(self, world):
        self.world = world

    def min_z(self):
        return float(self.world[2, 3] - self.spec["size"][2] / 2)


def _clip(clip_id="review", flags=()):
    n = 5
    return {
        "clip_id": clip_id, "timestamps_s": np.arange(n, dtype=np.float64) / 90,
        "root_pos": np.zeros((n, 3)), "root_quat": np.tile([1., 0., 0., 0.], (n, 1)),
        "joint_pos": np.zeros((n, 29)), "joint_names": list(G1_JOINT_NAMES),
        "object_pos": np.tile([0., 0., 1.], (n, 1)),
        "object_quat": np.tile([1., 0., 0., 0.], (n, 1)),
        "object_asset": {"kind": "cuboid", "geometry_ready": True, "size_m": np.array([.1, .1, .1])},
        "source_contacts": {"values": np.full((n, 4), np.nan), "type": "unobserved_reference_annotation"},
        "scene_supports": [], "provenance": {"review_flags": list(flags), "sha256": "fixture"},
    }


def _strict_json(path):
    def invalid(value):
        raise ValueError("Non-standard JSON constant " + value)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=invalid)


class PipelineReviewTests(unittest.TestCase):
    def setUp(self):
        self.fk_patch = patch.object(pipeline, "RobotFK", _FK)
        self.shape_patch = patch.object(pipeline, "Shape", _Shape)
        self.fk_patch.start()
        self.shape_patch.start()
        self.addCleanup(self.fk_patch.stop)
        self.addCleanup(self.shape_patch.stop)
        self.labeler = pipeline.Labeler({"shapes": []}, "unused.urdf", G1_JOINT_NAMES)

    def test_real_missing_geometry_nan_survives_50hz_npz_and_strict_json(self):
        arrays, metadata = self.labeler.label(_clip())
        self.assertEqual(metadata["schema"], SCHEMA)
        self.assertEqual(arrays["target_labels"].shape, (3, len(BODY_NAMES) + 1))
        self.assertEqual(arrays["source_labels"].shape, (5, len(BODY_NAMES) + 1))
        self.assertEqual(arrays["interval_source_known_fraction"].shape, (2, len(BODY_NAMES) + 1))
        np.testing.assert_allclose(arrays["timestamps_s"], [0, .02, .04], atol=1e-14)
        self.assertTrue(np.isnan(arrays["target_distance_m"][:, :-1]).all())
        self.assertTrue((arrays["target_labels"][:, :-1] == -1).all())
        self.assertFalse(arrays["target_valid"][:, :-1].any())
        self.assertTrue((arrays["target_labels"][:, -1] == 0).all())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clip"
            pipeline.save_clip(output, arrays, metadata)
            saved = _strict_json(output / "metadata.json")
            success = _strict_json(output / "SUCCESS.json")
            self.assertEqual(success["array_sha256"], sha256(output / "labels.npz"))
            self.assertEqual(success["metadata_sha256"], sha256(output / "metadata.json"))
            key = saved["source_contacts"]["values"]["npz_key"]
            with np.load(output / "labels.npz", allow_pickle=False) as result:
                self.assertTrue(np.isnan(result[key]).all())
                self.assertTrue(np.isnan(result["target_distance_m"][:, :-1]).all())
            before = (output / "SUCCESS.json").read_bytes()
            with self.assertRaises(FileExistsError):
                pipeline.save_clip(output, arrays, metadata)
            self.assertEqual((output / "SUCCESS.json").read_bytes(), before)

    def test_numpy_scalar_nan_annotation_is_extracted_before_json(self):
        arrays, metadata = self.labeler.label(_clip())
        metadata["source_contacts"]["unknown_threshold"] = np.float32(np.nan)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clip"
            pipeline.save_clip(output, arrays, metadata)
            saved = _strict_json(output / "metadata.json")
            key = saved["source_contacts"]["unknown_threshold"]["npz_key"]
            with np.load(output / "labels.npz", allow_pickle=False) as result:
                self.assertTrue(np.isnan(result[key]))

    def test_adapter_review_flags_survive_labeler(self):
        flags = ("source_group_mapping_unverified", "missing_source_contact_info")
        _, metadata = self.labeler.label(_clip(flags=flags))
        self.assertTrue(set(flags).issubset(metadata["review_flags"]))

    def test_failed_metadata_write_never_creates_success_marker(self):
        arrays, metadata = self.labeler.label(_clip())
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "clip"
            with patch.object(pipeline, "atomic_json", side_effect=OSError("simulated disk failure")):
                with self.assertRaises(OSError):
                    pipeline.save_clip(output, arrays, metadata)
            self.assertFalse((output / "SUCCESS.json").exists())

    def test_batch_summary_separates_saved_outputs_and_failures(self):
        config = {"urdf_path": "unused.urdf", "on_m": .02, "off_m": .04, "target_hz": 50.}
        robot = {"shapes": []}
        inventory = {"omnicontact": [{"clip_id": "good"}, {"clip_id": "bad"}]}
        def load(config, source, record):
            if record["clip_id"] == "bad":
                raise ValueError("unresolved source geometry")
            return _clip("good")
        with tempfile.TemporaryDirectory() as directory:
            frozen = Path(directory) / "frozen"
            frozen.mkdir()
            (frozen / "common_inputs.json").write_text("{}")
            output = Path(directory) / "output"
            with patch.object(cli, "verify", return_value=(config, robot, {"recipe_id": "review_recipe"}, inventory)), \
                 patch.object(cli, "load", side_effect=load), contextlib.redirect_stdout(io.StringIO()):
                cli.run(frozen, output, "omnicontact", 0, 1)
                summary_path = output / "summaries/omnicontact-0000-of-0001.json"
                summary = _strict_json(summary_path)
                self.assertEqual((summary["selected"], summary["labeled"], summary["failed"]), (2, 1, 1))
                self.assertTrue(summary["completed"])
                self.assertTrue((Path(summary["entries"][0]["output"]) / "SUCCESS.json").exists())
                self.assertFalse((Path(summary["entries"][1]["output"]) / "SUCCESS.json").exists())
                before = summary_path.read_bytes()
                with self.assertRaises(FileExistsError):
                    cli.run(frozen, output, "omnicontact", 0, 1)
                self.assertEqual(summary_path.read_bytes(), before)

    def test_source_hash_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.bin"
            source.write_bytes(b"immutable source")
            evidence = {"path": str(source), "sha256": sha256(source), "size": source.stat().st_size}
            cli.check_files([evidence], {})
            source.write_bytes(b"changed source")
            with self.assertRaisesRegex(ValueError, "SHA mismatch"):
                cli.check_files([evidence], {})


if __name__ == "__main__":
    unittest.main()
