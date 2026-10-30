import unittest
import numpy as np
from hoi_pipeline.omnicontact_catalog import checked_surface


class CatalogSurfaceTests(unittest.TestCase):
    def setUp(self):
        import trimesh
        self.mesh = trimesh.creation.icosphere(subdivisions=1, radius=.108)
        self.mesh.vertices += [-.009, -.014, -.003]

    def test_offset_convex_surface_keeps_exact_vertices_and_faces(self):
        shape, violation = checked_surface(self.mesh.vertices, self.mesh.faces)
        np.testing.assert_array_equal(shape["vertices"], self.mesh.vertices)
        np.testing.assert_array_equal(shape["faces"], self.mesh.faces)
        self.assertLess(violation, 1e-6)
        self.assertGreater(np.linalg.norm(np.asarray(shape["vertices"]).mean(0)), .01)

    def test_open_inverted_and_dented_surfaces_rejected(self):
        for faces in (self.mesh.faces[:-1], self.mesh.faces[:, ::-1]):
            with self.assertRaises(ValueError):
                checked_surface(self.mesh.vertices, faces)
        vertices = self.mesh.vertices.copy()
        vertices[0] = vertices.mean(0)
        with self.assertRaisesRegex(ValueError, "not convex"):
            checked_surface(vertices, self.mesh.faces)

    def test_degenerate_faces_rejected(self):
        faces = self.mesh.faces.copy(); faces[0] = [0, 0, 0]
        with self.assertRaisesRegex(ValueError, "Degenerate"):
            checked_surface(self.mesh.vertices, faces)


if __name__ == "__main__":
    unittest.main()
