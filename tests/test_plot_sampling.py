"""Lightweight postprocessing tests; no solver, GPU, or external data needed."""

import unittest

import matplotlib.pyplot as plt
import numpy as np

from planar_comparisons import (
    contiguous_segments,
    draw_profile,
    equal_arc_markers,
    sample_center_cut,
    replace_error_column,
)


class TestCutSampling(unittest.TestCase):
    def setUp(self):
        self.axis = (np.arange(8) + 0.5) / 8
        self.X, self.Y = np.meshgrid(self.axis, self.axis, indexing="ij")
        self.mask = np.ones_like(self.X, dtype=bool)
        self.values = 2 * self.X + 3 * self.Y

    def sample(self, varying_axis=0, fixed_coordinate=0.5, mask=None, fields=None):
        return sample_center_cut(
            self.X,
            self.Y,
            self.mask if mask is None else mask,
            {"affine": self.values} if fields is None else fields,
            varying_axis=varying_axis,
            fixed_coordinate=fixed_coordinate,
        )

    def test_horizontal_affine_cut(self):
        cut = self.sample()
        np.testing.assert_allclose(cut["fields"]["affine"], 2 * self.axis + 1.5)
        np.testing.assert_allclose(cut["coordinate_cm"], 10 * self.axis)
        self.assertEqual(cut["bracketing_indices"], [3, 4])
        self.assertEqual(cut["upper_weight"], 0.5)

    def test_vertical_affine_cut(self):
        cut = self.sample(varying_axis=1)
        np.testing.assert_allclose(cut["fields"]["affine"], 1 + 3 * self.axis)

    def test_exact_grid_row_needs_no_neighbour(self):
        mask = self.mask.copy()
        mask[:, 2] = False
        cut = self.sample(fixed_coordinate=self.axis[3], mask=mask)
        self.assertTrue(cut["valid"].all())
        np.testing.assert_array_equal(cut["fields"]["affine"], self.values[:, 3])

    def test_hole_not_bridged(self):
        mask = self.mask.copy()
        mask[3:5, 3:5] = False
        cut = self.sample(mask=mask)
        self.assertTrue(np.isnan(cut["fields"]["affine"][3:5]).all())
        self.assertEqual(len(contiguous_segments(cut["valid"])), 2)

    def test_both_interpolation_rows_must_be_active(self):
        mask = self.mask.copy()
        mask[1, 3] = False
        cut = self.sample(mask=mask)
        self.assertFalse(cut["valid"][1])
        self.assertTrue(np.isnan(cut["fields"]["affine"][1]))

    def test_no_extrapolation(self):
        for position in (-0.1, 1.1):
            with self.assertRaises(ValueError):
                self.sample(fixed_coordinate=position)

    def test_nonfinite_active_sample_rejected(self):
        field = self.values.copy()
        field[1, 3] = np.nan
        with self.assertRaises(ValueError):
            self.sample(fields={"bad": field})

    def test_arc_markers_keep_all_segment_endpoints(self):
        valid = np.ones(200, dtype=bool)
        valid[40:80] = False
        valid[120:180] = False
        x = np.linspace(0, 1, len(valid))
        samples = equal_arc_markers(x, x * x, valid, points_per_data_unit=[100, 100])
        self.assertTrue(valid[samples["source_left"]].all())
        self.assertTrue(valid[samples["source_right"]].all())
        self.assertTrue(np.all(samples["source_right"] - samples["source_left"] == 1))
        for segment in contiguous_segments(valid):
            self.assertIn(x[segment[0]], samples["x"])
            self.assertIn(x[segment[-1]], samples["x"])

    def test_empty_marker_selection(self):
        samples = equal_arc_markers(
            np.arange(10), np.arange(10), np.zeros(10, dtype=bool), points_per_data_unit=[1, 1]
        )
        self.assertEqual(samples["x"].size, 0)

    def test_marker_arc_spacing_is_uniform_without_index_rounding(self):
        x = np.linspace(0, 1, 36)
        y = np.exp(-4 * x)
        samples = equal_arc_markers(
            x, y, np.ones_like(x, dtype=bool), points_per_data_unit=[130, 100]
        )
        steps = np.diff(samples["arc_pt"])
        np.testing.assert_allclose(steps, steps[0], rtol=1e-13)
        self.assertTrue(np.any((samples["upper_weight"] > 0) & (samples["upper_weight"] < 1)))
        np.testing.assert_allclose(samples["lbm"], np.interp(samples["x"], x, y), atol=1e-14)

    def test_straight_line_markers_have_constant_screen_distance(self):
        x = np.linspace(0, 1, 36)
        samples = equal_arc_markers(
            x, 3 * x, np.ones_like(x, dtype=bool), points_per_data_unit=[120, 40]
        )
        xy = np.column_stack((samples["x"] * 120, samples["lbm"] * 40))
        distance = np.linalg.norm(np.diff(xy, axis=0), axis=1)
        np.testing.assert_allclose(distance, distance[0], rtol=1e-13)

    def test_one_native_point_segment(self):
        samples = equal_arc_markers(
            np.array([0.0, 1.0, 2.0]),
            np.array([0.0, 5.0, 0.0]),
            np.array([False, True, False]),
            points_per_data_unit=[120, 40],
        )
        np.testing.assert_array_equal(samples["x"], [1.0])
        np.testing.assert_array_equal(samples["lbm"], [5.0])

    def test_exact_line_and_hollow_lbm_markers(self):
        fig, ax = plt.subplots()
        valid = np.array([True, True, False, True, True])
        values = np.arange(5.0)
        draw_profile(ax, values, values, values + 0.1, valid=valid, reference_label="Exact")
        line, markers = ax.lines
        self.assertEqual(line.get_label(), "Exact")
        self.assertEqual(line.get_linestyle(), "-")
        self.assertTrue(np.isnan(line.get_ydata()[2]))
        self.assertEqual(markers.get_linestyle(), "None")
        self.assertEqual(markers.get_markerfacecolor(), "none")
        self.assertFalse(np.isin(2, markers.get_xdata()))
        plt.close(fig)

    def test_replacement_preserves_first_two_columns(self):
        fig, axs = plt.subplots(1, 3, figsize=(7.25, 2.4))
        for ax in axs:
            mesh = ax.pcolormesh(np.arange(16.0).reshape(4, 4))
            fig.colorbar(mesh, ax=ax)
        fig.canvas.draw()
        retained = [
            (
                ax.get_position().bounds,
                ax.collections[0],
                ax.collections[0].get_array().copy(),
                ax.collections[0].get_clim(),
            )
            for ax in axs[:2]
        ]
        x = np.linspace(0, 1, 30)
        profiles = [
            {
                "x": x,
                "reference": x * x,
                "lbm": x * x + 0.01,
                "xlabel": "x",
                "ylabel": "u",
                "note": "test cut",
            }
        ]
        replace_error_column(fig, profiles, match_column_spacing=True)
        for ax, (position, artist, data, limits) in zip(axs[:2], retained):
            np.testing.assert_allclose(ax.get_position().bounds, position, atol=1e-12)
            self.assertIs(ax.collections[0], artist)
            np.testing.assert_array_equal(ax.collections[0].get_array(), data)
            self.assertEqual(ax.collections[0].get_clim(), limits)
        self.assertTrue(any(tick.get_visible() for tick in axs[2].get_yticklabels()))
        first, second, third = [ax.get_position() for ax in axs]
        self.assertAlmostEqual(second.x0 - first.x1, third.x0 - second.x1, places=12)
        self.assertEqual(axs[2].yaxis.get_label_position(), "right")
        steps = np.diff(profiles[0]["marker_samples"]["arc_pt"])
        np.testing.assert_allclose(steps, steps[0], rtol=1e-13)
        plt.close(fig)


if __name__ == "__main__":
    unittest.main()
