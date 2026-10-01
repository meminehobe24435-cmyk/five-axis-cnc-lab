# -*- coding: utf-8 -*-
"""精度检测测试：ISO 230-2 定位精度、ISO 230-4 循圆测试与已知误差反推。"""
import math
import unittest

from fiveaxis import metrology as met


class TestPositioningAccuracy(unittest.TestCase):

    def _dataset(self, bias_step=0.004, backlash=0.006, ripple=0.002, n_runs=5, n_pos=5):
        up, dn = [], []
        for i in range(n_pos):
            bias = bias_step * i
            up.append([bias + ripple * math.sin(j) for j in range(n_runs)])
            dn.append([bias - backlash + ripple * math.cos(j) for j in range(n_runs)])
        return up, dn

    def test_recovers_known_backlash(self):
        # 数据里带 sin/cos 波动，两个方向的均值不完全相等，
        # 所以反推值会比真值差一点点（0.010 → 0.01066），容差放到 1e-3
        up, dn = self._dataset(backlash=0.010)
        res = met.positioning_accuracy(up, dn)
        self.assertAlmostEqual(res['reversal_difference'], 0.010, delta=1e-3)

    def test_zero_noise_recovers_exactly(self):
        up, dn = self._dataset(backlash=0.010, ripple=0.0)
        res = met.positioning_accuracy(up, dn)
        self.assertAlmostEqual(res['reversal_difference'], 0.010, places=12)

    def test_accuracy_covers_systematic_span(self):
        """双向定位精度必须把系统性螺距误差的跨度（0.004*4 = 0.016）包进去。"""
        up, dn = self._dataset(bias_step=0.004, backlash=0.0, ripple=0.0)
        res = met.positioning_accuracy(up, dn)
        self.assertAlmostEqual(res['bidirectional_accuracy'], 0.016, places=9)

    def test_repeatability_grows_with_noise(self):
        quiet = met.positioning_accuracy(*self._dataset(ripple=0.001))
        noisy = met.positioning_accuracy(*self._dataset(ripple=0.005))
        self.assertGreater(noisy['repeatability_up'], quiet['repeatability_up'])

    def test_perfect_data_gives_zero_repeatability(self):
        up = [[0.0] * 5 for _ in range(3)]
        dn = [[0.0] * 5 for _ in range(3)]
        res = met.positioning_accuracy(up, dn)
        self.assertAlmostEqual(res['repeatability_up'], 0.0, places=12)
        self.assertAlmostEqual(res['reversal_difference'], 0.0, places=12)

    def test_bidirectional_not_smaller_than_unidirectional(self):
        res = met.positioning_accuracy(*self._dataset())
        self.assertGreaterEqual(res['repeatability_bidirectional'],
                                max(res['repeatability_up'], res['repeatability_dn']))

    def test_row_count(self):
        res = met.positioning_accuracy(*self._dataset(n_pos=7))
        self.assertEqual(res['positions'], 7)
        self.assertEqual(len(res['rows']), 7)

    def test_mismatched_lengths_rejected(self):
        with self.assertRaises(ValueError):
            met.positioning_accuracy([[0.0], [0.1]], [[0.0]])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            met.positioning_accuracy([], [])

    def test_missing_run_rejected(self):
        with self.assertRaises(ValueError):
            met.positioning_accuracy([[0.0], []], [[0.0], [0.0]])


class TestCircularTestRecovery(unittest.TestCase):
    """核心自检：注入已知误差，看能不能反推出来。"""

    CENTER = (0.0, 0.0)

    def _run(self, **kw):
        xs, ys, _ = met.simulate_circular_test(points=720, nominal_r=100.0, **kw)
        return met.circular_test(xs, ys, 100.0, center=self.CENTER)

    def test_backlash_only(self):
        r = self._run(backlash_um=20.0)
        self.assertAlmostEqual(r['backlash'] * 1000.0, 20.0, delta=0.5)
        self.assertAlmostEqual(r['ovality'] * 1000.0, 0.0, delta=0.5)
        self.assertAlmostEqual(r['squareness'] * 1000.0, 0.0, delta=0.5)

    def test_ovality_only(self):
        r = self._run(ovality_um=15.0)
        self.assertAlmostEqual(r['ovality'] * 1000.0, 15.0, delta=0.5)
        self.assertAlmostEqual(r['backlash'] * 1000.0, 0.0, delta=0.5)

    def test_squareness_only(self):
        r = self._run(squareness_um=12.0)
        self.assertAlmostEqual(r['squareness'] * 1000.0, 12.0, delta=1.0)
        self.assertAlmostEqual(r['ovality'] * 1000.0, 0.0, delta=0.5)

    def test_mixed_case(self):
        r = self._run(backlash_um=20.0, ovality_um=15.0, noise_um=3.0)
        self.assertAlmostEqual(r['backlash'] * 1000.0, 20.0, delta=1.0)
        self.assertAlmostEqual(r['ovality'] * 1000.0, 15.0, delta=1.5)

    def test_circular_deviation_is_total(self):
        """总圆度必须把三种误差都算进去（20 + 15 的叠加）。"""
        r = self._run(backlash_um=20.0, ovality_um=15.0)
        self.assertGreater(r['circular_deviation'] * 1000.0, 30.0)

    def test_backlash_and_ovality_are_separable(self):
        """两个指标不能互相串 —— 这是本项目最容易做错的一处。"""
        bak = self._run(backlash_um=20.0)
        ova = self._run(ovality_um=20.0)
        self.assertGreater(bak['backlash'], ova['backlash'] * 10)
        self.assertGreater(ova['ovality'], bak['ovality'] * 10)

    def test_centroid_absorbs_first_harmonic(self):
        """用形心当圆心会把 1θ（垂直度）分量吸收掉 —— 这是必须显式测出来的行为。"""
        xs, ys, _ = met.simulate_circular_test(squareness_um=12.0, nominal_r=100.0, points=720)
        nominal = met.circular_test(xs, ys, 100.0, center=(0.0, 0.0))
        centroid = met.circular_test(xs, ys, 100.0)
        self.assertAlmostEqual(nominal['squareness'] * 1000.0, 12.0, delta=1.0)
        self.assertLess(centroid['squareness'], nominal['squareness'] / 5.0)

    def test_radius_deviation_zero_when_centred(self):
        r = self._run(ovality_um=15.0)
        self.assertAlmostEqual(r['radius_deviation'] * 1000.0, 0.0, delta=0.5)

    def test_too_few_points_rejected(self):
        with self.assertRaises(ValueError):
            met.circular_test([1.0] * 10, [0.0] * 10, 1.0)

    def test_mismatched_points_rejected(self):
        with self.assertRaises(ValueError):
            met.circular_test([1.0] * 40, [0.0] * 39, 1.0)


class TestSimulator(unittest.TestCase):

    def test_deterministic(self):
        a = met.simulate_circular_test(noise_um=5.0, points=180)
        b = met.simulate_circular_test(noise_um=5.0, points=180)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[1], b[1])

    def test_point_count(self):
        xs, ys, truth = met.simulate_circular_test(points=360)
        self.assertEqual(len(xs), 360)
        self.assertEqual(len(ys), 360)
        self.assertEqual(truth['points'], 360)

    def test_clean_circle_is_perfect(self):
        xs, ys, _ = met.simulate_circular_test(nominal_r=50.0, points=360)
        r = met.circular_test(xs, ys, 50.0, center=(0.0, 0.0))
        self.assertAlmostEqual(r['circular_deviation'], 0.0, places=9)
        self.assertAlmostEqual(r['radius_deviation'], 0.0, places=9)
        self.assertAlmostEqual(r['backlash'], 0.0, places=9)

    def test_too_few_points_rejected(self):
        with self.assertRaises(ValueError):
            met.simulate_circular_test(points=20)


class TestHarmonics(unittest.TestCase):

    def test_pure_second_harmonic(self):
        angs = [2.0 * math.pi * i / 360.0 for i in range(360)]
        rs = [1.0 + 0.01 * math.cos(2.0 * a) for a in angs]
        h = met._harmonics(rs, angs, max_order=2)
        self.assertAlmostEqual(h[2][3], 0.02, places=6)
        self.assertAlmostEqual(h[1][3], 0.0, places=6)

    def test_pure_first_harmonic(self):
        angs = [2.0 * math.pi * i / 360.0 for i in range(360)]
        rs = [1.0 + 0.01 * math.cos(a) for a in angs]
        h = met._harmonics(rs, angs, max_order=2)
        self.assertAlmostEqual(h[1][3], 0.02, places=6)
        self.assertAlmostEqual(h[2][3], 0.0, places=6)


if __name__ == '__main__':
    unittest.main(verbosity=2)
