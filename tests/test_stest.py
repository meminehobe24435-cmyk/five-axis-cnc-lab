# -*- coding: utf-8 -*-
"""S 形试件与工艺规范反推的测试。"""
import math
import unittest

from fiveaxis import interp
from fiveaxis import kinematics as kin
from fiveaxis import stest


class TestSCurve(unittest.TestCase):

    def test_endpoints(self):
        p0 = stest.s_curve(0.0)
        p1 = stest.s_curve(1.0)
        self.assertAlmostEqual(p0[0], 0.0, places=12)
        self.assertAlmostEqual(p1[0], stest.DEFAULT_LENGTH, places=12)
        self.assertAlmostEqual(p0[2], 0.0, places=12)
        self.assertAlmostEqual(p1[2], 0.0, places=12)

    def test_midpoint_crosses_zero(self):
        p = stest.s_curve(0.5)
        self.assertAlmostEqual(p[2], 0.0, places=9)

    def test_extremes(self):
        self.assertAlmostEqual(stest.s_curve(0.25)[2], stest.DEFAULT_AMPLITUDE, places=9)
        self.assertAlmostEqual(stest.s_curve(0.75)[2], -stest.DEFAULT_AMPLITUDE, places=9)

    def test_tangent_is_unit(self):
        for u in (0.0, 0.2, 0.5, 0.9, 1.0):
            t = stest.s_tangent(u)
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in t)), 1.0, places=12)

    def test_curvature_changes_sign(self):
        """S 形的定义就是曲率符号在中点翻转。

        ``z = A*sin(2*pi*u)``：``u = 0.25`` 处是**波峰**（下凹，kappa < 0），
        ``u = 0.75`` 处是**波谷**（上凹，kappa > 0）。
        """
        self.assertLess(stest.s_curvature(0.25), 0.0)
        self.assertGreater(stest.s_curvature(0.75), 0.0)

    def test_curvature_symmetric(self):
        self.assertAlmostEqual(stest.s_curvature(0.25), -stest.s_curvature(0.75), places=12)


class TestToolAxis(unittest.TestCase):

    def test_axis_is_unit(self):
        for u in (0.0, 0.15, 0.5, 0.83, 1.0):
            k = stest.tool_axis(u)
            self.assertAlmostEqual(math.sqrt(sum(v * v for v in k)), 1.0, places=12)

    def test_axis_stays_in_xz_plane(self):
        for u in (0.0, 0.3, 0.7, 1.0):
            self.assertAlmostEqual(stest.tool_axis(u)[1], 0.0, places=12)

    def test_tilt_never_reaches_singularity(self):
        """倾角区间不得跨过 0 —— 否则刀轴与 Z 轴平行，整段刀路解不出来。"""
        lo = stest.DEFAULT_TILT_MID - stest.DEFAULT_TILT_DEG
        hi = stest.DEFAULT_TILT_MID + stest.DEFAULT_TILT_DEG
        self.assertGreater(lo, 0.0, '倾角下界 %.1f° 已经进入奇异区' % lo)
        for i in range(101):
            u = i / 100.0
            k = stest.tool_axis(u)
            self.assertFalse(kin.is_singular(k), 'u=%.2f 处刀轴落进奇异点' % u)
            deg = math.degrees(kin.tilt_angle(k))
            self.assertGreaterEqual(deg, lo - 1e-9)
            self.assertLessEqual(deg, hi + 1e-9)

    def test_nominal_tilt_returns_mid_at_zero_crossing(self):
        self.assertAlmostEqual(stest.tilt_at(0.0), stest.DEFAULT_TILT_MID, places=9)
        self.assertAlmostEqual(stest.tilt_at(0.25),
                               stest.DEFAULT_TILT_MID + stest.DEFAULT_TILT_DEG, places=9)


class TestToolpath(unittest.TestCase):

    def test_point_count(self):
        path = stest.toolpath(40)
        self.assertEqual(len(path), 41)

    def test_u_spans_zero_to_one(self):
        path = stest.toolpath(10)
        self.assertAlmostEqual(path[0][0], 0.0, places=12)
        self.assertAlmostEqual(path[-1][0], 1.0, places=12)

    def test_tip_offset_applied(self):
        path = stest.toolpath(4, tip_offset=(10.0, 5.0, -2.0))
        p = path[0][1]
        base = stest.s_curve(0.0)
        self.assertAlmostEqual(p[0], base[0] + 10.0, places=12)
        self.assertAlmostEqual(p[1], base[1] + 5.0, places=12)
        self.assertAlmostEqual(p[2], base[2] - 2.0, places=12)

    def test_bad_n(self):
        with self.assertRaises(ValueError):
            stest.toolpath(0)


class TestEvaluate(unittest.TestCase):

    def setUp(self):
        self.path = stest.toolpath(60)
        self.res = stest.evaluate(self.path, 80.0, samples=40)

    def test_segment_count(self):
        self.assertEqual(self.res['segments_total'], 60)

    def test_no_singular_segments(self):
        self.assertEqual(self.res['singular_segments'], 0)

    def test_error_is_small_and_positive(self):
        self.assertGreater(self.res['max_nonlinearity_mm'], 0.0)
        self.assertLess(self.res['max_nonlinearity_mm'], 0.05)

    def test_max_error_exceeds_mean(self):
        self.assertGreater(self.res['max_nonlinearity_mm'], self.res['mean_nonlinearity_mm'])

    def test_max_tilt_matches_configured_range(self):
        self.assertAlmostEqual(self.res['max_tilt_deg'],
                               stest.DEFAULT_TILT_MID + stest.DEFAULT_TILT_DEG, delta=0.2)

    def test_rtcp_error_is_metres_scale_not_microns(self):
        """不开 RTCP 时偏差是几十毫米量级，和 μm 级的非线性误差不是一个概念。"""
        self.assertGreater(self.res['max_rtcp_error_mm'], 10.0)

    def test_worst_segment_at_max_tilt_rate(self):
        """最差段应当出现在 |dβ/du| 最大处（u = 0.5），而不是倾角最大处（u = 0.25）。"""
        segs = [s for s in self.res['segments'] if s['nonlinearity_mm'] is not None]
        worst = max(segs, key=lambda s: s['nonlinearity_mm'])
        self.assertAlmostEqual(worst['u'], 0.5, delta=0.06)

    def test_too_few_points(self):
        with self.assertRaises(ValueError):
            stest.evaluate([self.path[0]], 80.0)


class TestMaxStep(unittest.TestCase):

    def test_tighter_tolerance_means_smaller_step(self):
        s5, _ = stest.max_step_for_tolerance(0.005, 80.0)
        s10, _ = stest.max_step_for_tolerance(0.010, 80.0)
        s20, _ = stest.max_step_for_tolerance(0.020, 80.0)
        self.assertLess(s5, s10)
        self.assertLess(s10, s20)

    def test_longer_tool_means_smaller_step(self):
        s40, _ = stest.max_step_for_tolerance(0.010, 40.0)
        s120, _ = stest.max_step_for_tolerance(0.010, 120.0)
        self.assertLess(s120, s40)

    def test_larger_tilt_means_smaller_step(self):
        s20, _ = stest.max_step_for_tolerance(0.010, 80.0, tilt_deg=20.0)
        s50, _ = stest.max_step_for_tolerance(0.010, 80.0, tilt_deg=50.0)
        self.assertLess(s50, s20)

    def test_returned_step_actually_meets_tolerance(self):
        """反推出来的步距必须真的达标 —— 否则工艺卡上写的就是错的数。"""
        tol = 0.010
        step, err = stest.max_step_for_tolerance(tol, 80.0, tilt_deg=35.0)
        self.assertIsNotNone(step)
        self.assertLessEqual(err, tol * 1.02)

    def test_step_slightly_larger_should_fail(self):
        """在反推出的步距上放大 1.5 倍，就必须超差 —— 否则说明这个数是拍脑袋来的。

        ⚠️ 复核时**必须传同样的 ``tilt_deg``**：早先这里漏了参数、用了默认的 15°，
        算出来的误差比 35° 时小 5 倍多，于是"放大 1.5 倍仍然合格"的假象。
        """
        tol = 0.010
        tilt = 35.0
        step, _ = stest.max_step_for_tolerance(tol, 80.0, tilt_deg=tilt)
        self.assertIsNotNone(step)
        oversize = step * 1.5
        worst = 0.0
        for i in range(7):
            u = (i + 0.5) / 7.0
            du = oversize / stest.DEFAULT_LENGTH
            e, _ = interp.safe_nonlinearity(
                stest.s_curve(max(0.0, u - du / 2)), stest.tool_axis(max(0.0, u - du / 2), tilt),
                stest.s_curve(min(1.0, u + du / 2)), stest.tool_axis(min(1.0, u + du / 2), tilt),
                80.0, n=40)
            if e:
                worst = max(worst, e)
        self.assertGreater(worst, tol)

    def test_bad_tolerance(self):
        with self.assertRaises(ValueError):
            stest.max_step_for_tolerance(0.0, 80.0)
        with self.assertRaises(ValueError):
            stest.max_step_for_tolerance(0.01, 80.0, u_grid=2)


class TestCaseLibrary(unittest.TestCase):

    def test_shape(self):
        lib = stest.case_library(tool_lens=(40.0, 80.0), tilts=(20.0, 40.0),
                                 tolerances=(0.01, 0.02))
        self.assertEqual(len(lib['rows']), 4)
        self.assertEqual(lib['tolerances'], [0.01, 0.02])
        for row in lib['rows']:
            self.assertIn(0.01, row['steps'])
            self.assertIn(0.02, row['steps'])

    def test_step_monotonic_across_library(self):
        lib = stest.case_library(tool_lens=(40.0, 120.0), tilts=(20.0,), tolerances=(0.01,))
        steps = [r['steps'][0.01] for r in lib['rows']]
        self.assertLess(steps[1], steps[0])


if __name__ == '__main__':
    unittest.main(verbosity=2)
