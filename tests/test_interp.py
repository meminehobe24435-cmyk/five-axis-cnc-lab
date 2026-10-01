# -*- coding: utf-8 -*-
"""插补与误差测试：非线性误差、弦高误差、点线距离。"""
import math
import unittest

from fiveaxis import interp
from fiveaxis import kinematics as kin


class TestGeometryHelpers(unittest.TestCase):

    def test_point_on_segment_has_zero_distance(self):
        self.assertAlmostEqual(interp.point_segment_distance((1.0, 0.0, 0.0), (0, 0, 0), (2, 0, 0)), 0.0, places=12)
        self.assertAlmostEqual(interp.point_segment_distance((0.5, 0.0, 0.0), (0, 0, 0), (2, 0, 0)), 0.0, places=12)

    def test_perpendicular_distance(self):
        self.assertAlmostEqual(interp.point_segment_distance((1.0, 3.0, 0.0), (0, 0, 0), (2, 0, 0)), 3.0, places=12)

    def test_beyond_endpoint_clamps(self):
        """线段外应当量到端点距离，而不是量到无限直线的距离。"""
        self.assertAlmostEqual(interp.point_segment_distance((5.0, 0.0, 0.0), (0, 0, 0), (2, 0, 0)), 3.0, places=12)

    def test_degenerate_segment(self):
        self.assertAlmostEqual(interp.point_segment_distance((3.0, 4.0, 0.0), (0, 0, 0), (0, 0, 0)), 5.0, places=12)


class TestNonlinearity(unittest.TestCase):

    def test_pure_translation_has_no_error(self):
        """刀轴不变、只平移 —— 关节空间线性插补就是直线，非线性误差必须为 0。

        ⚠️ 刀轴不能取 ``(0,0,1)`` —— 那是**奇异点**，逆解本身就不唯一，
        会先抛 ``SingularPointError``，测不到这里想测的东西。
        """
        k = kin.forward_a(math.radians(30.0), 0.0)
        err, _ = interp.nonlinearity((0, 0, 0), k, (10.0, 0, 0), k, 80.0, n=50)
        self.assertAlmostEqual(err, 0.0, places=12)

    def test_error_depends_on_orientation_change_not_step(self):
        """**纯转动**造成的非线性误差与平移步距无关 —— 只取决于两端的姿态差。

        这一点很容易想当然搞反：以为"步距缩小误差就小"。实际上如果两端姿态不变，
        只是把这段平移拉长，刀尖轨迹的弯曲程度是一样的（弯的是转动那部分）。
        真实加工里"缩小步距能减小误差"，是因为**步距小了、两端的姿态差也小了** ——
        那是路径几何决定的，见 :meth:`test_error_grows_with_tilt_change`。
        """
        L = 80.0
        k0 = kin.forward_a(math.radians(20.0), 0.0)
        k1 = kin.forward_a(math.radians(35.0), 0.0)
        errs = [interp.nonlinearity((0, 0, 0), k0, (s, 0, 0), k1, L, n=80)[0]
                for s in (1.0, 2.0, 4.0, 8.0)]
        for e in errs[1:]:
            self.assertAlmostEqual(e, errs[0], delta=errs[0] * 1e-6)

    def test_error_grows_with_tilt_change(self):
        """步距固定时，姿态差越大，非线性误差越大（单调）。"""
        L = 80.0
        k0 = kin.forward_a(math.radians(20.0), 0.0)
        errs = []
        for tilt1 in (22.0, 26.0, 32.0, 40.0):
            k1 = kin.forward_a(math.radians(tilt1), 0.0)
            err, _ = interp.nonlinearity((0, 0, 0), k0, (2.0, 0, 0), k1, L, n=80)
            errs.append(err)
        for i in range(1, len(errs)):
            self.assertGreater(errs[i], errs[i - 1],
                               '姿态差增大而非线性误差没有增大：%s' % errs)
        # 姿态差 2° → 6° 是 3 倍，误差随姿态差平方增长，应当约 9 倍
        self.assertAlmostEqual(errs[1] / errs[0], 9.0, delta=0.6)

    def test_error_proportional_to_tool_length(self):
        """刀越长，同样的转角造成的偏差越大（近似线性）。"""
        k0 = kin.forward_a(math.radians(20.0), 0.0)
        k1 = kin.forward_a(math.radians(40.0), 0.0)
        e1, _ = interp.nonlinearity((0, 0, 0), k0, (2.0, 0, 0), k1, 40.0, n=80)
        e2, _ = interp.nonlinearity((0, 0, 0), k0, (2.0, 0, 0), k1, 80.0, n=80)
        self.assertAlmostEqual(e2 / e1, 2.0, delta=0.05)

    def test_axis_returns_to_start(self):
        """转角相同但方向相反，误差应当相同（对称性）。"""
        k0 = kin.forward_a(math.radians(20.0), 0.0)
        kp = kin.forward_a(math.radians(35.0), 0.0)
        km = kin.forward_a(math.radians(5.0), 0.0)
        ep, _ = interp.nonlinearity((0, 0, 0), k0, (2.0, 0, 0), kp, 80.0, n=80)
        em, _ = interp.nonlinearity((0, 0, 0), k0, (2.0, 0, 0), km, 80.0, n=80)
        self.assertAlmostEqual(ep, em, delta=max(ep, em) * 0.02)

    def test_singular_endpoint_raises(self):
        """终点刀轴平行于 Z 轴时必须报错，而不是悄悄给一个 C 值。"""
        with self.assertRaises(kin.SingularPointError):
            interp.nonlinearity((0, 0, 0), kin.forward_a(math.radians(30.0), 0.0),
                                (2.0, 0, 0), (0.0, 0.0, 1.0), 80.0, n=20)

    def test_safe_variant_returns_none(self):
        err, samples = interp.safe_nonlinearity((0, 0, 0), kin.forward_a(math.radians(30.0), 0.0),
                                                (2.0, 0, 0), (0.0, 0.0, 1.0), 80.0, n=20)
        self.assertIsNone(err)
        self.assertEqual(samples, [])

    def test_bad_n_rejected(self):
        with self.assertRaises(ValueError):
            interp.nonlinearity((0, 0, 0), (0, 0, 1), (1, 0, 0), (0, 0, 1), 80.0, n=0)

    def test_samples_span_full_interval(self):
        k = kin.forward_a(math.radians(30.0), 0.0)
        _, samples = interp.nonlinearity((0, 0, 0), k, (1, 0, 0), k, 80.0, n=10)
        self.assertEqual(len(samples), 11)
        self.assertAlmostEqual(samples[0][0], 0.0, places=12)
        self.assertAlmostEqual(samples[-1][0], 1.0, places=12)


class TestJointDeltas(unittest.TestCase):

    def test_deltas_are_differences(self):
        q0 = (0.0, 0.0, 0.0, 0.0, 0.0)
        q1 = (1.0, 2.0, 3.0, math.radians(10.0), math.radians(-20.0))
        d = interp.joint_deltas(q0, q1)
        self.assertAlmostEqual(d['X'], 1.0)
        self.assertAlmostEqual(d['Y'], 2.0)
        self.assertAlmostEqual(d['Z'], 3.0)
        self.assertAlmostEqual(d['A_deg'], 10.0)
        self.assertAlmostEqual(d['C_deg'], -20.0)


class TestChordError(unittest.TestCase):

    def test_segments_shrink_with_tolerance(self):
        coarse = interp.arc_segments(50.0, 0.02)
        fine = interp.arc_segments(50.0, 0.002)
        self.assertGreater(fine, coarse)

    def test_segments_grow_with_radius(self):
        small = interp.arc_segments(10.0, 0.005)
        large = interp.arc_segments(100.0, 0.005)
        self.assertGreater(large, small)

    def test_achieved_error_within_tolerance(self):
        for r, tol in ((10.0, 0.005), (50.0, 0.002), (200.0, 0.01)):
            n = interp.arc_segments(r, tol)
            self.assertLessEqual(interp.arc_max_chord_error(r, n), tol * 1.001)

    def test_one_fewer_segment_exceeds(self):
        """少分一段就应当超差 —— 否则说明分段数是拍脑袋来的。"""
        r, tol = 50.0, 0.002
        n = interp.arc_segments(r, tol)
        self.assertGreater(interp.arc_max_chord_error(r, n - 1), tol)

    def test_bad_input(self):
        with self.assertRaises(ValueError):
            interp.arc_segments(0.0, 0.01)
        with self.assertRaises(ValueError):
            interp.arc_segments(10.0, 0.0)
        with self.assertRaises(ValueError):
            interp.arc_max_chord_error(10.0, 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
