# -*- coding: utf-8 -*-
"""运动学测试：正逆解、RTCP 理论值、奇异点、行程限制。"""
import math
import unittest

from fiveaxis import kinematics as kin


class TestForwardInverse(unittest.TestCase):

    def test_zero_attitude_axis_is_z(self):
        self.assertEqual(kin.forward_a(0.0, 0.0), (0.0, 0.0, 1.0))

    def test_axis_is_unit_length(self):
        for a_deg in (5.0, 30.0, 75.0, 110.0):
            for c_deg in (0.0, 45.0, 130.0, 250.0):
                k = kin.forward_a(math.radians(a_deg), math.radians(c_deg))
                n = math.sqrt(sum(v * v for v in k))
                self.assertAlmostEqual(n, 1.0, places=12)

    def test_round_trip_matches_formula(self):
        """逆解回代必须回到原刀轴 —— 对本项目用到的所有姿态。"""
        worst = 0.0
        for a_deg in (3.0, 10.0, 30.0, 60.0, 90.0, 119.0):
            for c_deg in (0.0, 1.0, 45.0, 90.0, 180.0, 271.0, 359.0):
                k = kin.forward_a(math.radians(a_deg), math.radians(c_deg))
                a2, c2 = kin.inverse_a(k)
                k2 = kin.forward_a(a2, c2)
                worst = max(worst, max(abs(k[i] - k2[i]) for i in range(3)))
        self.assertLess(worst, 1e-12, '正解→逆解→正解 最大偏差 %.3e' % worst)

    def test_inverse_known_angles(self):
        a, c = kin.inverse_a((0.0, -1.0, 0.0))
        self.assertAlmostEqual(math.degrees(a), 90.0, places=9)
        self.assertAlmostEqual(math.degrees(c), 0.0, places=9)

        a, c = kin.inverse_a((1.0, 0.0, 0.0))
        self.assertAlmostEqual(math.degrees(a), 90.0, places=9)
        self.assertAlmostEqual(math.degrees(c), 90.0, places=9)

    def test_inverse_normalises_input(self):
        a1, c1 = kin.inverse_a((0.0, -60.0, 0.0))
        a2, c2 = kin.inverse_a((0.0, -1.0, 0.0))
        self.assertAlmostEqual(a1, a2, places=12)
        self.assertAlmostEqual(c1, c2, places=12)

    def test_zero_vector_rejected(self):
        with self.assertRaises(ValueError):
            kin.inverse_a((0.0, 0.0, 0.0))


class TestSingularity(unittest.TestCase):

    def test_parallel_axis_raises(self):
        for k in ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (1e-9, 0.0, 1.0)):
            with self.assertRaises(kin.SingularPointError):
                kin.inverse_a(k)

    def test_is_singular_predicate(self):
        self.assertTrue(kin.is_singular((0.0, 0.0, 1.0)))
        self.assertFalse(kin.is_singular((0.0, -0.2, 0.98)))

    def test_keep_c_allows_solving_through(self):
        q = kin.inverse((5.0, 5.0, 0.0), (0.0, 0.0, 1.0), 80.0, keep_c=math.radians(30.0))
        self.assertAlmostEqual(math.degrees(q[3]), 0.0, places=9)
        self.assertAlmostEqual(math.degrees(q[4]), 30.0, places=9)

    def test_without_keep_c_raises(self):
        with self.assertRaises(kin.SingularPointError):
            kin.inverse((5.0, 5.0, 0.0), (0.0, 0.0, 1.0), 80.0)


class TestRTCP(unittest.TestCase):

    def test_pivot_offset_is_tool_length_along_axis(self):
        L = 80.0
        p = (10.0, 20.0, -5.0)
        k = kin.forward_a(math.radians(30.0), math.radians(40.0))
        pivot = kin.pivot_from_tip(p, k, L)
        for i in range(3):
            self.assertAlmostEqual(pivot[i] - p[i], L * k[i], places=12)

    def test_inverse_places_tip_exactly(self):
        L = 80.0
        for a_deg in (5.0, 25.0, 60.0, 95.0):
            k = kin.forward_a(math.radians(a_deg), math.radians(70.0))
            p = (12.0, -3.0, 4.0)
            q = kin.inverse(p, k, L)
            p2, k2 = kin.forward(q, L)
            for i in range(3):
                self.assertAlmostEqual(p2[i], p[i], places=10)

    def test_rtcp_error_matches_closed_form(self):
        """无 RTCP 的刀尖偏差应等于 2L·sin(θ/2)。"""
        L = 80.0
        for deg in (5.0, 20.0, 45.0, 90.0):
            k = kin.forward_a(math.radians(deg), math.radians(15.0))
            err, _ = kin.rtcp_error((0.0, 0.0, 0.0), k, L)
            self.assertAlmostEqual(err, 2.0 * L * math.sin(math.radians(deg) / 2.0), places=9)

    def test_rtcp_error_at_ninety_deg_is_root_two_L(self):
        L = 80.0
        k = kin.forward_a(math.radians(90.0), 0.0)
        err, _ = kin.rtcp_error((0.0, 0.0, 0.0), k, L)
        self.assertAlmostEqual(err, L * math.sqrt(2.0), places=9)

    def test_no_error_when_axis_unchanged(self):
        k = kin.forward_a(math.radians(30.0), math.radians(40.0))
        err, _ = kin.rtcp_error((0.0, 0.0, 0.0), k, 80.0, k_ref=k)
        self.assertAlmostEqual(err, 0.0, places=12)


class TestLimits(unittest.TestCase):

    def test_a_out_of_range_raises(self):
        old = kin.set_limits(-30.0, 30.0)
        try:
            k = kin.forward_a(math.radians(60.0), 0.0)
            with self.assertRaises(kin.LimitError):
                kin.inverse((0.0, 0.0, 0.0), k, 80.0)
        finally:
            kin.set_limits(math.degrees(old[0]), math.degrees(old[1]))

    def test_tilt_angle(self):
        self.assertAlmostEqual(kin.tilt_angle((0.0, 0.0, 1.0)), 0.0, places=12)
        self.assertAlmostEqual(math.degrees(kin.tilt_angle((0.0, -1.0, 0.0))), 90.0, places=9)


if __name__ == '__main__':
    unittest.main(verbosity=2)
