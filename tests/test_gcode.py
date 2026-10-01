# -*- coding: utf-8 -*-
"""G 代码解析测试。"""
import math
import unittest

from fiveaxis import gcode as gc

PROGRAM = """
%
O1001 (S 形试件粗加工示例)
G21 G90 G54 G18
G43 H01 Z50.0
S8000 M03
G00 X0 Y0 Z10.0
G01 Z-2.0 F300
G01 X40.0 Z-6.0 A15.0 C0.0 F800
G02 X60.0 Z-6.0 I0.0 J10.0 A25.0
G01 X90.0 Z-4.0 A-15.0
G03 X120.0 Z-4.0 I0.0 J15.0 A0.0
G00 Z50.0
M09
M30
%
"""


class TestParsing(unittest.TestCase):

    def setUp(self):
        self.prog = gc.parse(PROGRAM)

    def test_block_count(self):
        self.assertEqual(len(self.prog.blocks), 13)

    def test_line_numbers_preserved(self):
        self.assertEqual(self.prog.blocks[0].line, 3)

    def test_modal_folding(self):
        self.assertEqual(self.prog.plane, 'ZX')
        self.assertTrue(self.prog.absolute)
        self.assertEqual(self.prog.units, 'mm')
        # blocks[0] 是 O1001 那行，**在 G18 之前**，它看到的模态状态本来就还是 XY ——
        # 这不是 bug，而是模态指令"从出现处往后生效"的正确含义
        self.assertEqual(self.prog.blocks[0].words['_PLANE'], 'XY')
        for b in self.prog.blocks[1:]:
            self.assertEqual(b.words['_PLANE'], 'ZX', '第 %d 行平面应为 ZX' % b.line)

    def test_g_codes_collected(self):
        b = self.prog.blocks[1]
        self.assertEqual(b.g, [21, 90, 54, 18])

    def test_m_codes_collected(self):
        # blocks[3] = 'S8000 M03'
        self.assertEqual(self.prog.blocks[3].m, [3])
        self.assertEqual(self.prog.blocks[-2].m, [9])
        self.assertEqual(self.prog.blocks[-1].m, [30])

    def test_words(self):
        # blocks[4] = 'G00 X0 Y0 Z10.0'
        b = self.prog.blocks[4]
        self.assertAlmostEqual(b.words['X'], 0.0)
        self.assertAlmostEqual(b.words['Z'], 10.0)

    def test_comment_extracted(self):
        self.assertEqual(self.prog.blocks[0].comment, 'S 形试件粗加工示例')

    def test_no_unknown_codes(self):
        self.assertEqual(self.prog.unknown, [])

    def test_stats(self):
        st = gc.stats(self.prog)
        self.assertEqual(st['moves'], 7)
        self.assertEqual(st['rapid'], 2)
        self.assertEqual(st['linear'], 3)
        self.assertEqual(st['arc_cw'], 1)
        self.assertEqual(st['arc_ccw'], 1)


class TestComments(unittest.TestCase):

    def test_semicolon_comment(self):
        prog = gc.parse('G00 X1.0 ; go home\n')
        self.assertEqual(prog.blocks[0].comment, 'go home')

    def test_paren_comment_preserved_with_semicolon(self):
        body, comment = gc._strip_comment('G01 X1 (rapid) ; tail')
        self.assertIn('rapid', comment)
        self.assertIn('tail', comment)
        self.assertNotIn('(', body)

    def test_blank_lines_skipped(self):
        prog = gc.parse('\n\n   \nG00 X1.0\n')
        self.assertEqual(len(prog.blocks), 1)


class TestUnknownCodes(unittest.TestCase):

    def test_unimplemented_g_is_recorded_not_silently_dropped(self):
        prog = gc.parse('G81 X1.0 Y2.0\n')
        self.assertEqual(len(prog.unknown), 1)
        self.assertEqual(prog.unknown[0][1], 'G81')

    def test_truly_unknown_g_is_recorded(self):
        prog = gc.parse('G999 X1.0\n')
        self.assertEqual(prog.unknown[0][1], 'G999')

    def test_decimal_g_rejected(self):
        with self.assertRaises(gc.GCodeError):
            gc.parse('G1.5 X1.0\n')


class TestArcs(unittest.TestCase):

    def test_semicircle_from_ijk(self):
        prev = (0.0, 0.0, 0.0)
        b = gc.parse('G17 G02 X10.0 Y0.0 I5.0 J0.0\n').blocks[0]
        arc = gc.arc_from_blocks(prev, b)
        self.assertAlmostEqual(arc.radius, 5.0, places=9)
        self.assertAlmostEqual(abs(arc.angle_deg), 180.0, places=6)
        self.assertTrue(arc.cw)

    def test_quarter_arc_ccw(self):
        prev = (5.0, 0.0, 0.0)
        b = gc.parse('G17 G03 X0.0 Y5.0 I-5.0 J0.0\n').blocks[0]
        arc = gc.arc_from_blocks(prev, b)
        self.assertAlmostEqual(arc.radius, 5.0, places=9)
        self.assertAlmostEqual(arc.angle_deg, 90.0, places=6)
        self.assertFalse(arc.cw)

    def test_arc_point_matches_endpoints(self):
        prev = (5.0, 0.0, 0.0)
        b = gc.parse('G17 G03 X0.0 Y5.0 I-5.0 J0.0\n').blocks[0]
        arc = gc.arc_from_blocks(prev, b)
        p0 = gc.arc_point(arc, 0.0)
        p1 = gc.arc_point(arc, 1.0)
        for i in range(3):
            self.assertAlmostEqual(p0[i], arc.start[i], places=9)
            self.assertAlmostEqual(p1[i], arc.end[i], places=9)

    def test_arc_midpoint_lies_on_circle(self):
        prev = (5.0, 0.0, 0.0)
        b = gc.parse('G17 G03 X0.0 Y5.0 I-5.0 J0.0\n').blocks[0]
        arc = gc.arc_from_blocks(prev, b)
        pm = gc.arc_point(arc, 0.5)
        d = math.hypot(pm[0] - arc.center[0], pm[1] - arc.center[1])
        self.assertAlmostEqual(d, arc.radius, places=9)

    def test_zx_plane_arc(self):
        """G18 下圆弧在 ZX 平面内，与 demo 程序一致。"""
        prev = (40.0, 0.0, -6.0)
        b = gc.parse('G18 G02 X60.0 Z-6.0 I0.0 J10.0\n').blocks[0]
        arc = gc.arc_from_blocks(prev, b)
        self.assertEqual(arc.plane, 'ZX')
        self.assertAlmostEqual(arc.radius, 10.0, places=9)
        self.assertAlmostEqual(arc.center[0], 50.0, places=9)
        self.assertAlmostEqual(arc.center[2], -6.0, places=9)

    def test_radius_form(self):
        prev = (0.0, 0.0, 0.0)
        b = gc.parse('G17 G02 X10.0 Y0.0 R5.0\n').blocks[0]
        arc = gc.arc_from_blocks(prev, b)
        self.assertAlmostEqual(arc.radius, 5.0, places=9)

    def test_radius_too_small_rejected(self):
        prev = (0.0, 0.0, 0.0)
        b = gc.parse('G17 G02 X10.0 Y0.0 R1.0\n').blocks[0]
        with self.assertRaises(gc.GCodeError):
            gc.arc_from_blocks(prev, b)

    def test_missing_centre_rejected(self):
        prev = (0.0, 0.0, 0.0)
        b = gc.parse('G17 G02 X10.0 Y0.0\n').blocks[0]
        with self.assertRaises(gc.GCodeError):
            gc.arc_from_blocks(prev, b)

    def test_coincident_endpoints_rejected(self):
        prev = (0.0, 0.0, 0.0)
        b = gc.parse('G17 G02 X0.0 Y0.0 R5.0\n').blocks[0]
        with self.assertRaises(gc.GCodeError):
            gc.arc_from_blocks(prev, b)


class TestIncrementalAndUnits(unittest.TestCase):

    def test_incremental_modal(self):
        prog = gc.parse('G91\nG01 X1.0\nG01 X2.0\n')
        self.assertFalse(prog.absolute)
        self.assertFalse(prog.blocks[1].words['_ABS'])

    def test_back_to_absolute(self):
        prog = gc.parse('G91\nG90\nG01 X1.0\n')
        self.assertTrue(prog.blocks[1].words['_ABS'])

    def test_inch_units(self):
        prog = gc.parse('G20\nG01 X1.0\n')
        self.assertEqual(prog.units, 'inch')

    def test_plane_switching(self):
        prog = gc.parse('G17\nG18\nG19\n')
        self.assertEqual(prog.blocks[0].words['_PLANE'], 'XY')
        self.assertEqual(prog.blocks[1].words['_PLANE'], 'ZX')
        self.assertEqual(prog.blocks[2].words['_PLANE'], 'YZ')


class TestLexing(unittest.TestCase):

    def test_lowercase_accepted(self):
        prog = gc.parse('g01 x1.5 y-2.25\n')
        self.assertAlmostEqual(prog.blocks[0].words['X'], 1.5)
        self.assertAlmostEqual(prog.blocks[0].words['Y'], -2.25)

    def test_no_space_between_words(self):
        prog = gc.parse('G01X10.0Y20.0Z-3.0\n')
        w = prog.blocks[0].words
        self.assertAlmostEqual(w['X'], 10.0)
        self.assertAlmostEqual(w['Y'], 20.0)
        self.assertAlmostEqual(w['Z'], -3.0)

    def test_negative_and_leading_dot(self):
        prog = gc.parse('G01 X-.5 Y+.25\n')
        self.assertAlmostEqual(prog.blocks[0].words['X'], -0.5)
        self.assertAlmostEqual(prog.blocks[0].words['Y'], 0.25)

    def test_n_line_number(self):
        prog = gc.parse('N100 G01 X1.0\n')
        self.assertEqual(prog.blocks[0].n, 100)

    def test_moves_filter(self):
        prog = gc.parse(PROGRAM)
        self.assertTrue(all(any(g in (0, 1, 2, 3) for g in b.g) for b in prog.moves()))


if __name__ == '__main__':
    unittest.main(verbosity=2)
