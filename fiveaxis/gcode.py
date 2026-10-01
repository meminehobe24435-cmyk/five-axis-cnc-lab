# -*- coding: utf-8 -*-
"""G 代码解析：把数控程序解析成结构化的程序段，供运动学与精度模块使用。

支持范围（够用于五轴应用测试，**不是**完整数控系统前端）
----------------------------------------------------------
* 运动：``G00`` 快速、``G01`` 直线、``G02`` / ``G03`` 圆弧（``I/J/K`` 与 ``R`` 两种写法）
* 平面：``G17`` XY、``G18`` ZX、``G19`` YZ（决定圆弧的插补平面）
* 坐标系：``G54`` ~ ``G59`` 工件坐标系；``G90`` / ``G91`` 绝对 / 增量
* 刀补：``G43`` 刀具长度正补偿（``H``）、``G49`` 取消
* 工艺：``F`` 进给、``S`` 主轴转速、``M03`` / ``M04`` / ``M05`` 主轴、``M08`` / ``M09`` 冷却、``M30`` 程序结束
* 其他：``N`` 行号、``()`` 与 ``;`` 注释、多指令同行、大小写混写

**明确不支持**（README 边界一节也写了）：固定循环 ``G73``~``G89``、刀具半径补偿 ``G41`` / ``G42``、
``G68`` 坐标旋转、``G93`` / ``G94`` / ``G95`` 时间与每转进给、子程序 ``M98`` / ``M99``、
以及各家控制器的宏与扩展指令。遇到不认识的 G / M 代码会**记进 ``Program.unknown``**，
不静默丢弃 —— 解析器宁可说出来，也不假装看懂了。
"""
import math
import re

_WORD = re.compile(r'([A-Za-z])\s*([-+]?(?:\d+\.?\d*|\.\d+))')
_COMMENT_PAREN = re.compile(r'\([^)]*\)')

#: 已知但没有实现语义的 G 代码（记进 unknown，不报错）
_KNOWN_UNIMPLEMENTED_G = {73, 74, 76, 81, 82, 83, 84, 85, 86, 87, 88, 89,
                          40, 41, 42, 68, 69, 93, 94, 95,
                          28, 30, 92}


class GCodeError(ValueError):
    """语法或语义错误。带行号，便于对着程序改。"""


class Block(object):
    """一个程序段（一行）。"""

    __slots__ = ('n', 'line', 'g', 'm', 'words', 'comment', 'raw')

    def __init__(self, n, line, g, m, words, comment, raw):
        self.n = n               # N 行号（可能为 None）
        self.line = line         # 源文件行号，从 1 开始
        self.g = g               # 本段出现的 G 代码列表，如 [1]
        self.m = m               # 本段出现的 M 代码列表
        self.words = words       # 其余地址字，如 {'X': 10.0, 'Y': 5.0, 'F': 200.0}
        self.comment = comment   # 注释文本（已去掉括号）
        self.raw = raw           # 原始行

    def __repr__(self):
        return '<Block L%s G%s M%s %s>' % (
            self.line, self.g, self.m, {k: round(v, 4) for k, v in self.words.items()})


class Arc(object):
    """圆弧段的几何参数（已解析成圆心 + 方向，便于后续离散化）。"""

    __slots__ = ('plane', 'cw', 'start', 'end', 'center', 'radius', 'angle_deg')

    def __init__(self, plane, cw, start, end, center, radius, angle_deg):
        self.plane = plane       # 'XY' / 'ZX' / 'YZ'
        self.cw = cw             # True = G02
        self.start = start       # 起点（三元组）
        self.end = end           # 终点
        self.center = center     # 圆心（三元组）
        self.radius = radius
        self.angle_deg = angle_deg


class Program(object):
    """解析结果。"""

    def __init__(self, blocks):
        self.blocks = blocks
        self.unknown = []        # [(行号, 'G123')]
        self.units = 'mm'
        self.plane = 'XY'
        self.absolute = True

    def moves(self):
        """只取有运动语义的段（G00/G01/G02/G03）。"""
        return [b for b in self.blocks if any(g in (0, 1, 2, 3) for g in b.g)]


def _strip_comment(line):
    """去掉注释，返回 (剩余正文, 注释文本)。"""
    comments = []

    def _keep(m):
        comments.append(m.group(0)[1:-1].strip())
        return ' '

    body = _COMMENT_PAREN.sub(_keep, line)
    if ';' in body:
        head, _, tail = body.partition(';')
        comments.append(tail.strip())
        body = head
    return body, ' '.join(c for c in comments if c)


def parse(text):
    """解析 G 代码文本，返回 :class:`Program`。"""
    blocks = []
    unknown = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        body, comment = _strip_comment(raw)
        if not body.strip():
            continue

        words = _WORD.findall(body)
        if not words:
            continue

        g, m, rest, n = [], [], {}, None
        for letter, value in words:
            letter = letter.upper()
            number = float(value)
            if letter == 'G':
                code = int(number)
                if number != code:
                    raise GCodeError('第 %d 行：不支持小数 G 代码 G%s' % (lineno, value))
                g.append(code)
                if code in _KNOWN_UNIMPLEMENTED_G:
                    unknown.append((lineno, 'G%02d' % code))
                elif code not in (0, 1, 2, 3, 17, 18, 19, 20, 21, 43, 49, 54, 55, 56, 57, 58, 59,
                                  90, 91, 94):
                    unknown.append((lineno, 'G%02d' % code))
            elif letter == 'M':
                m.append(int(number))
            elif letter == 'N':
                n = int(number)
            else:
                rest[letter] = number

        blocks.append(Block(n, lineno, g, m, rest, comment, raw.rstrip()))

    prog = Program(blocks)
    prog.unknown = unknown
    _fold_modals(prog)
    return prog


def _fold_modals(prog):
    """把模态指令（平面、绝对/增量、单位）沿程序向下传递，写回每个 Block 的 words。

    这样后面的模块不用自己维护状态机，直接看 ``block.words['_PLANE']`` 即可。
    """
    plane = 'XY'
    absolute = True
    units = 'mm'

    for b in prog.blocks:
        for code in b.g:
            if code == 17:
                plane = 'XY'
            elif code == 18:
                plane = 'ZX'
            elif code == 19:
                plane = 'YZ'
            elif code == 90:
                absolute = True
            elif code == 91:
                absolute = False
            elif code == 20:
                units = 'inch'
            elif code == 21:
                units = 'mm'
        b.words['_PLANE'] = plane
        b.words['_ABS'] = absolute
        b.words['_UNITS'] = units

    prog.plane = plane
    prog.absolute = absolute
    prog.units = units


def arc_from_blocks(prev, cur):
    """由「上一段终点 + 本段」构造 :class:`Arc`。

    ``prev`` 为上一段结束时的 ``(x, y, z)``；``cur`` 是含 G02/G03 的 Block。
    """
    plane = cur.words.get('_PLANE', 'XY')
    idx = {'XY': (0, 1, 2), 'ZX': (2, 0, 1), 'YZ': (1, 2, 0)}[plane]
    i1, i2, i3 = idx

    start = list(prev)
    end = list(prev)
    for letter, pos in (('X', 0), ('Y', 1), ('Z', 2)):
        if letter in cur.words:
            end[pos] = cur.words[letter]
    end = tuple(end)

    w = cur.words
    cw = 2 in cur.g

    if 'R' in w:
        r = w['R']
        sx, sy = start[i1], start[i2]
        ex, ey = end[i1], end[i2]
        chord = math.hypot(ex - sx, ey - sy)
        if chord == 0:
            raise GCodeError('第 %d 行：圆弧起点与终点重合，无法由 R 定圆心' % cur.line)
        if abs(r) < chord / 2.0 - 1e-9:
            raise GCodeError('第 %d 行：R=%.4f 小于弦长的一半 %.4f，圆弧不存在'
                             % (cur.line, r, chord / 2.0))
        h = math.sqrt(max(0.0, r * r - (chord / 2.0) ** 2))
        mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
        ux, uy = (ex - sx) / chord, (ey - sy) / chord
        # 优弧 / 劣弧：R 为负表示优弧（>180°），此时圆心在另一侧
        sign = 1.0 if (r > 0) == (not cw) else -1.0
        cx = mx + sign * h * (-uy)
        cy = my + sign * h * ux
        center = list(start)
        center[i1], center[i2] = cx, cy
        radius = abs(r)
    else:
        if not any(k in w for k in ('I', 'J', 'K')):
            raise GCodeError('第 %d 行：G02/G03 缺少 R 或 I/J/K' % cur.line)
        off = {'I': i1, 'J': i2, 'K': i3}
        center = list(start)
        for letter, pos in off.items():
            center[pos] = start[pos] + w.get(letter, 0.0)
        radius = math.hypot(start[i1] - center[i1], start[i2] - center[i2])
        if radius == 0:
            raise GCodeError('第 %d 行：圆弧半径为零' % cur.line)

    a0 = math.atan2(start[i2] - center[i2], start[i1] - center[i1])
    a1 = math.atan2(end[i2] - center[i2], end[i1] - center[i1])
    sweep = a1 - a0
    if cw:
        while sweep >= 0:
            sweep -= 2 * math.pi
    else:
        while sweep <= 0:
            sweep += 2 * math.pi
    if abs(sweep) < 1e-12:
        sweep = -2 * math.pi if cw else 2 * math.pi

    return Arc(plane, cw, tuple(start), end, tuple(center), radius, math.degrees(sweep))


def arc_point(arc, t):
    """圆弧上参数 ``t ∈ [0, 1]`` 处的点（``t`` 按弧长比例）。"""
    idx = {'XY': (0, 1, 2), 'ZX': (2, 0, 1), 'YZ': (1, 2, 0)}[arc.plane]
    i1, i2, i3 = idx
    a0 = math.atan2(arc.start[i2] - arc.center[i2], arc.start[i1] - arc.center[i1])
    sweep = math.radians(arc.angle_deg)
    ang = a0 + sweep * t

    p = list(arc.start)
    p[i1] = arc.center[i1] + arc.radius * math.cos(ang)
    p[i2] = arc.center[i2] + arc.radius * math.sin(ang)
    p[i3] = arc.start[i3] + (arc.end[i3] - arc.start[i3]) * t
    return tuple(p)


def stats(prog):
    """程序统计，用于报告。"""
    moves = prog.moves()
    kinds = {0: 0, 1: 0, 2: 0, 3: 0}
    for b in moves:
        for g in b.g:
            if g in kinds:
                kinds[g] += 1
    return {
        'blocks': len(prog.blocks),
        'moves': len(moves),
        'rapid': kinds[0],
        'linear': kinds[1],
        'arc_cw': kinds[2],
        'arc_ccw': kinds[3],
        'unknown': len(prog.unknown),
        'plane': prog.plane,
        'units': prog.units,
    }
