# -*- coding: utf-8 -*-
"""插补与误差：五轴加工最核心的两个工艺问题。

1. **非线性误差（nonlinearity error）**
   CAM 输出的是刀位数据（刀尖点 + 刀轴矢量），两个刀位点之间是**刀尖走直线**。
   但大多数控制器在 ``G00`` / ``G01`` 时是按**关节轴各自线性插补**的 ——
   关节空间走直线，刀尖在工件坐标系里走的就是一条**曲线**。
   这条曲线偏离理想直线的最大距离，就是非线性误差。
   摆角越大、刀越长，误差越大；这也是五轴加工必须限制步距、或开启 RTCP/RTCP-C 类
   插补功能的原因。

2. **弦高误差（chord error）**
   圆弧离散成直线段时的逼近误差，用来决定分段数::

       n >= (1/2) * sqrt(8 * R * tol / ...)  →  实用近似： n = ceil(sqrt(R / (8 * tol)))

   本项目用标准公式 ``tol = R - sqrt(R^2 - (L/2)^2)`` 反解，比较稳。

另外给出**关节速度放大比** —— 关节空间线性插补时各轴速度的比例，
用于观察奇异点附近 C 轴速度暴涨的现象（摆到 A≈0 时 C 轴解不唯一，实际机床会突然甩头）。
"""
import math

from .kinematics import forward, inverse, is_singular, SingularPointError


def lerp(a, b, t):
    return a + (b - a) * t


def lerp_q(q0, q1, t):
    return tuple(lerp(q0[i], q1[i], t) for i in range(len(q0)))


def point_segment_distance(p, a, b):
    """点 ``p`` 到线段 ``ab`` 的距离。"""
    abx, aby, abz = b[0] - a[0], b[1] - a[1], b[2] - a[2]
    apx, apy, apz = p[0] - a[0], p[1] - a[1], p[2] - a[2]
    denom = abx * abx + aby * aby + abz * abz
    if denom == 0.0:
        return math.sqrt(apx * apx + apy * apy + apz * apz)
    t = (apx * abx + apy * aby + apz * abz) / denom
    t = max(0.0, min(1.0, t))
    cx, cy, cz = a[0] + t * abx, a[1] + t * aby, a[2] + t * abz
    return math.sqrt((p[0] - cx) ** 2 + (p[1] - cy) ** 2 + (p[2] - cz) ** 2)


def nonlinearity(p0, k0, p1, k1, tool_len, n=200, c_hint=None):
    """两个刀位点之间按**关节空间线性插补**时的非线性误差。

    参数
    ----
    p0, k0 : 起点刀尖点与刀轴矢量
    p1, k1 : 终点刀尖点与刀轴矢量
    tool_len : pivot 到刀尖的距离（mm）
    n : 采样段数（越大越接近真实最大值）
    c_hint : 落在奇异点上时冻结使用的 C 值。**默认 ``None``，即不冻结、直接报错** ——
        奇异点是工艺上要避开的位置，不该被静默算成一个数。确实想强行算过去时才显式传值。

    返回 ``(max_err, samples)``，``samples`` 是 ``[(t, tip, err), ...]``。
    """
    if n < 1:
        raise ValueError('采样段数 n 必须 >= 1')

    q0 = inverse(p0, k0, tool_len, keep_c=c_hint)
    q1 = inverse(p1, k1, tool_len, keep_c=c_hint)

    samples = []
    worst = 0.0
    for i in range(n + 1):
        t = i / float(n)
        q = lerp_q(q0, q1, t)
        tip, _ = forward(q, tool_len)
        err = point_segment_distance(tip, p0, p1)
        samples.append((t, tip, err))
        worst = max(worst, err)
    return worst, samples


def joint_deltas(q0, q1):
    """两级之间各关节的变化量，返回 ``{'X':.., 'Y':.., 'Z':.., 'A_deg':.., 'C_deg':..}``。"""
    return {
        'X': q1[0] - q0[0],
        'Y': q1[1] - q0[1],
        'Z': q1[2] - q0[2],
        'A_deg': math.degrees(q1[3] - q0[3]),
        'C_deg': math.degrees(q1[4] - q0[4]),
    }


def joint_speed_ratio(p0, k0, p1, k1, tool_len, n=200):
    """关节空间线性插补时，各关节**单位刀尖位移**所需的变化量。

    用来观察奇异点附近旋转轴的"甩头"：``C`` 值随 ``A`` 接近 0 而急剧放大。
    返回 ``{关节名: 每毫米刀尖位移对应的变化量}``。
    """
    q0 = inverse(p0, k0, tool_len)
    q1 = inverse(p1, k1, tool_len)
    tip_len = math.sqrt(sum((p1[i] - p0[i]) ** 2 for i in range(3))) or 1.0
    d = joint_deltas(q0, q1)
    return {k: v / tip_len for k, v in d.items()}


def arc_segments(radius, tol):
    """按弦高误差容差 ``tol`` 计算圆（整圈）需要分多少段。

    弦高 ``tol = R - sqrt(R^2 - (L/2)^2)``，其中 ``L`` 是弦长；
    反解得 ``L = 2*sqrt(2*R*tol - tol^2)``，段数 ``n = ceil(2*pi*R / L)``。
    """
    if radius <= 0:
        raise ValueError('半径必须为正')
    if tol <= 0:
        raise ValueError('弦高容差必须为正')
    if tol >= radius:
        return 3
    chord = 2.0 * math.sqrt(2.0 * radius * tol - tol * tol)
    return max(3, int(math.ceil(2.0 * math.pi * radius / chord)))


def arc_max_chord_error(radius, n):
    """整圆分成 ``n`` 段时的最大弦高误差。"""
    if n < 3:
        raise ValueError('圆弧至少分 3 段')
    half = math.pi / n
    return radius * (1.0 - math.cos(half))


def safe_nonlinearity(p0, k0, p1, k1, tool_len, n=200, c_hint=None):
    """同 :func:`nonlinearity`，但落在奇异点上时返回 ``(None, [])`` 而不是抛异常。

    批量跑刀路时用这个：一条刀路上有几个奇异段是正常的，
    不该让整条刀路算不出来 —— 但要能被数出来（见 :func:`fiveaxis.stest.evaluate`
    返回的 ``singular_segments``）。
    """
    try:
        return nonlinearity(p0, k0, p1, k1, tool_len, n=n, c_hint=c_hint)
    except (SingularPointError, ValueError):
        return None, []
