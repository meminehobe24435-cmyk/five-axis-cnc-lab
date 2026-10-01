# -*- coding: utf-8 -*-
"""五轴（AC 双摆头）运动学：正解 / 逆解 / RTCP / 奇异点。

坐标与符号约定（全项目唯一口径）
--------------------------------
机床构型：**A 轴（绕 X 摆动）+ C 轴（绕 Z 回转）双摆头**，工件不摆动。

* 关节量 ``q = (x, y, z, a, c)``：直线轴单位 mm，旋转轴单位**弧度**。
* ``(x, y, z)`` 是**摆头回转中心（pivot）**在机床坐标系中的位置，
  **不是刀尖点** —— 这是本项目最容易搞错的一点。
* 刀轴单位矢量 ``k``：**由刀尖指向刀柄**。关节量为零时 ``k = (0, 0, 1)``。
* 刀尖点 ``p`` 与 pivot 的关系：``p = pivot - L * k``（``L`` 为 pivot 到刀尖的距离）。

刀轴矢量的推导（先绕 C 转、再绕 A 摆，即 C 轴带着 A 轴走）::

    k = Rz(c) @ Rx(a) @ (0, 0, 1)
      = ( sin(a)*sin(c), -sin(a)*cos(c), cos(a) )

逆解::

    a = arccos(k_z)          a ∈ [0, π)
    c = atan2(k_x, -k_y)     （sin(a) 不为零时）

RTCP（Rotational Tool Center Point，刀尖点跟随）
------------------------------------------------
CAM 输出的是**刀位数据**（刀尖点 ``p`` + 刀轴矢量 ``k``），而机床需要的是关节量。
RTCP 的含义是：把 ``(p, k)`` 反解成 ``(a, c)`` 之后，直线轴要**自动补偿**摆头旋转带来的偏移，
使刀尖点始终停在编程位置上::

    pivot = p + L * k

没有 RTCP 时旋转轴一动刀尖就会跟着跑偏，刀位数据无法直接使用。本项目把这条关系写进
:func:`inverse`，并用 :func:`rtcp_error` 量化"不启用 RTCP 会偏多少"。
"""
import math

#: A 轴行程（弧度）。默认取常见的 ±120°，可用 :func:`set_limits` 改。
A_LIMIT = (math.radians(-120.0), math.radians(120.0))

#: 判定奇异点的阈值：|sin(a)| 小于它就认为刀轴与 Z 轴平行。
#: 取 1e-6 而不是 0，是因为浮点 arccos 在 k_z≈±1 时给出的 a 有 ~1e-8 量级的抖动。
SINGULAR_EPS = 1e-6


class SingularPointError(ValueError):
    """刀轴与 Z 轴平行时 C 轴解不唯一，无法给出确定的关节量。"""


class LimitError(ValueError):
    """关节量超出机床行程。"""


def _check_a(a):
    lo, hi = A_LIMIT
    if not (lo - 1e-12 <= a <= hi + 1e-12):
        raise LimitError("A 轴 %.4f° 超出行程 [%.1f°, %.1f°]"
                         % (math.degrees(a), math.degrees(lo), math.degrees(hi)))


def set_limits(a_lo_deg, a_hi_deg):
    """设置 A 轴行程（度）。返回旧值，便于测试里恢复。"""
    global A_LIMIT
    old = A_LIMIT
    A_LIMIT = (math.radians(a_lo_deg), math.radians(a_hi_deg))
    return old


def forward_a(a, c):
    """由 (a, c) 得到刀轴单位矢量 k = (kx, ky, kz)。"""
    sa, ca = math.sin(a), math.cos(a)
    sc, cc = math.sin(c), math.cos(c)
    return (sa * sc, -sa * cc, ca)


def inverse_a(k):
    """由刀轴矢量 k 反解 (a, c)。

    奇异点（``|k_z| ≈ 1``）抛 :class:`SingularPointError` —— 此时刀轴与 Z 轴平行，
    C 轴取任何值都能得到同一个刀轴，逆解不唯一。真实控制器靠"保持 C 不动"
    或走专用奇异点策略处理，本项目选择**显式报错**而不是悄悄返回 0。
    """
    kx, ky, kz = k
    n = math.sqrt(kx * kx + ky * ky + kz * kz)
    if n == 0.0:
        raise ValueError("刀轴矢量不能是零向量")
    kx, ky, kz = kx / n, ky / n, kz / n

    s = math.sqrt(kx * kx + ky * ky)
    if s < SINGULAR_EPS:
        raise SingularPointError("刀轴近似平行于 Z 轴（|kx,ky| = %.3e），C 轴解不唯一" % s)

    a = math.atan2(s, kz)          # ∈ [0, π)，与 arccos(kz) 等价但数值更稳
    c = math.atan2(kx, -ky)
    return a, c


def is_singular(k, eps=SINGULAR_EPS):
    """刀轴是否落在奇异点附近。"""
    kx, ky, _ = k
    return math.sqrt(kx * kx + ky * ky) < eps


def forward(q, tool_len):
    """正解：关节量 -> (刀尖点 p, 刀轴矢量 k)。

    ``q`` 为 ``(x, y, z, a, c)``。
    """
    x, y, z, a, c = q
    k = forward_a(a, c)
    p = (x - tool_len * k[0], y - tool_len * k[1], z - tool_len * k[2])
    return p, k


def inverse(p, k, tool_len, keep_c=None):
    """逆解（含 RTCP）：刀位数据 -> 关节量。

    ``p`` 为刀尖点，``k`` 为刀轴矢量，``tool_len`` 为 pivot 到刀尖的距离。

    返回 ``(x, y, z, a, c)``，其中 ``(x, y, z)`` 是 pivot 位置::

        pivot = p + tool_len * k

    ``keep_c``：当刀位落在奇异点上时，用这个 C 值代替（模拟控制器"保持 C 不动"）。
    传 ``None`` 则抛 :class:`SingularPointError`。
    """
    try:
        a, c = inverse_a(k)
    except SingularPointError:
        if keep_c is None:
            raise
        a, c = 0.0, keep_c
    _check_a(a)

    kx, ky, kz = _unit(k)
    x = p[0] + tool_len * kx
    y = p[1] + tool_len * ky
    z = p[2] + tool_len * kz
    return (x, y, z, a, c)


def _unit(v):
    n = math.sqrt(v[0] ** 2 + v[1] ** 2 + v[2] ** 2)
    if n == 0.0:
        raise ValueError("零向量没有单位化方向")
    return (v[0] / n, v[1] / n, v[2] / n)


def rtcp_error(p, k, tool_len, k_ref=(0.0, 0.0, 1.0)):
    """不启用 RTCP 时，因摆头旋转未被补偿而产生的刀尖偏差。

    设定：机床在**参考刀轴** ``k_ref``（默认 A=C=0，即 ``(0,0,1)``）处对好刀
    （即 G43 刀长补偿是在这个姿态下做的），此后直线轴按编程点走、
    但旋转**不做 RTCP 补偿**。于是刀尖实际位置相对编程位置的偏移为::

        Δ = tool_len * (k_ref - k)

    摆头摆过 θ 角时 ``|Δ| = 2 * tool_len * sin(θ/2)`` ——
    **摆 90° 时偏差约为刀具长度的 1.414 倍**，这就是五轴必须开 RTCP 的原因。

    返回 ``(error_mm, delta_vector)``。
    """
    kx, ky, kz = _unit(k)
    rx, ry, rz = _unit(k_ref)
    dx = tool_len * (rx - kx)
    dy = tool_len * (ry - ky)
    dz = tool_len * (rz - kz)
    return math.sqrt(dx * dx + dy * dy + dz * dz), (dx, dy, dz)


def tilt_angle(k, k_ref=(0.0, 0.0, 1.0)):
    """刀轴相对参考姿态的摆角（弧度），取值范围 [0, π]。"""
    kx, ky, kz = _unit(k)
    rx, ry, rz = _unit(k_ref)
    d = kx * rx + ky * ry + kz * rz
    return math.acos(max(-1.0, min(1.0, d)))


def pivot_from_tip(p, k, tool_len):
    """便捷函数：刀尖点 + 刀轴 -> pivot（即 RTCP 的补偿量来源）。"""
    kx, ky, kz = _unit(k)
    return (p[0] + tool_len * kx, p[1] + tool_len * ky, p[2] + tool_len * kz)


def joint_distance(q1, q2, rot_weight=1.0):
    """两个关节位形之间的距离（直线轴 mm，旋转轴按 ``rot_weight`` mm/rad 折算）。

    只用于插补取样与"哪条路径更省"的粗略比较，不是机床动力学模型。
    """
    d = math.sqrt(sum((q1[i] - q2[i]) ** 2 for i in range(3)))
    dr = math.sqrt(sum((q1[i] - q2[i]) ** 2 for i in range(3, 5)))
    return math.sqrt(d * d + (rot_weight * dr) ** 2)
