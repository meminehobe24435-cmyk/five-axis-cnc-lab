# -*- coding: utf-8 -*-
"""S 形试件：五轴机床精度检验的典型试件，以及**工艺规范**的反推。

为什么用 S 形试件
----------------
S 形试件同时具备五轴加工最难的几件事：**连续变曲率**、**连续变倾斜角**、
**开角与闭角交替**、以及薄壁带来的刚度问题。它比圆锥台、NAS 件这类试件更能暴露
五轴机床的动态误差，因此被写进 ISO 10791-7 的附录，也是国内机床验收常用的检验件。

⚠️ **本项目实现的是 S 形试件的运动学特征（几何曲线 + 刀轴摆动），不是真实试件的
   实体造型与标准检验流程。** 真实检验用三坐标测量机测实体轮廓，
   本项目只算**刀位层面的运动学误差** —— 也就是"机床还没切，光走刀就偏了多少"。
   这一层误差是工艺策划阶段必须预估的，但它不等于最终零件的轮廓误差。

三个用途
--------
1. :func:`toolpath` 生成 S 形试件的刀位数据（刀尖点 + 刀轴矢量序列）；
2. :func:`evaluate` 逐段评估**非线性误差**、**摆角**与**无 RTCP 时的刀尖偏差**；
3. :func:`max_step_for_tolerance` **反推工艺规范**：给定非线性误差容差，
   最大允许的刀位步距是多少 —— 这就是编制五轴加工程序时要写进工艺卡的那个数。
"""
import math

from .interp import nonlinearity, safe_nonlinearity, joint_deltas
from .kinematics import forward, inverse, rtcp_error, tilt_angle, is_singular

#: S 形试件默认几何（mm）：长度、S 形幅值
DEFAULT_LENGTH = 120.0
DEFAULT_AMPLITUDE = 22.0
#: 刀轴倾角的中间值（度）。抬到非零是为了**避开奇异点**，见 :func:`tool_axis`。
DEFAULT_TILT_MID = 20.0
#: 刀轴倾角的摆动幅值（度）—— 连续变化，模拟开角 / 闭角转换
DEFAULT_TILT_DEG = 15.0


def s_curve(u, length=DEFAULT_LENGTH, amplitude=DEFAULT_AMPLITUDE):
    """S 形中心曲线上的点，``u ∈ [0, 1]``。

    ``p(u) = (L*u, 0, A*sin(2*pi*u))`` —— 一个完整 S，曲率符号在中点翻转。
    """
    return (length * u, 0.0, amplitude * math.sin(2.0 * math.pi * u))


def s_tangent(u, length=DEFAULT_LENGTH, amplitude=DEFAULT_AMPLITUDE):
    """中心曲线在 ``u`` 处的单位切向（XZ 平面内）。"""
    dx = length
    dz = amplitude * 2.0 * math.pi * math.cos(2.0 * math.pi * u)
    n = math.hypot(dx, dz)
    return (dx / n, 0.0, dz / n)


def s_curvature(u, length=DEFAULT_LENGTH, amplitude=DEFAULT_AMPLITUDE):
    """中心曲线在 ``u`` 处的曲率（带符号，XZ 平面内）。"""
    dx = length
    dz1 = amplitude * 2.0 * math.pi * math.cos(2.0 * math.pi * u)
    dz2 = -amplitude * (2.0 * math.pi) ** 2 * math.sin(2.0 * math.pi * u)
    denom = (dx * dx + dz1 * dz1) ** 1.5
    return (dx * dz2 - dz1 * 0.0) / denom


def tool_axis(u, tilt_deg=DEFAULT_TILT_DEG, tilt_mid_deg=DEFAULT_TILT_MID,
              length=DEFAULT_LENGTH, amplitude=DEFAULT_AMPLITUDE):
    """刀轴单位矢量：在 **XZ 平面内**相对 +Z 倾斜，倾角连续变化。

    ``beta(u) = tilt_mid_deg + tilt_deg * sin(2*pi*u)``，于是::

        k(u) = (sin beta, 0, cos beta)

    **为什么倾斜角要带一个中间值**：``beta`` 若穿过 0，刀轴就与 Z 轴平行 ——
    那正是 AC 双摆头的**奇异点**（C 轴解不唯一），整段刀路会解不出来。
    S 形试件在真实机床上确实会经过奇异区，本项目把中间值抬到 :data:`DEFAULT_TILT_MID`，
    让倾角在一个**不穿越奇异点的区间内连续变化**，这样算出来的是工艺参数的影响，
    而不是被奇异点淹没。奇异点本身在 :func:`fiveaxis.kinematics.inverse_a` 里单独处理与测试。

    ⚠️ 早先版本把倾斜方向取成"曲线法向"，切向沿 X 时法向恰好等于 +Z，
    倾斜量被完全抵消（倾角恒为 0）—— 这是一个真实踩过的坑，见 README「开发中踩到的问题」。
    """
    beta = math.radians(tilt_mid_deg + tilt_deg * math.sin(2.0 * math.pi * u))
    return (math.sin(beta), 0.0, math.cos(beta))


def tilt_at(u, tilt_deg=DEFAULT_TILT_DEG, tilt_mid_deg=DEFAULT_TILT_MID):
    """``u`` 处相对 +Z 的名义倾角（度）。"""
    return tilt_mid_deg + tilt_deg * math.sin(2.0 * math.pi * u)


def toolpath(n, tilt_deg=DEFAULT_TILT_DEG, tilt_mid_deg=DEFAULT_TILT_MID,
             length=DEFAULT_LENGTH, amplitude=DEFAULT_AMPLITUDE, tip_offset=None):
    """生成 S 形试件的刀位数据。

    返回 ``[(u, p, k), ...]``，``n`` 是段数（点数为 ``n+1``）。
    ``tip_offset`` 可给刀尖点叠加一个固定偏移，用来模拟试件在工作台上的安装位置。
    """
    if n < 1:
        raise ValueError('段数 n 必须 >= 1')
    ox, oy, oz = tip_offset or (0.0, 0.0, 0.0)
    path = []
    for i in range(n + 1):
        u = i / float(n)
        px, py, pz = s_curve(u, length, amplitude)
        path.append((u, (px + ox, py + oy, pz + oz),
                     tool_axis(u, tilt_deg, tilt_mid_deg, length, amplitude)))
    return path


def evaluate(path, tool_len, samples=60, c_hint=None):
    """逐段评估刀路。

    对每一段返回：摆角、非线性误差、无 RTCP 时的刀尖偏差、关节变化量。
    落在奇异点上的段会被**计入 ``singular_segments``** 并把该段的误差与关节量置 ``None``，
    而不是静默丢弃、也不是让整条刀路算不出来。
    """
    if len(path) < 2:
        raise ValueError('刀路至少要有两个刀位点')

    segments = []
    singular = 0
    for i in range(len(path) - 1):
        u0, p0, k0 = path[i]
        u1, p1, k1 = path[i + 1]

        err, _ = safe_nonlinearity(p0, k0, p1, k1, tool_len, n=samples, c_hint=c_hint)

        rtcp_err, _ = rtcp_error((0, 0, 0), k1, tool_len)
        tilt = math.degrees(tilt_angle(k1))

        d = None
        step = math.sqrt((p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2 + (p1[2] - p0[2]) ** 2)
        try:
            q0 = inverse(p0, k0, tool_len, keep_c=c_hint)
            q1 = inverse(p1, k1, tool_len, keep_c=c_hint)
            d = joint_deltas(q0, q1)
        except (SingularPointError, ValueError):
            pass

        if err is None:
            singular += 1

        segments.append({
            'index': i,
            'u': u1,
            'step_mm': step,
            'tilt_deg': tilt,
            'nonlinearity_mm': err,
            'rtcp_error_mm': rtcp_err,
            'dA_deg': d['A_deg'] if d else None,
            'dC_deg': d['C_deg'] if d else None,
        })

    errs = [s['nonlinearity_mm'] for s in segments if s['nonlinearity_mm'] is not None]
    tilts = [s['tilt_deg'] for s in segments]
    steps = [s['step_mm'] for s in segments]

    return {
        'segments': segments,
        'segments_total': len(segments),
        'singular_segments': singular,
        'max_nonlinearity_mm': max(errs) if errs else None,
        'mean_nonlinearity_mm': (sum(errs) / len(errs)) if errs else None,
        'max_tilt_deg': max(tilts) if tilts else None,
        'max_step_mm': max(steps) if steps else None,
        'max_rtcp_error_mm': max(s['rtcp_error_mm'] for s in segments),
    }


def max_step_for_tolerance(tolerance_mm, tool_len, tilt_deg=DEFAULT_TILT_DEG,
                           tilt_mid_deg=DEFAULT_TILT_MID, lo=0.02, hi=20.0,
                           iters=30, u_grid=7, samples=40):
    """**反推工艺规范**：给定非线性误差容差，允许的最大刀位步距是多少。

    非线性误差主要来自**刀轴转角**，而转角变化率 ``dβ/du`` 在曲线上各处不同
    （``beta(u) = tilt_mid + tilt_deg*sin(2*pi*u)`` 对 ``u`` 求导后正比于 ``cos(2*pi*u)``，
    在 ``u = 0 / 0.5 / 1`` 处最大）。所以这里**沿曲线扫多个位置**，
    对每个位置二分求允许步距，最后取**最保守的那个（最小值）**——
    工艺卡上要写的是"整条刀路都不能超差"，不是"某一点刚好合格"。

    ⚠️ 早先版本只取 ``u = 0.25`` 一个点，而那里 ``cos(2*pi*u) = 0``、转角变化率为零，
    算出来的允许步距**偏大**（偏危险）—— 这是踩过的坑，见 README。

    返回 ``(step_mm, achieved_error_mm)``；若在 ``lo`` 处仍不达标返回 ``(None, err)``。
    """
    if tolerance_mm <= 0:
        raise ValueError('容差必须为正')
    if u_grid < 3:
        raise ValueError('u_grid 至少为 3')

    worst = None

    for gi in range(u_grid):
        u_c = (gi + 0.5) / float(u_grid)

        def err_at(step, _u=u_c):
            du = step / DEFAULT_LENGTH
            u0 = max(0.0, _u - du / 2.0)
            u1 = min(1.0, _u + du / 2.0)
            e, _ = safe_nonlinearity(s_curve(u0), tool_axis(u0, tilt_deg, tilt_mid_deg),
                                     s_curve(u1), tool_axis(u1, tilt_deg, tilt_mid_deg),
                                     tool_len, n=samples)
            return e

        e_lo = err_at(lo)
        if e_lo is None:
            continue
        if e_lo > tolerance_mm:
            worst = (None, e_lo) if worst is None else min(worst, (None, e_lo), key=lambda t: t[1] or 0)
            continue

        a, b = lo, hi
        for _ in range(iters):
            mid = (a + b) / 2.0
            e = err_at(mid)
            if e is None or e > tolerance_mm:
                b = mid
            else:
                a = mid
        cand = (a, err_at(a))
        worst = cand if worst is None else min(worst, cand, key=lambda t: t[0] if t[0] else -1)

    if worst is None:
        return None, None
    return worst


def case_library(tool_lens=(40.0, 80.0, 120.0), tilts=(20.0, 35.0, 50.0),
                 tolerances=(0.005, 0.010, 0.020)):
    """生成**典型零件加工案例库**：不同刀具长度 / 摆角下，各容差对应的允许步距。

    这张表就是工艺策划时"这个件能不能用这套刀、步距给多少"的依据。
    返回 ``{'tolerances': [...], 'rows': [...]}``。
    """
    rows = []
    for L in tool_lens:
        for tdeg in tilts:
            row = {'tool_len_mm': L, 'tilt_deg': tdeg, 'steps': {}}
            for tol in tolerances:
                step, _ = max_step_for_tolerance(tol, L, tilt_deg=tdeg)
                row['steps'][tol] = step
            rows.append(row)
    return {'tolerances': list(tolerances), 'rows': rows}
