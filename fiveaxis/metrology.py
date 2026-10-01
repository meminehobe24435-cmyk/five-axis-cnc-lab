# -*- coding: utf-8 -*-
"""机床精度检测数据的分析：定位精度（ISO 230-2）与循圆测试（ISO 230-4）。

⚠️ **本模块是"按标准定义写一遍"的教学/自检实现，不是认证机构的合格判定实现。**
   标准原文对采样条件（环境温度、预热、探针、加载、最少测量次数）有严格要求，
   本项目只实现**数据处理与指标计算**这一层，并且把用到的公式逐条写在 docstring 里，
   便于逐条对照。用它出的报告**不能**当作机床验收报告。

ISO 230-2 定位精度
------------------
对每个目标位置 ``i`` 在正、反两个方向各测若干次::

    x̄↑i, σ↑i   正行程均值与标准差
    x̄↓i, σ↓i   反行程均值与标准差

* **反向差值** ``B_i = |x̄↑i - x̄↓i|``，``B = max(B_i)``
* **双向定位精度** ``A = max(x̄↑i + 2σ↑i, x̄↓i + 2σ↓i) - min(x̄↑i - 2σ↑i, x̄↓i - 2σ↓i)``
* **单向重复定位精度** ``R↑ = 4σ↑``，``R↓ = 4σ↓``
* **双向重复定位精度** 取保守形式 ``R = max(R↑, R↓, B + 4σ)``

ISO 230-4 循圆测试（球杆仪）
---------------------------
沿名义圆走一圈，记录半径方向的偏差::

    r_i = |P_i - C|

* **半径偏差** ``mean(r) - R_nominal``
* **圆度偏差（circular deviation）** ``max(r) - min(r)``
* **椭圆度** —— 相互垂直的两个方向直径之差的最大值，典型成因是两个直线轴的**伺服增益不匹配**，
  表现为与坐标轴成 45° 的椭圆
* **反向跃冲** —— 过象限处的半径突跳，典型成因是**反向间隙（背隙）**
"""
import math

# ---------------------------------------------------------------- ISO 230-2


def _mean(xs):
    return sum(xs) / float(len(xs))


def _stdev(xs):
    """样本标准差（n-1）。测量数据是抽样，用样本标准差而不是总体标准差。"""
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / float(n - 1))


def positioning_accuracy(up_runs, dn_runs):
    """ISO 230-2 定位精度与重复定位精度。

    ``up_runs`` / ``dn_runs``：二维序列，``[目标位置][第几次测量]``，单位 mm，
    数值是**偏差量**（实测值 − 目标值），不是绝对位置。
    """
    if len(up_runs) != len(dn_runs):
        raise ValueError('正反行程的目标位置个数必须一致')
    if not up_runs:
        raise ValueError('至少要有一个目标位置')

    n_pos = len(up_runs)
    rows = []
    for i in range(n_pos):
        u, d = list(up_runs[i]), list(dn_runs[i])
        if not u or not d:
            raise ValueError('第 %d 个目标位置缺少测量数据' % (i + 1))
        mu, md = _mean(u), _mean(d)
        su, sd = _stdev(u), _stdev(d)
        rows.append({
            'index': i + 1,
            'mean_up': mu, 'mean_dn': md,
            'std_up': su, 'std_dn': sd,
            'reversal': abs(mu - md),
            'bi_mean': (mu + md) / 2.0,
        })

    hi = max(max(r['mean_up'] + 2 * r['std_up'], r['mean_dn'] + 2 * r['std_dn']) for r in rows)
    lo = min(min(r['mean_up'] - 2 * r['std_up'], r['mean_dn'] - 2 * r['std_dn']) for r in rows)

    sigma_max = max(max(r['std_up'], r['std_dn']) for r in rows)
    r_up = 4.0 * max(r['std_up'] for r in rows)
    r_dn = 4.0 * max(r['std_dn'] for r in rows)
    b = max(r['reversal'] for r in rows)

    return {
        'positions': n_pos,
        'bidirectional_accuracy': hi - lo,
        'reversal_difference': b,
        'repeatability_up': r_up,
        'repeatability_dn': r_dn,
        'repeatability_bidirectional': max(r_up, r_dn, b + 4.0 * sigma_max),
        'max_std': sigma_max,
        'rows': rows,
    }


# ---------------------------------------------------------------- ISO 230-4


def _harmonics(rs, angs, max_order=2):
    """半径偏差的谐波分解（最小二乘意义下的傅里叶系数）。

    ``r(θ) ≈ A0 + Σ [a_k·cos(kθ) + b_k·sin(kθ)]``

    球杆仪数据的标准拆法就是看各阶谐波 —— **不同阶对应不同故障源**：

    =====  ==================  ==============================
    阶次   形态                典型成因
    =====  ==================  ==============================
    1θ     "香蕉形" / 偏心     两轴**垂直度**误差、工件偏心
    2θ     **椭圆**            两直线轴**伺服增益不匹配**
    高阶   局部凸起 / 台阶      反向间隙、爬行、导轨局部误差
    =====  ==================  ==============================

    返回 ``{阶次: (a, b, 幅值, 峰峰值)}``，峰峰值 = ``2 * 幅值``。
    """
    n = len(rs)
    out = {}
    for k in range(1, max_order + 1):
        ca = sum(r * math.cos(k * a) for r, a in zip(rs, angs)) * 2.0 / n
        sa = sum(r * math.sin(k * a) for r, a in zip(rs, angs)) * 2.0 / n
        amp = math.hypot(ca, sa)
        out[k] = (ca, sa, amp, 2.0 * amp)
    return out


def circular_test(xs, ys, nominal_r, center=None):
    """ISO 230-4 循圆测试数据分析。

    ``xs`` / ``ys``：沿名义圆采到的点位（mm）。``nominal_r``：名义半径（mm）。

    ⚠️ **``center`` 请尽量传名义圆心。** 默认取采样点形心，但形心会把 **1θ 分量吸收掉** ——
    一个纯 1θ（limaçon / 香蕉形）的形心恰好偏了 ``c/2``，从形心去量半径，1θ 就没了。
    真实球杆仪用的是**机床编程的名义圆心**，所以只要知道圆心就应当传进来。
    本项目在测试里对同一组数据分别传名义圆心与形心，把这件事显式测出来。

    除了 ISO 230-4 的**圆度偏差**（``max(r) - min(r)``）与**半径偏差**
    （``mean(r) - R``）之外，这里做三件事：

    1. **先量反向间隙**（过象限的半径台阶）并**从数据里扣掉**——
       台阶是分段常数，其傅里叶基波正好落在 **2θ** 上（周期 180° 的方波），
       不扣掉的话 20 μm 的间隙会被误报成 25 μm 的椭圆度；
    2. 在扣掉台阶的数据上做**谐波分解**：1θ（垂直度 / 偏心）、2θ（伺服增益不匹配的椭圆）；
    3. 给出**高阶残差圆度**（减掉 1θ、2θ 后剩下的起伏，对应爬行、导轨局部误差等）。

    不拆开的话，总圆度只能告诉你"差"，告诉不了你"哪儿差"。
    """
    if len(xs) != len(ys):
        raise ValueError('x 与 y 的点数必须一致')
    if len(xs) < 36:
        raise ValueError('循圆测试至少要 36 个点，否则象限特征分辨不出来')

    if center is None:
        cx, cy = _mean(xs), _mean(ys)
    else:
        cx, cy = center

    rs, angs = [], []
    for x, y in zip(xs, ys):
        dx, dy = x - cx, y - cy
        rs.append(math.hypot(dx, dy))
        angs.append(math.atan2(dy, dx) % (2.0 * math.pi))

    circ_dev = max(rs) - min(rs)
    radius_dev = _mean(rs) - nominal_r
    mean_r = _mean(rs)

    # ---- 第一步：先量反向间隙（台阶），再从数据里扣掉，否则它会被算进 2θ 椭圆度 ----
    # 一圈有 4 个象限台阶，台阶是**分段常数**，而它的傅里叶基波正好落在 2θ 上
    # （周期 180° 的方波）。不扣掉的话，20 μm 的间隙会被误报成 ~25 μm 的椭圆度。
    quad_off, step_sizes = _quadrant_steps(rs, angs, mean_r)
    backlash = _mean(step_sizes) if step_sizes else 0.0

    rs_ds = []
    for r, a in zip(rs, angs):
        q = int((a % (2.0 * math.pi)) // (0.5 * math.pi)) % 4
        rs_ds.append(r - quad_off[q])

    # ---- 第二步：在扣掉台阶的数据上做谐波分解 ----
    har = _harmonics(rs_ds, angs, max_order=2)

    resid = []
    for r, a in zip(rs_ds, angs):
        fit = 0.0
        for k, (ca, sa, _amp, _pp) in har.items():
            fit += ca * math.cos(k * a) + sa * math.sin(k * a)
        resid.append(r - fit)
    residual_roundness = max(resid) - min(resid)

    return {
        'points': len(rs),
        'radius_deviation': radius_dev,
        'circular_deviation': circ_dev,
        'max_radius': max(rs),
        'min_radius': min(rs),
        'mean_radius': mean_r,
        'std_radius': _stdev(rs),
        'backlash': backlash,
        'quadrant_offsets': quad_off,
        'ovality': har[2][3],                 # 2θ 峰峰值 —— 伺服增益不匹配
        'ovality_phase_deg': math.degrees(math.atan2(har[2][1], har[2][0])) / 2.0,
        'squareness': har[1][3],              # 1θ 峰峰值 —— 垂直度 / 偏心
        'squareness_phase_deg': math.degrees(math.atan2(har[1][1], har[1][0])),
        'residual_roundness': residual_roundness,
        'harmonics': {k: v[3] for k, v in har.items()},
    }


def _quadrant_steps(rs, angs, mean_r, window_deg=3.0):
    """测过象限的**半径台阶**，返回 ``(每象限偏置, 台阶高度列表)``。

    ⚠️ **关键在"窄窗"而不是"象限均值"。** 最早我用每个象限内部的半径均值之差当台阶，
    结果把 **1θ（垂直度）分量一起扣掉了** —— 因为 1θ 在各象限的均值本来就不同
    （``cos θ`` 在四个象限的均值是 ``+0.64, -0.64, -0.64, +0.64``）。
    真实的反向间隙是一条**不连续**的台阶，而 1θ / 2θ 是**连续**的，
    所以只在边界前后各取一个窄窗（默认 ±3°）求差，就能把两者分开：

    * 纯台阶：窗口差 ≈ 台阶高度；
    * 纯 1θ：``cos θ`` 在 ±3° 的窗口差只有 ``2*sin(3°)*c ≈ 0.05c``，几乎测不到。
    """
    steps = []
    w = math.radians(window_deg)
    for q in range(4):
        qa = q * 0.5 * math.pi
        before = [r for r, a in zip(rs, angs) if _in_window(a, qa - w, qa)]
        after = [r for r, a in zip(rs, angs) if _in_window(a, qa, qa + w)]
        if len(before) >= 3 and len(after) >= 3:
            steps.append(_mean(after) - _mean(before))
        else:
            steps.append(0.0)

    # 由台阶累积出每象限的偏置，再整体去均值（台阶只关心相对量）。
    # ⚠️ 索引要对：``steps[q]`` 是**边界 q×90° 处**的突跳，即第 q-1 象限到第 q 象限的落差
    # （q=0 的边界在 0°，是 q3 → q0）。所以从 q0 出发累加时要用 steps[1]、steps[2]、steps[3]。
    # 早先写成 steps[:3] 是**错的** —— 方向反了，扣完之后方波不减反增（20 μm 间隙被算成 51 μm 椭圆）。
    off = [0.0]
    for s in steps[1:4]:
        off.append(off[-1] + s)
    mu = _mean(off)
    off = [v - mu for v in off]
    return off, [abs(s) for s in steps]


def _in_window(a, lo, hi):
    """角度 ``a``（已归一化到 [0, 2π)）是否落在窗口 ``[lo, hi)``（可跨 0）。"""
    two_pi = 2.0 * math.pi
    a = a % two_pi
    lo %= two_pi
    hi %= two_pi
    if lo <= hi:
        return lo <= a < hi
    return a >= lo or a < hi


def simulate_circular_test(backlash_um=0.0, ovality_um=0.0, squareness_um=0.0,
                           noise_um=0.0, nominal_r=100.0, points=720, seed=20261002,
                           phase_deg=0.0):
    """**带已知误差的循圆测试仿真** —— 用来验证 :func:`circular_test` 能不能把它们反推出来。

    注入三种典型误差，都是球杆仪实际会看到的形态：

    * ``backlash_um``：**反向间隙**。过象限时半径出现阶跃突跳（在 0/90/180/270° 处各一次）。
    * ``ovality_um``：**椭圆度**（典型成因是两个直线轴伺服增益不匹配），
      在 45°/135° 方向拉长、在 0°/90° 方向压扁。
    * ``squareness_um``：**垂直度误差**，表现为与坐标轴成 45° 的"香蕉形"或双叶形，
      这里用 ``sin(2θ)`` 与 ``cos(2θ)`` 的组合建模。
    * ``noise_um``：测量噪声（伪随机，用固定种子保证可复现 —— 不用 ``random`` 是为了跨版本稳定）。

    返回 ``(xs, ys, truth)``，``truth`` 是注入的真值，便于断言。
    """
    if points < 36:
        raise ValueError('至少 36 个点')
    bak, ova, sq, nz = backlash_um / 1000.0, ovality_um / 1000.0, squareness_um / 1000.0, noise_um / 1000.0

    xs, ys = [], []
    for i in range(points):
        ang = 2.0 * math.pi * i / points + math.radians(phase_deg)
        deg = math.degrees(ang) % 360.0

        r = nominal_r

        # 反向间隙：每过一个象限，半径出现一次**等量阶跃**（方向交替）
        # → 一圈形成 4 个台阶，台阶高度就是间隙值，这也是球杆仪上最典型的背隙形态
        if bak:
            r += bak * (0.5 if int(deg // 90.0) % 2 == 0 else -0.5)

        # 椭圆度（伺服增益不匹配）：2θ 分量，峰峰值 = ovality_um
        if ova:
            r += ova * 0.5 * math.cos(2.0 * ang)

        # 垂直度 / 偏心：1θ 分量（香蕉形），峰峰值 = squareness_um
        if sq:
            r += sq * 0.5 * math.cos(ang)

        if nz:
            r += nz * (_pseudo(i) - 0.5)

        xs.append(r * math.cos(ang))
        ys.append(r * math.sin(ang))

    truth = {
        'backlash_um': backlash_um,
        'ovality_um': ovality_um,
        'squareness_um': squareness_um,
        'noise_um': noise_um,
        'nominal_r': nominal_r,
        'points': points,
    }
    return xs, ys, truth


def _pseudo(i):
    """确定性伪随机数，返回 [0, 1)。不用 ``random`` 是为了跨 Python 版本可复现。"""
    x = (i * 1103515245 + 12345) & 0x7FFFFFFF
    x ^= (x >> 13)
    x = (x * 1274126177) & 0x7FFFFFFF
    return ((x >> 8) & 0xFFFF) / 65536.0
