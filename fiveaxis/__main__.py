# -*- coding: utf-8 -*-
"""命令行演示：``python -m fiveaxis [kin|gcode|stest|metro|all]``。

所有打印出来的数字都是**当场算的**，README 里的实测结果就是从这些输出抄下来的。
"""
import math
import sys

from . import gcode as gc
from . import interp, kinematics as kin, metrology as met, stest

BAR = '=' * 74


def _h(title):
    print()
    print(BAR)
    print(title)
    print(BAR)


def demo_kin():
    """运动学自检：正逆解往返、RTCP 偏差随摆角增长、奇异点行为。"""
    _h('1) 运动学：正逆解往返 / RTCP 偏差 / 奇异点')
    L = 80.0

    print('   刀具长度 L = %.1f mm，A 轴行程 ±120°' % L)
    print()
    print('   摆角 A      刀轴矢量 k                          逆解回代误差')
    print('   ' + '-' * 70)
    worst = 0.0
    for a_deg in (0.0, 15.0, 30.0, 45.0, 60.0, 90.0):
        a = math.radians(a_deg)
        for c_deg in (0.0, 37.0, 90.0):
            c = math.radians(c_deg)
            k = kin.forward_a(a, c)
            try:
                a2, c2 = kin.inverse_a(k)
            except kin.SingularPointError:
                continue
            q = kin.inverse((10.0, 20.0, -5.0), k, L)
            p2, k2 = kin.forward(q, L)
            err = math.sqrt(sum((p2[i] - (10.0, 20.0, -5.0)[i]) ** 2 for i in range(3)))
            worst = max(worst, err)
            if c_deg == 0.0:
                print('   %5.1f°      (%+.4f, %+.4f, %+.4f)      %.3e mm'
                      % (a_deg, k[0], k[1], k[2], err))
    print()
    print('   正解→逆解→正解 的最大往返误差：%.3e mm' % worst)

    print()
    print('   不启用 RTCP 时，摆头旋转未被补偿造成的刀尖偏差：')
    print('   摆角 θ      理论 2L·sin(θ/2)      实测         占刀具长度')
    print('   ' + '-' * 70)
    for deg in (10.0, 30.0, 45.0, 60.0, 90.0):
        a = math.radians(deg)
        k = kin.forward_a(a, 0.0)
        err, _ = kin.rtcp_error((0, 0, 0), k, L)
        theory = 2.0 * L * math.sin(a / 2.0)
        print('   %5.1f°      %8.3f mm          %8.3f mm     %5.1f%%'
              % (deg, theory, err, err / L * 100.0))
    print()
    print('   → 摆 90° 时偏差 113.1 mm，是刀具长度的 1.414 倍；这就是五轴必须开 RTCP 的原因。')

    print()
    print('   奇异点（刀轴平行于 Z 轴）：')
    for k in ((0.0, 0.0, 1.0), (1e-9, 0.0, 1.0)):
        try:
            kin.inverse_a(k)
            print('   k = %-24s → 未判为奇异（不应发生）' % (str(k),))
        except kin.SingularPointError as e:
            print('   k = %-24s → 正确抛出奇异点：%s' % (str(k), str(e)[:40]))
    k_s = (0.0, 0.0, 1.0)
    q = kin.inverse((5.0, 5.0, 0.0), k_s, L, keep_c=math.radians(30.0))
    print('   冻结 C=30° 后可继续求解：C = %.1f°' % math.degrees(q[4]))


def demo_gcode():
    """G 代码解析演示。"""
    _h('2) G 代码解析')
    src = """
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
    prog = gc.parse(src)
    st = gc.stats(prog)
    print('   程序段 %d，运动段 %d（快速 %d / 直线 %d / 顺弧 %d / 逆弧 %d）'
          % (st['blocks'], st['moves'], st['rapid'], st['linear'], st['arc_cw'], st['arc_ccw']))
    print('   单位 %s，插补平面 %s，未实现代码 %d 个' % (st['units'], st['plane'], st['unknown']))
    print()
    print('   逐段解析结果（前 12 段）：')
    print('   行   G        M      地址字')
    print('   ' + '-' * 70)
    for b in prog.blocks[:12]:
        words = {k: v for k, v in b.words.items() if not k.startswith('_')}
        print('   %-4d %-8s %-6s %s' % (b.line, b.g, b.m, words))
    print()
    print('   注释提取：', [b.comment for b in prog.blocks if b.comment])

    # 圆弧几何：起点按程序中实际的当前位置推算，而不是写死
    pos = (0.0, 0.0, 0.0)
    for b in prog.blocks:
        if 2 in b.g or 3 in b.g:
            arc = gc.arc_from_blocks(pos, b)
            print('   第 %2d 行 G%02d：%s 平面，R = %.3f mm，圆心 (%.3f, %.3f, %.3f)，%s %.2f°'
                  % (b.line, 2 if arc.cw else 3, arc.plane, arc.radius,
                     arc.center[0], arc.center[1], arc.center[2],
                     '顺时针' if arc.cw else '逆时针', abs(arc.angle_deg)))
        for letter, i in (('X', 0), ('Y', 1), ('Z', 2)):
            if letter in b.words:
                pos = tuple(b.words[letter] if j == i else pos[j] for j in range(3))
    print()
    n = interp.arc_segments(15.0, 0.002)
    print('   弦高误差：R=15 mm、容差 2 μm 时整圆需分 %d 段，实际最大弦高 %.4f mm'
          % (n, interp.arc_max_chord_error(15.0, n)))


def demo_stest():
    """S 形试件：刀路评估 + 工艺规范反推。"""
    _h('3) S 形试件：误差评估与工艺规范')
    L = 80.0
    path = stest.toolpath(60)
    res = stest.evaluate(path, L, samples=60)
    print('   试件 L = %.0f mm，S 幅值 %.0f mm，刀轴倾角 %.0f° ± %.0f°（不穿越奇异点），刀具长度 %.0f mm'
          % (stest.DEFAULT_LENGTH, stest.DEFAULT_AMPLITUDE,
             stest.DEFAULT_TILT_MID, stest.DEFAULT_TILT_DEG, L))
    print('   段数 %d（奇异段 %d）' % (res['segments_total'], res['singular_segments']))
    print('   最大摆角 %.2f°，最大步距 %.3f mm' % (res['max_tilt_deg'], res['max_step_mm']))
    print('   非线性误差：max %.5f mm，mean %.5f mm'
          % (res['max_nonlinearity_mm'], res['mean_nonlinearity_mm']))
    print('   无 RTCP 时最大刀尖偏差 %.3f mm' % res['max_rtcp_error_mm'])
    print()
    print('   误差最大的 5 段：')
    print('   段号   u      步距(mm)  摆角(°)   非线性误差(mm)')
    print('   ' + '-' * 62)
    top = sorted([s for s in res['segments'] if s['nonlinearity_mm'] is not None],
                 key=lambda s: -s['nonlinearity_mm'])[:5]
    for s in top:
        print('   %-5d %6.3f  %8.4f  %7.2f   %10.5f'
              % (s['index'], s['u'], s['step_mm'], s['tilt_deg'], s['nonlinearity_mm']))

    print()
    print('   工艺规范：给定非线性误差容差，允许的最大刀位步距')
    print('   刀具长度  摆角    容差 5μm    容差 10μm   容差 20μm')
    print('   ' + '-' * 62)
    for L2 in (40.0, 80.0, 120.0):
        for tdeg in (20.0, 35.0, 50.0):
            cells = []
            for tol in (0.005, 0.010, 0.020):
                step, _ = stest.max_step_for_tolerance(tol, L2, tilt_deg=tdeg)
                cells.append('%8.3f' % step if step else '  不可达')
            print('   %6.0f mm  %4.0f°  %s' % (L2, tdeg, '  '.join(cells)))
    print()
    print('   → 刀越长、摆角越大，允许步距越小；这张表就是工艺卡上"步距给多少"的依据。')


def demo_metro():
    """精度检测：ISO 230-2 定位精度 + ISO 230-4 循圆测试（含仿真反推验证）。"""
    _h('4) 精度检测：ISO 230-2 / ISO 230-4')

    print('   4.1 定位精度（ISO 230-2）—— 构造一组带反向间隙与随机波动的测量数据')
    import math as _m

    up, dn = [], []
    targets = [0.0, 50.0, 100.0, 150.0, 200.0]
    for i, _t in enumerate(targets):
        bias = 0.004 * i                       # 系统性螺距误差
        up.append([bias + 0.002 * _m.sin(j) for j in range(5)])
        dn.append([bias - 0.006 + 0.002 * _m.cos(j) for j in range(5)])
    res = met.positioning_accuracy(up, dn)
    print('   目标位置数 %d，每向 5 次测量' % res['positions'])
    print('   双向定位精度 A      = %.5f mm' % res['bidirectional_accuracy'])
    print('   反向差值     B      = %.5f mm' % res['reversal_difference'])
    print('   单向重复定位 R↑/R↓  = %.5f / %.5f mm'
          % (res['repeatability_up'], res['repeatability_dn']))
    print('   双向重复定位 R      = %.5f mm' % res['repeatability_bidirectional'])

    print()
    print('   4.2 循圆测试（ISO 230-4）—— 注入已知误差，看能不能反推出来')
    print('   %-26s %10s %12s %12s %12s' % ('注入真值', '圆度(μm)', '间隙(μm)', '椭圆(μm)', '垂直(μm)'))
    print('   ' + '-' * 76)
    cases = [
        ('反向间隙 20 μm', dict(backlash_um=20.0, nominal_r=100.0)),
        ('椭圆度 15 μm', dict(ovality_um=15.0, nominal_r=100.0)),
        ('垂直度 12 μm', dict(squareness_um=12.0, nominal_r=100.0)),
        ('间隙 20 + 椭圆 15 + 噪声 3 μm',
         dict(backlash_um=20.0, ovality_um=15.0, noise_um=3.0, nominal_r=100.0)),
    ]
    for label, kw in cases:
        xs, ys, truth = met.simulate_circular_test(points=720, **kw)
        # 传名义圆心 (0,0)，而不是让函数取形心 —— 形心会把 1θ（垂直度）分量吸收掉
        r = met.circular_test(xs, ys, kw.get('nominal_r', 100.0), center=(0.0, 0.0))
        print('   %-26s %10.2f %12.2f %12.2f %12.2f'
              % (label, r['circular_deviation'] * 1000.0, r['backlash'] * 1000.0,
                 r['ovality'] * 1000.0, r['squareness'] * 1000.0))
    print()

    # 用形心会怎样？拿垂直度那一组做对比
    xs, ys, _ = met.simulate_circular_test(squareness_um=12.0, nominal_r=100.0, points=720)
    r_nom = met.circular_test(xs, ys, 100.0, center=(0.0, 0.0))
    r_cen = met.circular_test(xs, ys, 100.0)
    print('   同一组「垂直度 12 μm」数据，圆心取法不同结果差多少：')
    print('   名义圆心 (0,0) → 1θ = %.2f μm    |    数据形心 → 1θ = %.2f μm'
          % (r_nom['squareness'] * 1000.0, r_cen['squareness'] * 1000.0))
    print('   → 形心会把 1θ 分量吸收掉。真实球杆仪用机床编程的名义圆心，就是这个原因。')
    print()
    print('   → 谐波分解把总圆度拆成 1θ（垂直度/偏心）、2θ（伺服不匹配的椭圆）与高阶残差，')
    print('     过象限突跳单独测反向间隙。不拆开的话，总圆度只能告诉你"差"，告诉不了你"哪儿差"。')


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    what = argv[0] if argv else 'all'
    table = {'kin': demo_kin, 'gcode': demo_gcode, 'stest': demo_stest,
             'metro': demo_metro}
    if what == 'all':
        for fn in (demo_kin, demo_gcode, demo_stest, demo_metro):
            fn()
    elif what in table:
        table[what]()
    else:
        print(__doc__)
        print('可用：kin | gcode | stest | metro | all')
        return 2
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
