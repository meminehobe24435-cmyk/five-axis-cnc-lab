# -*- coding: utf-8 -*-
"""five-axis-cnc-lab —— 五轴（AC 双摆头）运动学、插补误差与机床精度检测。

模块划分::

    kinematics  正解 / 逆解 / RTCP / 奇异点
    gcode       G 代码解析（够用于五轴应用测试，不是完整数控系统前端）
    interp      关节空间线性插补的非线性误差、弦高误差、圆弧分段
    metrology   ISO 230-2 定位精度、ISO 230-4 循圆测试（含可反推已知误差的仿真器）
    stest       S 形试件刀位生成、误差评估、按容差反推工艺规范（允许步距）

纯 Python 标准库，零第三方依赖。
"""

__version__ = '1.0.0'

from . import gcode, interp, kinematics, metrology, stest  # noqa: F401

__all__ = ['kinematics', 'gcode', 'interp', 'metrology', 'stest']
