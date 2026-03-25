# import numpy as np
# import matplotlib.pyplot as plt

# # 设置 P 的取值范围
# P = np.linspace(0, 300, 500)   # 可按需要修改范围，例如 np.linspace(0, 200, 1000)

# # 方案A
# L1 = 9.4191e-05 * P**2 + 0.00271 * P + 1.8309   # BING
# L2 = 1.0825e-04 * P**2 - 0.01526 * P + 2.4127   # DING
# L3 = 9.5455e-05 * P**2 + 0.01739 * P + 3.2400   # WU

# # 方案B
# L4 = 8.7839e-05 * P**2 + 0.00294 * P + 1.8123   # BING
# L5 = 1.0634e-04 * P**2 - 0.01545 * P + 2.4406   # DING
# L6 = 8.4424e-05 * P**2 + 0.01733 * P + 3.1571   # WU

# plt.figure(figsize=(8, 5))
# plt.plot(P, L1, label='L1')
# plt.plot(P, L2, label='L2')
# plt.plot(P, L3, label='L3')
# plt.plot(P, L4, label='L4', linestyle='--')
# plt.plot(P, L5, label='L5', linestyle='--')
# plt.plot(P, L6, label='L6', linestyle='--')

# plt.xlabel('P')
# plt.ylabel('L̂')
# plt.title('三条曲线在统一坐标系中的绘制')
# plt.legend()
# plt.grid(True)
# plt.tight_layout()
# plt.show()

import os
import numpy as np
import matplotlib.pyplot as plt

# ── 输出路径配置（相对于项目根目录）──────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR   = os.path.join(_SCRIPT_DIR, "..")
_OUT_DIR    = os.path.join(_ROOT_DIR, "DATA", "峡阳B", "analysis_output")
os.makedirs(_OUT_DIR, exist_ok=True)
_OUT_PNG = os.path.join(_OUT_DIR, "绘制拟合曲线.png")

# 设置基础功率范围
P = np.linspace(0, 300, 500)

# 方案A
L1 = 9.4191e-05 * P**2 + 0.00271 * P + 1.8309   # BING
L2 = 1.0825e-04 * P**2 - 0.01526 * P + 2.4127   # DING
L3 = 9.5455e-05 * P**2 + 0.01739 * P + 3.2400   # WU

# 方案B
L4 = 8.7839e-05 * P**2 + 0.00294 * P + 1.8123   # BING
L5 = 1.0634e-04 * P**2 - 0.01545 * P + 2.4406   # DING
L6 = 8.4424e-05 * P**2 + 0.01733 * P + 3.1571   # WU

# 方案A的新横轴：集电线路功率 + 损耗
X1 = P + L1
X2 = P + L2
X3 = P + L3

plt.figure(figsize=(9, 6))

# 方案A：横轴改为 P + L
plt.plot(X1, L1, label='方案A-BING (横轴=P+L)')
plt.plot(X2, L2, label='方案A-DING (横轴=P+L)')
plt.plot(X3, L3, label='方案A-WU (横轴=P+L)')

# 方案B：横轴保持为 P
plt.plot(P, L4, '--', label='方案B-BING (横轴=P)')
plt.plot(P, L5, '--', label='方案B-DING (横轴=P)')
plt.plot(P, L6, '--', label='方案B-WU (横轴=P)')

plt.xlabel('功率横轴（方案A: P+L；方案B: P）')
plt.ylabel(r'损耗 $\hat{L}$')
plt.title('方案A与方案B拟合曲线对比')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(_OUT_PNG, dpi=150, bbox_inches="tight")
print(f"✅ 图表已保存：{_OUT_PNG}")
# plt.show() 在无头（服务器）环境下会报错，若在本地运行可取消注释：
# plt.show()