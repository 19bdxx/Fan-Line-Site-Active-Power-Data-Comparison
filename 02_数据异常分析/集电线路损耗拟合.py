"""
集电线路传输损耗拟合工具（独立脚本）
=====================================
功能：
    从 SCADA 合并 CSV 数据和已生成的异常段分类结果（fan_repeat_classified.csv），
    分别用两种方案拟合丙/丁/戊三条集电线路的传输损耗模型：

    方案 A（CT 作为 P，含截距）
        P = max(CT, 0)，即集电线路有功功率的有效值（"输出侧"）
        L̂(P) = a·P² + b·P + c  （np.polyfit 含截距二次多项式）
        当 P=0 时 L̂(0)=c≠0，存在常数铁损项

    方案 B（FAN_SUM_S2 作为 P，过原点）
        P = FAN_SUM_S2，即集电线路所属风机有功功率和（已 ≥ 0，"输入侧"）
        L̂(P) = a·P² + b·P  （lstsq 无截距，强制 L̂(0)=0）
        物理含义：P=0 时无电流，损耗自然为 0
        适合功率预测：CT_pred = FAN_SUM_S2 − L̂(FAN_SUM_S2)

变量定义：
    FAN_SUM_S2 = {BING,DING,WU}_ACTIVE_POWER_SUM_S2
        各风机正值保留、负值取 0 后的汇总（已 ≥ 0）
    CT = ACTIVE_POWER_{BING,DING,WU}
        集电线路有功测量值，可能为负（反向潮流/厂用电）
    实际线损 L = FAN_SUM_S2 − max(CT, 0)

输入文件（自动搜索，相对于本脚本的父目录）：
    DATA/峡阳B/*with_sum*.csv        SCADA 合并数据（含 FAN_SUM_S2 列）
    分析结果/数据质量分析/峡阳B/fan_repeat_classified.csv
        或
    分析结果/风机重复检测/峡阳B/联合重复值检测结果.xlsx
        异常段分类结果（用于排除拟合时的异常时段）

输出：
    DATA/峡阳B/analysis_output/loss_fit_comparison.png
        三线两方案拟合对比图（散点 + 两条拟合曲线 + R²/σ 标注）
    分析结果/数据质量分析/峡阳B/loss_fit_comparison.csv
        两种方案对比结果表（系数、R²、σ、各功率点预测值）
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════
# 0. 字体 & 路径配置
# ══════════════════════════════════════════════════════════════
_fm._load_fontmanager(try_read_cache=False)
_CJK_FONT = next(
    (f.name for f in _fm.fontManager.ttflist
     if any(k in f.name for k in (
         'Noto Sans CJK', 'Noto Serif CJK', 'WenQuanYi', 'SimHei', 'Microsoft YaHei'))),
    None
)
if _CJK_FONT:
    matplotlib.rcParams['font.family']      = _CJK_FONT
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    matplotlib.rcParams['axes.unicode_minus'] = False
    print("⚠️  未找到中文字体，图表中文字符可能显示为方块。")

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.join(_SCRIPT_DIR, "..")
SCADA_DIR    = os.path.join(_ROOT_DIR, "DATA",     "峡阳B")
OUT_FIG_DIR  = os.path.join(_ROOT_DIR, "DATA",     "峡阳B", "analysis_output")
OUT_DATA_DIR = os.path.join(_ROOT_DIR, "分析结果", "数据质量分析", "峡阳B")

os.makedirs(OUT_FIG_DIR,  exist_ok=True)
os.makedirs(OUT_DATA_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════
# 1. 集电线路列映射
# ══════════════════════════════════════════════════════════════
LINE_CT_COL  = {
    'BING': 'ACTIVE_POWER_BING',
    'DING': 'ACTIVE_POWER_DING',
    'WU':   'ACTIVE_POWER_WU',
}
LINE_FAN_COL = {
    'BING': 'BING_ACTIVE_POWER_SUM_S2',
    'DING': 'DING_ACTIVE_POWER_SUM_S2',
    'WU':   'WU_ACTIVE_POWER_SUM_S2',
}
LINE_NAME_ZH = {'BING': '丙（BING）', 'DING': '丁（DING）', 'WU': '戊（WU）'}
LINE_COLOR   = {'BING': '#ff7f0e',   'DING': '#2ca02c',     'WU': '#1f77b4'}

SCADA_COLS = (
    ['timestamp']
    + list(LINE_CT_COL.values())
    + list(LINE_FAN_COL.values())
)

N_BINS = 15   # 等频分位数箱数

# ══════════════════════════════════════════════════════════════
# 2. 加载 SCADA 数据
# ══════════════════════════════════════════════════════════════
print("=" * 65)
print("  加载 SCADA 合并数据 ...")
print("=" * 65)

_csv_files = sorted(glob.glob(os.path.join(SCADA_DIR, "*with_sum*.csv")))
if not _csv_files:
    raise FileNotFoundError(
        f"未找到 SCADA 合并 CSV 文件（*with_sum*.csv）\n"
        f"搜索路径：{SCADA_DIR}"
    )

_parts = []
for f in _csv_files:
    try:
        _df = pd.read_csv(f, usecols=SCADA_COLS, parse_dates=['timestamp'])
        _parts.append(_df)
        print(f"  ✅ {os.path.basename(f)}  ({len(_df):,} 条)")
    except Exception as e:
        print(f"  ⚠️ 跳过 {os.path.basename(f)}: {e}")

scada = pd.concat(_parts, ignore_index=True)
scada.sort_values('timestamp', inplace=True)
scada.drop_duplicates(subset='timestamp', inplace=True)
scada.reset_index(drop=True, inplace=True)
scada['timestamp'] = pd.to_datetime(scada['timestamp'])
print(f"\n  合并后: {len(scada):,} 条  "
      f"({scada['timestamp'].min().date()} ~ {scada['timestamp'].max().date()})")

# ══════════════════════════════════════════════════════════════
# 3. 加载异常段分类结果（用于排除拟合时段）
# ══════════════════════════════════════════════════════════════
print("\n  加载异常段分类结果 ...")

# 优先读已分类 CSV；若不存在则退化为读原始 Excel
_classified_csv = os.path.join(OUT_DATA_DIR, 'fan_repeat_classified.csv')
_raw_excel      = os.path.join(_ROOT_DIR, "分析结果", "风机重复检测",
                               "峡阳B", "联合重复值检测结果.xlsx")

if os.path.exists(_classified_csv):
    anom_df = pd.read_csv(_classified_csv, encoding='utf-8-sig')
    anom_df.rename(columns={'异常类型': 'anomaly_type'}, inplace=True)
    print(f"  ✅ 已读取 fan_repeat_classified.csv（{len(anom_df):,} 条）")
elif os.path.exists(_raw_excel):
    # 基础加载：读 Excel 并赋一个全包含的 anomaly_type（把所有段都排除）
    anom_df = pd.read_excel(_raw_excel, engine='openpyxl')
    anom_df['anomaly_type'] = '第三类-单机非零卡值'   # 当作全排除，保守处理
    print(f"  ✅ 已读取 联合重复值检测结果.xlsx（{len(anom_df):,} 条）"
          f"（无 classified.csv，保守排除全部异常段）")
else:
    print("  ⚠️ 未找到异常段分类文件，将不排除异常段（拟合结果可能偏差较大）")
    anom_df = pd.DataFrame(columns=['开始时间', '结束时间'])

anom_df['开始时间'] = pd.to_datetime(anom_df['开始时间'])
anom_df['结束时间'] = pd.to_datetime(anom_df['结束时间'])

# 需要排除的异常类型（全部有效异常段）
EXCLUDE_TYPES = {
    '第一类-全场通讯中断', '第二类-部分通讯中断',
    '第三类-单机通讯故障', '第三类-发电状态零值',
    '第三类-单机非零卡值', '正常停机-保留', '零值-状态待核实',
}

def _get_line(fan_num):
    if 63  <= fan_num <= 109: return 'WU'
    if 110 <= fan_num <= 152: return 'DING'
    if 153 <= fan_num <= 199: return 'BING'
    return '未知'

if '风机编号' in anom_df.columns:
    anom_df['line'] = anom_df['风机编号'].apply(_get_line)

# ══════════════════════════════════════════════════════════════
# 4. 辅助：向量化区间掩码
# ══════════════════════════════════════════════════════════════
def _build_exclude_mask(scada_ts: np.ndarray, segs_df: pd.DataFrame) -> np.ndarray:
    """返回 bool 数组：True 表示该时间戳属于某个异常段。"""
    mask = np.zeros(len(scada_ts), dtype=bool)
    if len(segs_df) == 0:
        return mask
    ts_np  = scada_ts.astype('datetime64[ns]')
    starts = segs_df['开始时间'].values.astype('datetime64[ns]')
    ends   = segs_df['结束时间'].values.astype('datetime64[ns]')
    for t0, t1 in zip(starts, ends):
        mask |= (ts_np >= t0) & (ts_np <= t1)
    return mask

# ══════════════════════════════════════════════════════════════
# 5. 核心拟合函数
# ══════════════════════════════════════════════════════════════
def fit_loss_model_ct(p_vals: np.ndarray, l_vals: np.ndarray, n_bins: int = N_BINS):
    """
    方案 A：P = max(CT,0)，含截距二次拟合
    L̂(P) = a·P² + b·P + c

    返回：
        coeffs  [a, b, c]，与 np.polyval 兼容
        bins_df  分箱中位数 DataFrame（P_med, L_med, n）
        r2_bins  R²（对分箱中位数）
        sigma    σ（对 P>2 MW 的原始数据）
    """
    valid = (p_vals > 0) & np.isfinite(p_vals) & np.isfinite(l_vals)
    p = p_vals[valid]
    l = l_vals[valid]

    # 等频分位箱
    _tmp = pd.DataFrame({'_P': p, '_L': l})
    _tmp['_bin'] = pd.qcut(_tmp['_P'], q=n_bins, duplicates='drop')
    bins_df = _tmp.groupby('_bin', observed=True).agg(
        P_med=('_P', 'median'), L_med=('_L', 'median'), n=('_P', 'count')
    ).reset_index(drop=True)

    # np.polyfit（含截距）
    coeffs = np.polyfit(bins_df['P_med'].values, bins_df['L_med'].values, deg=2)

    # R²
    l_hat_bins = np.polyval(coeffs, bins_df['P_med'].values)
    ss_res = np.sum((bins_df['L_med'].values - l_hat_bins) ** 2)
    ss_tot = np.sum((bins_df['L_med'].values - bins_df['L_med'].mean()) ** 2)
    r2_bins = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # σ（P > 2 MW 原始数据）
    mask2 = p > 2
    sigma = float(np.std(l[mask2] - np.polyval(coeffs, p[mask2]))) if mask2.sum() > 0 else float('nan')

    return coeffs, bins_df, r2_bins, sigma


def fit_loss_model_fan(p_vals: np.ndarray, l_vals: np.ndarray, n_bins: int = N_BINS):
    """
    方案 B：P = FAN_SUM_S2，过原点二次拟合
    L̂(P) = a·P² + b·P  （lstsq 无截距，L̂(0)=0）

    返回：
        coeffs  [a, b, 0]，与 np.polyval 兼容
        bins_df  分箱中位数 DataFrame（P_med, L_med, n）
        r2_bins  R²（对分箱中位数）
        sigma    σ（对 P>2 MW 的原始数据）
    """
    valid = (p_vals > 0) & np.isfinite(p_vals) & np.isfinite(l_vals)
    p = p_vals[valid]
    l = l_vals[valid]

    # 等频分位箱
    _tmp = pd.DataFrame({'_P': p, '_L': l})
    _tmp['_bin'] = pd.qcut(_tmp['_P'], q=n_bins, duplicates='drop')
    bins_df = _tmp.groupby('_bin', observed=True).agg(
        P_med=('_P', 'median'), L_med=('_L', 'median'), n=('_P', 'count')
    ).reset_index(drop=True)

    # lstsq 过原点（无截距）
    _X = np.column_stack([bins_df['P_med'].values ** 2, bins_df['P_med'].values])
    _ab, _, _, _ = np.linalg.lstsq(_X, bins_df['L_med'].values, rcond=None)
    coeffs = np.array([_ab[0], _ab[1], 0.0])

    # R²
    l_hat_bins = _X @ _ab
    ss_res = np.sum((bins_df['L_med'].values - l_hat_bins) ** 2)
    ss_tot = np.sum((bins_df['L_med'].values - bins_df['L_med'].mean()) ** 2)
    r2_bins = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')

    # σ（P > 2 MW 原始数据）
    mask2 = p > 2
    sigma = float(np.std(l[mask2] - np.polyval(coeffs, p[mask2]))) if mask2.sum() > 0 else float('nan')

    return coeffs, bins_df, r2_bins, sigma

# ══════════════════════════════════════════════════════════════
# 6. 主循环：逐线路拟合
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  开始拟合（两种方案）")
print("=" * 65)

results = {}   # line → dict

ts_np = scada['timestamp'].values.astype('datetime64[ns]')

for line in ['BING', 'DING', 'WU']:
    ct_col  = LINE_CT_COL[line]
    fan_col = LINE_FAN_COL[line]
    zh_name = LINE_NAME_ZH[line]

    print(f"\n{'─'*60}")
    print(f"  {zh_name}")
    print(f"{'─'*60}")

    # ── 排除本线路异常段 ────────────────────────────────────────
    if 'line' in anom_df.columns:
        _segs = (anom_df[anom_df['line'] == line][['开始时间', '结束时间']]
                 .drop_duplicates())
    else:
        _segs = anom_df[['开始时间', '结束时间']].drop_duplicates()

    _mask_exc = _build_exclude_mask(ts_np, _segs)
    normal = scada[~_mask_exc].copy()
    # 排除 CT 和 FAN_SUM 同时接近 0 的真正停电时段
    normal = normal[
        ~((normal[ct_col].abs() < 0.1) & (normal[fan_col].abs() < 0.1))
    ].copy()

    # ── 计算 P 和 L ──────────────────────────────────────────────
    normal['_CT_eff']  = normal[ct_col].clip(lower=0)          # max(CT, 0)
    normal['_FAN']     = normal[fan_col].clip(lower=0)          # FAN_SUM_S2 (已≥0)
    normal['_L']       = normal['_FAN'] - normal['_CT_eff']     # 实际线损

    P_ct  = normal['_CT_eff'].values
    P_fan = normal['_FAN'].values
    L     = normal['_L'].values

    n_gen_ct  = (P_ct  > 0).sum()
    n_gen_fan = (P_fan > 0).sum()
    print(f"  正常时段数据量：{len(normal):,} 条  "
          f"（CT>0: {n_gen_ct:,}，FAN>0: {n_gen_fan:,}）")

    # ── 方案 A：P = max(CT,0)，含截距 ───────────────────────────
    coeffs_a, bins_a, r2_a, sigma_a = fit_loss_model_ct(P_ct, L)
    sign_b_a = '+' if coeffs_a[1] >= 0 else '-'
    sign_c_a = '+' if coeffs_a[2] >= 0 else '-'
    print(f"\n  方案 A（P=max(CT,0)，含截距）：")
    print(f"    L̂(P) = {coeffs_a[0]:.4e}·P²"
          f" {sign_b_a} {abs(coeffs_a[1]):.5f}·P"
          f" {sign_c_a} {abs(coeffs_a[2]):.4f}")
    print(f"    L̂(0) = {coeffs_a[2]:.4f} MW   "
          f"R²(箱) = {r2_a:.4f}   σ(P>2 MW) = {sigma_a:.4f} MW")

    # ── 方案 B：P = FAN_SUM_S2，过原点 ──────────────────────────
    coeffs_b, bins_b, r2_b, sigma_b = fit_loss_model_fan(P_fan, L)
    sign_b_b = '+' if coeffs_b[1] >= 0 else '-'
    print(f"\n  方案 B（P=FAN_SUM_S2，过原点）：")
    print(f"    L̂(P) = {coeffs_b[0]:.4e}·P²"
          f" {sign_b_b} {abs(coeffs_b[1]):.5f}·P  [L̂(0)=0]")
    print(f"    L̂(0) = 0.0000 MW   "
          f"R²(箱) = {r2_b:.4f}   σ(P>2 MW) = {sigma_b:.4f} MW")

    results[line] = {
        'P_ct': P_ct, 'P_fan': P_fan, 'L': L,
        'bins_a': bins_a, 'bins_b': bins_b,
        'coeffs_a': coeffs_a, 'coeffs_b': coeffs_b,
        'r2_a': r2_a, 'r2_b': r2_b,
        'sigma_a': sigma_a, 'sigma_b': sigma_b,
    }

# ══════════════════════════════════════════════════════════════
# 7. 对比预测值（P = 20/50/100/150/200/250 MW）
# ══════════════════════════════════════════════════════════════
P_REFS = [20, 50, 100, 150, 200, 250]

print("\n" + "=" * 65)
print("  对比预测值（MW）")
print("=" * 65)
print(f"\n{'P':>6s} | " + " | ".join(
    f"{LINE_NAME_ZH[l]:8s}  A    B" for l in ['BING', 'DING', 'WU']
))
print("─" * 80)
for p in P_REFS:
    row = f"{p:>5.0f} | "
    for line in ['BING', 'DING', 'WU']:
        va = np.polyval(results[line]['coeffs_a'], p)
        vb = np.polyval(results[line]['coeffs_b'], p)
        row += f"  {va:5.2f}  {vb:5.2f}  |"
    print(row)

# ══════════════════════════════════════════════════════════════
# 8. 保存对比结果表
# ══════════════════════════════════════════════════════════════
_rows = []
for line in ['BING', 'DING', 'WU']:
    r = results[line]
    ca, cb = r['coeffs_a'], r['coeffs_b']

    base_row = {
        '集电线路': LINE_NAME_ZH[line],
        '方案': None,
        '自变量P': None,
        '系数a': None,
        '系数b': None,
        '系数c': None,
        'L̂(0)[MW]': None,
        'R²(箱中位数)': None,
        'σ_P>2MW[MW]': None,
    }
    for p_ref in P_REFS:
        base_row[f'L̂({p_ref}MW)_A'] = None
        base_row[f'L̂({p_ref}MW)_B'] = None

    row_a = dict(base_row)
    row_a.update({
        '方案': 'A（含截距）',
        '自变量P': 'max(CT,0)',
        '系数a': ca[0], '系数b': ca[1], '系数c': ca[2],
        'L̂(0)[MW]': round(ca[2], 4),
        'R²(箱中位数)': round(r['r2_a'], 4),
        'σ_P>2MW[MW]': round(r['sigma_a'], 4),
    })
    for p_ref in P_REFS:
        row_a[f'L̂({p_ref}MW)_A'] = round(float(np.polyval(ca, p_ref)), 4)

    row_b = dict(base_row)
    row_b.update({
        '方案': 'B（过原点）',
        '自变量P': 'FAN_SUM_S2',
        '系数a': cb[0], '系数b': cb[1], '系数c': 0.0,
        'L̂(0)[MW]': 0.0,
        'R²(箱中位数)': round(r['r2_b'], 4),
        'σ_P>2MW[MW]': round(r['sigma_b'], 4),
    })
    for p_ref in P_REFS:
        row_b[f'L̂({p_ref}MW)_B'] = round(float(np.polyval(cb, p_ref)), 4)

    _rows.extend([row_a, row_b])

_csv_out = os.path.join(OUT_DATA_DIR, 'loss_fit_comparison.csv')
pd.DataFrame(_rows).to_csv(_csv_out, index=False, encoding='utf-8-sig')
print(f"\n  ✅ 对比结果表已保存：{_csv_out}")

# ══════════════════════════════════════════════════════════════
# 9. 可视化：三线 × 两方案对比图
# ══════════════════════════════════════════════════════════════
print("\n  生成对比图 loss_fit_comparison.png ...")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle('集电线路传输损耗拟合对比\n方案A: P=max(CT,0) 含截距  vs  方案B: P=FAN_SUM_S2 过原点',
             fontsize=13, y=1.02)

for ax, line in zip(axes, ['BING', 'DING', 'WU']):
    r    = results[line]
    color = LINE_COLOR[line]

    # ── 原始散点（降采样，最多 5000 点）────────────────────────
    p_ct_all  = r['P_ct']
    p_fan_all = r['P_fan']
    L_all     = r['L']

    valid_a = (p_ct_all  > 0) & np.isfinite(p_ct_all)  & np.isfinite(L_all)
    valid_b = (p_fan_all > 0) & np.isfinite(p_fan_all) & np.isfinite(L_all)

    np.random.seed(42)
    idx_a = np.where(valid_a)[0]
    idx_b = np.where(valid_b)[0]
    if len(idx_a) > 5000:
        idx_a = np.random.choice(idx_a, 5000, replace=False)
    if len(idx_b) > 5000:
        idx_b = np.random.choice(idx_b, 5000, replace=False)

    ax.scatter(p_ct_all[idx_a],  L_all[idx_a],  s=3, alpha=0.12,
               color='steelblue', label='实测点(P=CT)')
    ax.scatter(p_fan_all[idx_b], L_all[idx_b],  s=3, alpha=0.12,
               color='darkorange', label='实测点(P=FAN)')

    # ── 分箱中位数 ──────────────────────────────────────────────
    ax.scatter(r['bins_a']['P_med'], r['bins_a']['L_med'], s=40, zorder=5,
               color='steelblue', marker='o', label='分箱中位数(CT)')
    ax.scatter(r['bins_b']['P_med'], r['bins_b']['L_med'], s=40, zorder=5,
               color='darkorange', marker='s', label='分箱中位数(FAN)')

    # ── 拟合曲线 ────────────────────────────────────────────────
    p_max = max(np.nanmax(p_ct_all[valid_a]),
                np.nanmax(p_fan_all[valid_b])) * 1.05
    p_range = np.linspace(0, p_max, 300)
    ax.plot(p_range, np.polyval(r['coeffs_a'], p_range),
            lw=2, color='steelblue', ls='--',
            label=f"方案A: R²={r['r2_a']:.3f} σ={r['sigma_a']:.2f}MW")
    ax.plot(p_range, np.polyval(r['coeffs_b'], p_range),
            lw=2, color='darkorange', ls='-',
            label=f"方案B: R²={r['r2_b']:.3f} σ={r['sigma_b']:.2f}MW")

    # ── 零点标注 ────────────────────────────────────────────────
    c_val = r['coeffs_a'][2]
    ax.axhline(0, color='gray', lw=0.8, ls=':')
    ax.annotate(f'A:L̂(0)={c_val:.2f}MW', xy=(0, c_val),
                xytext=(p_max * 0.05, c_val + 0.3),
                fontsize=7, color='steelblue',
                arrowprops=dict(arrowstyle='->', color='steelblue', lw=0.8))
    ax.annotate('B:L̂(0)=0', xy=(0, 0),
                xytext=(p_max * 0.05, -0.6),
                fontsize=7, color='darkorange',
                arrowprops=dict(arrowstyle='->', color='darkorange', lw=0.8))

    ax.set_title(LINE_NAME_ZH[line], fontsize=11)
    ax.set_xlabel('有效功率 P（MW）', fontsize=9)
    ax.set_ylabel('传输损耗 L（MW）', fontsize=9)
    ax.set_xlim(left=0)
    ax.legend(fontsize=6.5, loc='upper left')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
_fig_out = os.path.join(OUT_FIG_DIR, 'loss_fit_comparison.png')
plt.savefig(_fig_out, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✅ 图表已保存：{_fig_out}")

# ══════════════════════════════════════════════════════════════
# 10. 最终汇总打印
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  拟合结果汇总")
print("=" * 65)

_hdr = f"{'线路':10s} {'方案':20s} {'系数a':>12s} {'系数b':>10s} {'系数c':>8s} {'L̂(0)':>7s} {'R²':>6s} {'σ(MW)':>7s}"
print(_hdr)
print("─" * 80)
for line in ['BING', 'DING', 'WU']:
    r  = results[line]
    ca = r['coeffs_a']
    cb = r['coeffs_b']
    print(f"{LINE_NAME_ZH[line]:10s} {'A 含截距 P=CT':20s}"
          f" {ca[0]:12.4e} {ca[1]:10.5f} {ca[2]:8.4f}"
          f" {ca[2]:7.4f} {r['r2_a']:6.4f} {r['sigma_a']:7.4f}")
    print(f"{LINE_NAME_ZH[line]:10s} {'B 过原点 P=FAN_SUM_S2':20s}"
          f" {cb[0]:12.4e} {cb[1]:10.5f} {0.0:8.4f}"
          f" {0.0:7.4f} {r['r2_b']:6.4f} {r['sigma_b']:7.4f}")
    print()

print("  说明：")
print("  - 方案 A：P=max(CT,0)，含截距项（铁损/空载损耗），L̂(0)=c≠0")
print("  - 方案 B：P=FAN_SUM_S2，过原点，L̂(0)=0（无电流时无损耗）")
print("  - 功率预测用途：CT_pred = FAN_SUM_S2 - L̂_B(FAN_SUM_S2)")
print("  - 对比 R² 说明：含截距模型通常 R² 更高，但在 P→0 时物理解释有悖")
print("  - 两种方案在 P>50 MW 工作区间预测值差异通常 < 0.5 MW")
