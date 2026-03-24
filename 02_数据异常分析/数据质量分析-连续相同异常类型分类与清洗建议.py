"""
#7-4 数据质量分析：连续时间相同数据的异常类型分类与清洗建议
==============================================================
目标：
    针对风机 SCADA 数据中出现的"多个连续时间戳五元组（STATUS, ACTIVE_POWER, REACTIVE_POWER,
    WINDSPEED, WINDDIRECTION）完全相同"的情况，区分以下两大场景：

    ① 确定异常（数据冻结/通讯卡死）：有功功率 ≠ 0 连续相同 → 删除
    ② 模糊情况（零值卡值或真实停机）：有功功率 = 0 连续相同 → 按规则进一步分类

分类规则（三类异常）：
    第一类 —— 全场通讯中断（同一分钟 ≥ 80 台风机同时开始重复）
    第二类 —— 部分通讯中断（同一分钟 5~79 台风机同时开始重复）
    第三类 —— 单机异常（≤4 台同时，需按状态码和持续时长进一步判断）

集电线路分组（用于分线路分析）：
    WU  （戊线）—— 风机 63~109，47 台，全部明阳 MySE6.45-180
    DING（丁线）—— 风机 110~152，43 台，全部东气 DEW-D7000-184
    BING（丙线）—— 风机 153~199，47 台，明阳×16 + 金风×31

异常期间 Fan vs Line 比较：
    在 Type1 全场通讯中断事件中，风机汇总 SCADA 值与集电线路 CT 测点均会冻结，
    但两者恢复时间不同：线路 CT 通常比风机 SCADA 提前恢复，期间产生"幻影差值"。

输入：
    #7-3检查风机数据连续相同情况/峡阳B/联合重复值检测结果.xlsx
    #7-2检查集电线路-全站功率数据连续相同情况/峡阳B/每列连续重复检测结果.csv
    DATA/峡阳B/#7-1*.csv（合并 SCADA 数据，用于异常期间对比分析）

输出：
    DATA/峡阳B/analysis_output/
        25_fan_anomaly_type_distribution.png     异常类型分布（段数+记录数双轴）
        26_fan_zero_power_classification.png     零值卡值细分分类
        27_mass_event_timeline.png               第一类事件时间轴
        28_line_repeat_analysis.png              集电线路连续重复分析
        29_per_line_anomaly_distribution.png     分集电线路的异常类型分布（新增）
        30a_anomaly_fan_vs_line_timeseries.png   典型异常事件 Fan vs Line 时序（新增）
        30b_anomaly_vs_normal_diff_distribution.png  正常 vs 异常 Fan−Line 差值分布（新增）
    #7-4分析结果/峡阳B/
        fan_repeat_classified.csv                风机卡值分类明细
        fan_repeat_cleaning_summary.csv          清洗建议汇总（按风机）
        line_repeat_classified.csv               集电线路卡值分类明细
        per_line_anomaly_breakdown.csv           分集电线路异常类型统计（新增）
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.font_manager as _fm
import warnings
warnings.filterwarnings("ignore")

# ── 中文字体配置 ──────────────────────────────────────────────
# 重建字体缓存（确保新安装的 Noto CJK 字体被识别）
_fm._load_fontmanager(try_read_cache=False)
_CJK_FONT = next(
    (f.name for f in _fm.fontManager.ttflist
     if any(k in f.name for k in ('Noto Sans CJK', 'Noto Serif CJK', 'WenQuanYi', 'SimHei', 'Microsoft YaHei'))),
    None
)
if _CJK_FONT:
    matplotlib.rcParams['font.family']      = _CJK_FONT
    matplotlib.rcParams['axes.unicode_minus'] = False
else:
    matplotlib.rcParams['axes.unicode_minus'] = False
    print("⚠️  未找到中文字体，图表中文字符可能显示为方块。"
          "请安装 fonts-noto-cjk 后重新运行。")

# ── 路径配置 ──────────────────────────────────────────────────
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR    = os.path.join(_SCRIPT_DIR, "..")
FAN_EXCEL    = os.path.join(_ROOT_DIR, "分析结果", "风机重复检测", "峡阳B", "联合重复值检测结果.xlsx")
LINE_CSV     = os.path.join(_ROOT_DIR, "分析结果", "集电线路重复检测", "峡阳B", "每列连续重复检测结果.csv")
OUT_FIG_DIR  = os.path.join(_ROOT_DIR, "DATA", "峡阳B", "analysis_output")
OUT_DATA_DIR = os.path.join(_ROOT_DIR, "分析结果", "数据质量分析", "峡阳B")

os.makedirs(OUT_FIG_DIR,  exist_ok=True)
os.makedirs(OUT_DATA_DIR, exist_ok=True)

# ── 常量 ──────────────────────────────────────────────────────
TOTAL_FANS = 137            # 63~199 号，共 137 台
TOTAL_MINUTES = 226079      # 2024-07-19 之后的有效分钟数（集电线路分析用时段）
MASS_THRESH_TYPE1 = 80      # ≥80 台 → 第一类（全场通讯中断）
MASS_THRESH_TYPE2 = 5       # ≥5  台 → 第二类（部分通讯中断）

# 风机厂家分组
MINGYANG = set(range(63, 110)) | set(range(153, 169))
DONGQI   = set(range(110, 153))
JINFENG  = set(range(169, 200))

STATUS_COMM_FAULT = {
    '明阳': {1.0},         # 无通讯
    '东气': {113.0},       # 通讯故障
    '金风': set(),         # 金风无专用通讯故障码
}
STATUS_GEN = {
    '明阳': {4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0},
    '东气': {101.0, 102.0, 103.0, 104.0, 106.0},
    '金风': {4.0, 5.0},
}
STATUS_STOP = {
    '明阳': {0.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0},
    '东气': {105.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0},
    '金风': {1.0, 2.0, 3.0, 6.0},
}

# ── 辅助函数 ──────────────────────────────────────────────────
def get_manufacturer(fan_num: int) -> str:
    if fan_num in MINGYANG: return '明阳'
    if fan_num in DONGQI:   return '东气'
    if fan_num in JINFENG:  return '金风'
    return '未知'


def get_line(fan_num: int) -> str:
    """将风机编号映射到对应的集电线路（WU/DING/BING）。"""
    if 63  <= fan_num <= 109: return 'WU'
    if 110 <= fan_num <= 152: return 'DING'
    if 153 <= fan_num <= 199: return 'BING'
    return '未知'


def parse_combo(combo: str):
    """解析重复值组合字符串为 (status, active_power, reactive_power, windspeed, winddirection)"""
    try:
        vals = combo.strip('()').split(',')
        return tuple(float(v.strip()) for v in vals[:5])
    except Exception:
        return (None,) * 5


def classify_anomaly(row) -> str:
    """
    按三类异常规则给单条重复段打标签。

    返回标签说明
    ─────────────────────────────────────────────────────────────────
    第一类-全场通讯中断      同时 ≥80 台风机冻结，短时（5-15 min）
    第二类-部分通讯中断      同时 5-79 台风机冻结
    第三类-单机通讯故障      单机，状态码=无通讯/通讯故障
    第三类-发电状态零值      单机，状态码=发电类，但有功=0（疑似冻结）
    第三类-单机非零卡值      单机，有功≠0，持续冻结
    正常停机-保留           单机，状态码=停机/维护类，有功=0
    零值-状态待核实          其他零值情况，需人工复核
    """
    mfr      = row['mfr']
    status   = row['status']
    is_zero  = row['is_zero_power']
    mass_cnt = row['mass_fan_count']

    # ── 按同时重复的风机数分类 ────────────────────────────────
    if mass_cnt >= MASS_THRESH_TYPE1:
        return '第一类-全场通讯中断'
    if mass_cnt >= MASS_THRESH_TYPE2:
        return '第二类-部分通讯中断'

    # 以下均为单机/少数机（<5 台同时）
    if not is_zero:
        return '第三类-单机非零卡值'

    # 有功=0 细分
    if status in STATUS_COMM_FAULT.get(mfr, set()):
        return '第三类-单机通讯故障'
    if status in STATUS_GEN.get(mfr, set()):
        return '第三类-发电状态零值'
    if status in STATUS_STOP.get(mfr, set()):
        return '正常停机-保留'
    return '零值-状态待核实'


# ── 1. 加载风机重复数据 ────────────────────────────────────────
print("加载风机联合重复值检测结果 ...")
fan_df = pd.read_excel(FAN_EXCEL, engine='openpyxl')
fan_df['开始时间'] = pd.to_datetime(fan_df['开始时间'])
fan_df['结束时间'] = pd.to_datetime(fan_df['结束时间'])
print(f"  共 {len(fan_df):,} 条重复段记录")

# 解析重复值组合
parsed = fan_df['重复值组合'].apply(parse_combo)
fan_df[['status', 'active_power', 'reactive_power', 'windspeed', 'winddirection']] = pd.DataFrame(
    parsed.tolist(), index=fan_df.index
)
fan_df['mfr']          = fan_df['风机编号'].apply(get_manufacturer)
fan_df['is_zero_power'] = fan_df['active_power'] == 0.0
fan_df['is_all_zeros']  = (fan_df['active_power'] == 0) & (fan_df['reactive_power'] == 0) & (fan_df['windspeed'] == 0)

# 同一分钟有多少台风机开始重复
time_start_counts = fan_df.groupby('开始时间')['风机编号'].count()
fan_df['mass_fan_count'] = fan_df['开始时间'].map(time_start_counts)

# 分类
fan_df['anomaly_type'] = fan_df.apply(classify_anomaly, axis=1)
fan_df['line']         = fan_df['风机编号'].apply(get_line)

print("分类完成，异常类型分布：")
type_summary = fan_df.groupby('anomaly_type')['持续长度'].agg(段数='count', 总记录数='sum').sort_values('总记录数', ascending=False)
print(type_summary.to_string())
print()


# ── 2. 清洗建议汇总 ────────────────────────────────────────────
DELETE_TYPES = {
    '第一类-全场通讯中断',
    '第二类-部分通讯中断',
    '第三类-单机通讯故障',
    '第三类-发电状态零值',
    '第三类-单机非零卡值',
}
KEEP_TYPES = {'正常停机-保留'}
REVIEW_TYPES = {'零值-状态待核实'}

def get_action(t):
    if t in DELETE_TYPES: return '删除'
    if t in KEEP_TYPES:   return '保留'
    return '人工复核'

fan_df['清洗建议'] = fan_df['anomaly_type'].apply(get_action)

action_summary = fan_df.groupby('清洗建议')['持续长度'].agg(段数='count', 总记录数='sum')
action_summary['占全量比%'] = action_summary['总记录数'] / action_summary['总记录数'].sum() * 100
print("=== 清洗建议汇总 ===")
print(action_summary.to_string())
print()

# 按风机汇总
fan_clean_summary = fan_df.groupby(['风机编号', 'mfr', '清洗建议'])['持续长度'].sum().reset_index()
fan_clean_summary.columns = ['风机编号', '厂家', '清洗建议', '受影响记录数']
fan_clean_summary.to_csv(os.path.join(OUT_DATA_DIR, 'fan_repeat_cleaning_summary.csv'), index=False, encoding='utf-8-sig')
print(f"✅ 按风机清洗汇总已保存")

# 完整分类明细（去掉原始解析字段，保留关键信息）
fan_classified = fan_df[[
    '风机编号', 'mfr', '重复值组合', 'status', 'active_power', 'windspeed',
    '开始时间', '结束时间', '持续长度', 'mass_fan_count', 'is_zero_power',
    'anomaly_type', '清洗建议'
]].copy()
fan_classified.rename(columns={'mfr': '厂家', 'mass_fan_count': '同时重复风机数',
                                'is_zero_power': '有功为零', 'anomaly_type': '异常类型'}, inplace=True)
fan_classified.to_csv(os.path.join(OUT_DATA_DIR, 'fan_repeat_classified.csv'), index=False, encoding='utf-8-sig')
print(f"✅ 风机卡值分类明细已保存")


# ── 3. 加载集电线路重复数据 ────────────────────────────────────
print("\n加载集电线路连续重复检测结果 ...")
line_df = pd.read_csv(LINE_CSV, encoding='gbk')
line_df.columns = line_df.columns.str.strip()
line_df['开始时间'] = pd.to_datetime(line_df['开始时间'], errors='coerce')
line_df['结束时间'] = pd.to_datetime(line_df['结束时间'], errors='coerce')
print(f"  共 {len(line_df):,} 条记录")
print(line_df.head(5).to_string())
print()

line_df['is_zero_value'] = line_df['重复值'].apply(
    lambda x: float(x) == 0.0 if pd.notna(x) else False
)

# 集电线路各字段分类
def classify_line_repeat(row):
    col = row['字段名']
    val = row['重复值']
    dur = row['持续长度']
    try: val_f = float(val)
    except: val_f = None

    if col == 'LIMIT_POWER':
        return '限电指令-可能正常'
    if col == 'ACTIVE_POWER_STATION':
        if dur > 100000:
            return '全站功率长期零值-疑似测点未投运'
        return '全站功率重复-待核实'
    # Collection line columns
    if val_f is not None and val_f != 0.0:
        return '集电线路非零卡值-删除'
    return '集电线路零值-需结合风机数据判断'

line_df['异常类型'] = line_df.apply(classify_line_repeat, axis=1)
line_summary = line_df.groupby(['字段名', '异常类型'])['持续长度'].agg(段数='count', 总记录数='sum')
print("=== 集电线路重复分类汇总 ===")
print(line_summary.to_string())
line_df.to_csv(os.path.join(OUT_DATA_DIR, 'line_repeat_classified.csv'), index=False, encoding='utf-8-sig')
print(f"\n✅ 集电线路卡值分类明细已保存")


# ══════════════════════════════════════════════════════════════
# 图表生成
# ══════════════════════════════════════════════════════════════

COLORS = {
    '第一类-全场通讯中断':    '#d62728',
    '第二类-部分通讯中断':    '#ff7f0e',
    '第三类-单机非零卡值':    '#e377c2',
    '第三类-单机通讯故障':    '#9467bd',
    '第三类-发电状态零值':    '#8c564b',
    '正常停机-保留':          '#2ca02c',
    '零值-状态待核实':        '#bcbd22',
}

# ── Fig 25: 风机异常类型分布（双轴：段数 + 记录数） ─────────────
print("\n生成图表 25 ...")
fig, ax1 = plt.subplots(figsize=(13, 5))

order = type_summary.sort_values('总记录数', ascending=True).index.tolist()
vals_n = [type_summary.loc[t, '段数'] for t in order]
vals_r = [type_summary.loc[t, '总记录数'] for t in order]
colors_bar = [COLORS.get(t, '#7f7f7f') for t in order]

y_pos = np.arange(len(order))

# Bar: 总记录数
bars = ax1.barh(y_pos, vals_r, color=colors_bar, alpha=0.8, height=0.5, label='总记录数')
ax1.set_yticks(y_pos)
ax1.set_yticklabels(order, fontsize=9)
ax1.set_xlabel('总记录数（分钟数）', fontsize=10)
ax1.set_title('风机 SCADA 连续相同数据：异常类型分布\n(2024-07-19～2024-12-24, 137台风机, 已有检测结果)', fontsize=11)

# Annotate segment counts on right
ax2 = ax1.twiny()
ax2.set_xlim(ax1.get_xlim())
ax2.set_xlabel('（数字为段数）', fontsize=9, color='gray')
for i, (seg, rec) in enumerate(zip(vals_n, vals_r)):
    ax1.text(rec + 2000, i, f'{rec:,} 条 / {seg} 段', va='center', fontsize=8.5)

# Action color legend
patches = [
    mpatches.Patch(color='#d62728', label='第一类：全场通讯中断（确定删除）'),
    mpatches.Patch(color='#ff7f0e', label='第二类：部分通讯中断（确定删除）'),
    mpatches.Patch(color='#e377c2', label='第三类：单机非零卡值（确定删除）'),
    mpatches.Patch(color='#9467bd', label='第三类：通讯故障零值（建议删除）'),
    mpatches.Patch(color='#8c564b', label='第三类：发电状态零值（建议删除）'),
    mpatches.Patch(color='#2ca02c', label='正常停机零值（建议保留）'),
    mpatches.Patch(color='#bcbd22', label='零值-状态待核实（人工复核）'),
]
ax1.legend(handles=patches, fontsize=8, loc='lower right', framealpha=0.9)
ax1.grid(True, alpha=0.2, axis='x')
ax2.set_xticks([])
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, '25_fan_anomaly_type_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ 25_fan_anomaly_type_distribution.png")


# ── Fig 26: 零值卡值细分分类饼图 ─────────────────────────────────
print("生成图表 26 ...")
zero_df = fan_df[fan_df['is_zero_power']].copy()
zero_type_summary = zero_df.groupby('anomaly_type')['持续长度'].sum().sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Left: All zero repeats pie
ax = axes[0]
labels_pie = [t.replace('-', '\n') for t in zero_type_summary.index]
colors_pie = [COLORS.get(t, '#7f7f7f') for t in zero_type_summary.index]
sizes = zero_type_summary.values
wedges, texts, autotexts = ax.pie(
    sizes, labels=None, colors=colors_pie, autopct='%1.1f%%',
    startangle=140, pctdistance=0.8,
    wedgeprops={'edgecolor': 'white', 'linewidth': 1.5}
)
for at in autotexts:
    at.set_fontsize(8)
ax.legend(
    wedges, [f'{l}: {v:,}条' for l, v in zip(zero_type_summary.index, sizes)],
    fontsize=8, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=1
)
ax.set_title('零值连续重复段的类型构成\n（按总记录分钟数）', fontsize=10)

# Right: Duration distribution for "ambiguous" vs "clear anomaly" vs "real stop"
ax2 = axes[1]
bins = [0, 10, 30, 60, 120, 360, 720, 1440, 10080, 50000, 300000]
bin_labels = ['≤10', '11-30', '31-60', '1-2h', '2-6h', '6-12h', '12-24h', '1-7天', '7-35天', '>35天']

delete_zero = fan_df[fan_df['is_zero_power'] & fan_df['anomaly_type'].isin(
    ['第一类-全场通讯中断','第二类-部分通讯中断','第三类-单机通讯故障','第三类-发电状态零值'])]
keep_zero   = fan_df[fan_df['anomaly_type'] == '正常停机-保留']
review_zero = fan_df[fan_df['anomaly_type'] == '零值-状态待核实']

for data, label, color, alpha in [
    (delete_zero, '删除（通讯/发电冻结）', '#d62728', 0.7),
    (keep_zero,   '保留（正常停机）',      '#2ca02c', 0.7),
    (review_zero, '待核实',               '#bcbd22', 0.7),
]:
    if len(data) == 0:
        continue
    counts, _ = np.histogram(data['持续长度'], bins=bins)
    x = np.arange(len(counts))
    ax2.bar(x + (0 if label=='删除（通讯/发电冻结）' else 0.3 if label=='保留（正常停机）' else 0.6),
            counts, width=0.28, color=color, alpha=alpha, label=label)

ax2.set_xticks(np.arange(len(bin_labels)) + 0.3)
ax2.set_xticklabels(bin_labels, rotation=25, ha='right', fontsize=8)
ax2.set_ylabel('段数')
ax2.set_xlabel('持续时长')
ax2.set_title('零值段持续时长 × 类型分布\n（用于确定阈值）', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('有功功率=0 的连续相同数据：分类分析', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, '26_fan_zero_power_classification.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ 26_fan_zero_power_classification.png")


# ── Fig 27: 第一类事件时间轴 ─────────────────────────────────────
print("生成图表 27 ...")
mass1_events = fan_df[fan_df['mass_fan_count'] >= MASS_THRESH_TYPE1].copy()
mass1_times = mass1_events.groupby('开始时间').agg(
    风机数=('风机编号', 'count'),
    平均持续=('持续长度', 'mean'),
    有功零占比=('is_zero_power', 'mean')
).reset_index()
mass1_times['开始时间_dt'] = pd.to_datetime(mass1_times['开始时间'])

fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)

ax = axes[0]
ax.scatter(mass1_times['开始时间_dt'], mass1_times['风机数'],
           c=mass1_times['有功零占比'], cmap='RdYlGn_r',
           s=mass1_times['平均持续'] * 0.8, alpha=0.7, edgecolors='gray', linewidths=0.4)
ax.axhline(137, color='red', ls='--', lw=1, label='全场 137 台')
ax.axhline(80,  color='orange', ls=':', lw=1, label='第一类阈值 80 台')
ax.set_ylabel('同时开始重复的风机数')
ax.set_title('第一类异常：全场通讯中断事件时间分布\n（点大小=平均持续时长，颜色=有功为零占比：绿=多零/红=多非零）', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

ax2 = axes[1]
ax2.bar(mass1_times['开始时间_dt'], mass1_times['平均持续'],
        width=1.5, color='steelblue', alpha=0.7, label='平均持续时长（分钟）')
ax2.set_ylabel('平均持续时长（分钟）')
ax2.set_xlabel('时间')
ax2.set_title('第一类事件的平均持续时长（大多数为 5~15 分钟 → 短时通讯中断）', fontsize=10)
ax2.legend(fontsize=9)
ax2.grid(True, alpha=0.3)

plt.xticks(rotation=20, fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, '27_mass_event_timeline.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ 27_mass_event_timeline.png")


# ── Fig 28: 集电线路连续重复分析 ─────────────────────────────────
print("生成图表 28 ...")
line_focus = line_df[~line_df['字段名'].isin(['ACTIVE_POWER_STATION', 'LIMIT_POWER'])].copy()
line_focus['重复值_float'] = pd.to_numeric(line_focus['重复值'], errors='coerce')
line_focus['is_zero'] = line_focus['重复值_float'] == 0.0

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Number of segments by field and type
ax = axes[0]
line_pivot = line_focus.groupby(['字段名', 'is_zero'])['持续长度'].agg(['count','sum']).reset_index()
fields = ['ACTIVE_POWER_BING', 'ACTIVE_POWER_DING', 'ACTIVE_POWER_WU']
x = np.arange(len(fields))
w = 0.3
for i, (is_z, label, color) in enumerate([(False, '非零卡值（确定异常）', '#d62728'), (True, '零值（参考风机状态判断）', '#1f77b4')]):
    seg_counts = [line_pivot[(line_pivot['字段名']==f) & (line_pivot['is_zero']==is_z)]['count'].sum() for f in fields]
    ax.bar(x + i*w, seg_counts, width=w, color=color, alpha=0.8, label=label)

ax.set_xticks(x + w/2)
ax.set_xticklabels([f.replace('ACTIVE_POWER_', '') for f in fields])
ax.set_ylabel('段数')
ax.set_title('集电线路各字段连续重复段分布', fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis='y')

# Right: Duration distribution for line non-zero repeats
ax2 = axes[1]
line_nonzero = line_focus[~line_focus['is_zero']].copy()
bins_l = [0, 5, 10, 15, 30, 60, 120, 300, 1000, 5000]
line_nonzero['bin'] = pd.cut(line_nonzero['持续长度'], bins=bins_l)
dist = line_nonzero.groupby(['字段名', 'bin'])['持续长度'].count().unstack('字段名')
dist.index = [str(b) for b in dist.index]
if not dist.empty:
    dist.plot(kind='bar', ax=ax2, width=0.7, alpha=0.8)
ax2.set_xlabel('持续时长区间（分钟）')
ax2.set_ylabel('段数')
ax2.set_title('集电线路非零卡值持续时长分布\n（非零连续重复=确定异常）', fontsize=10)
ax2.tick_params(axis='x', rotation=30)
ax2.grid(True, alpha=0.3, axis='y')

plt.suptitle('集电线路功率数据连续相同分析\n（BING/DING/WU 三条集电线路）', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, '28_line_repeat_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ 28_line_repeat_analysis.png")


# ══════════════════════════════════════════════════════════════
# 新增：集电线路维度数据质量分析
# ══════════════════════════════════════════════════════════════

# ── 新增：按集电线路统计异常类型 ─────────────────────────────────────
print("\n=== 按集电线路的异常类型分布 ===")
per_line_anomaly = (
    fan_df.groupby(['line', 'anomaly_type'])['持续长度']
    .agg(段数='count', 总记录数='sum')
    .reset_index()
    .sort_values(['line', '总记录数'], ascending=[True, False])
)
print(per_line_anomaly.to_string())
per_line_anomaly.to_csv(
    os.path.join(OUT_DATA_DIR, 'per_line_anomaly_breakdown.csv'),
    index=False, encoding='utf-8-sig'
)
print("✅ 按线路异常汇总已保存")


# ── Fig 29: 分集电线路的风机异常类型分布 ────────────────────────────────
print("\n生成图表 29 ...")
LINE_META = [
    ('BING', 47, '明阳×16 + 金风×31'),
    ('DING', 43, '东气×43'),
    ('WU',   47, '明阳×47'),
]
fig, axes = plt.subplots(1, 3, figsize=(19, 6))
for (lname, n_fans, mfr_desc), ax in zip(LINE_META, axes):
    sub = fan_df[fan_df['line'] == lname]
    ls  = sub.groupby('anomaly_type')['持续长度'].agg(
        段数='count', 总记录数='sum'
    ).sort_values('总记录数', ascending=True)
    y_pos      = np.arange(len(ls))
    colors_bar = [COLORS.get(t, '#7f7f7f') for t in ls.index]
    ax.barh(y_pos, ls['总记录数'], color=colors_bar, alpha=0.85, height=0.55)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(ls.index, fontsize=8.5)
    ax.set_xlabel('总记录数（分钟）', fontsize=9)
    ax.set_title(f'{lname} 线路（{n_fans} 台）\n{mfr_desc}', fontsize=10)
    for j, (seg, rec) in enumerate(zip(ls['段数'], ls['总记录数'])):
        ax.text(rec + max(ls['总记录数']) * 0.01, j,
                f'{rec:,} / {seg}段', va='center', fontsize=7.5)
    ax.set_xlim(0, max(ls['总记录数']) * 1.35)
    ax.grid(True, alpha=0.2, axis='x')
patches29 = [
    mpatches.Patch(color='#d62728', label='第一类：全场通讯中断（确定删除）'),
    mpatches.Patch(color='#ff7f0e', label='第二类：部分通讯中断（确定删除）'),
    mpatches.Patch(color='#e377c2', label='第三类：单机非零卡值（确定删除）'),
    mpatches.Patch(color='#9467bd', label='第三类：通讯故障零值（建议删除）'),
    mpatches.Patch(color='#8c564b', label='第三类：发电状态零值（建议删除）'),
    mpatches.Patch(color='#2ca02c', label='正常停机零值（建议保留）'),
    mpatches.Patch(color='#bcbd22', label='零值-状态待核实（人工复核）'),
]
axes[1].legend(handles=patches29, fontsize=7.5, loc='lower right',
               bbox_to_anchor=(0.5, -0.45), ncol=1)
plt.suptitle(
    '分集电线路：风机 SCADA 连续相同异常类型分布\n'
    '（BING=丙线, DING=丁线, WU=戊线 | 横轴：各线路风机重复段总记录数）',
    fontsize=11, y=1.02
)
plt.tight_layout()
plt.savefig(os.path.join(OUT_FIG_DIR, '29_per_line_anomaly_distribution.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print("  ✅ 29_per_line_anomaly_distribution.png")


# ── 加载 SCADA 合并数据（用于异常期间 Fan vs Line 对比）─────────────────
SCADA_DATA_DIR = os.path.join(_ROOT_DIR, "DATA", "峡阳B")
SCADA_COLS = [
    'timestamp',
    'ACTIVE_POWER_BING', 'ACTIVE_POWER_DING', 'ACTIVE_POWER_WU',
    'BING_ACTIVE_POWER_SUM_S1', 'DING_ACTIVE_POWER_SUM_S1', 'WU_ACTIVE_POWER_SUM_S1',
    'BING_ACTIVE_POWER_SUM_S2', 'DING_ACTIVE_POWER_SUM_S2', 'WU_ACTIVE_POWER_SUM_S2',
]
print("\n加载 SCADA 合并数据（用于异常期间 Fan vs Line 对比）...")
_csv_files = sorted(glob.glob(os.path.join(SCADA_DATA_DIR, "*with_sum*.csv")))
_scada_parts = []
for f in _csv_files:
    try:
        _df = pd.read_csv(f, usecols=SCADA_COLS, parse_dates=['timestamp'])
        _scada_parts.append(_df)
        print(f"  加载: {os.path.basename(f)}  ({len(_df):,} 条)")
    except Exception as e:
        print(f"  跳过: {os.path.basename(f)} ({e})")

if _scada_parts:
    scada = pd.concat(_scada_parts, ignore_index=True)
    scada.sort_values('timestamp', inplace=True)
    scada.drop_duplicates(subset='timestamp', inplace=True)
    scada.reset_index(drop=True, inplace=True)
    scada['timestamp'] = pd.to_datetime(scada['timestamp'])
    print(f"  合并后: {len(scada):,} 条  "
          f"({scada['timestamp'].min().date()} ~ {scada['timestamp'].max().date()})")

    # ── 标注 SCADA 时间戳的异常状态 ─────────────────────────────────────
    def _mark_intervals(ts_arr, segs_df):
        """向量化区间标注：返回 bool ndarray，与 ts_arr 等长。"""
        if len(segs_df) == 0:
            return np.zeros(len(ts_arr), dtype=bool)
        mask   = np.zeros(len(ts_arr), dtype=bool)
        starts = segs_df['开始时间'].values.astype('datetime64[ns]')
        ends   = segs_df['结束时间'].values.astype('datetime64[ns]')
        ts_np  = ts_arr.astype('datetime64[ns]')
        for t0, t1 in zip(starts, ends):
            mask |= (ts_np >= t0) & (ts_np <= t1)
        return mask

    ts_arr = scada['timestamp'].values

    # Type1：全场通讯中断（全线路同步冻结）
    type1_segs_u = (fan_df[fan_df['anomaly_type'] == '第一类-全场通讯中断']
                    .drop_duplicates(subset=['开始时间', '结束时间'])
                    [['开始时间', '结束时间']])
    print(f"  标注 Type1（{len(type1_segs_u)} 个唯一事件窗口）...")
    scada['is_type1'] = _mark_intervals(ts_arr, type1_segs_u)

    # Type2：按集电线路分别标注
    for lname in ['BING', 'DING', 'WU']:
        t2_segs = (fan_df[(fan_df['anomaly_type'] == '第二类-部分通讯中断') &
                          (fan_df['line'] == lname)]
                   .drop_duplicates(subset=['开始时间', '结束时间'])
                   [['开始时间', '结束时间']])
        print(f"  标注 Type2-{lname}（{len(t2_segs)} 个唯一事件窗口）...")
        scada[f'is_type2_{lname}'] = _mark_intervals(ts_arr, t2_segs)

    n_t1 = scada['is_type1'].sum()
    print(f"  Type1 标注: {n_t1:,} 条 ({n_t1 / len(scada) * 100:.1f}% of SCADA timestamps)")

    # ── Fig 30a: 典型 Type1 事件前后的 Fan vs Line 时序图 ─────────────────
    print("\n生成图表 30a ...")
    # 选最大事件（fan_count 最多且 avg_dur ≥ 10 min）
    t1_events = (fan_df[fan_df['anomaly_type'] == '第一类-全场通讯中断']
                 .groupby('开始时间').agg(
                     fan_count=('风机编号', 'count'),
                     avg_dur=('持续长度', 'mean')
                 ).reset_index()
                 .sort_values('fan_count', ascending=False))
    long_events = t1_events[t1_events['avg_dur'] >= 10]
    best_event  = long_events.iloc[0] if len(long_events) > 0 else t1_events.iloc[0]
    evt_time    = best_event['开始时间']
    evt_dur     = int(best_event['avg_dur'])
    evt_fans    = int(best_event['fan_count'])

    PRE_MIN  = 30
    POST_MIN = evt_dur + 30
    t_win_s  = evt_time - pd.Timedelta(minutes=PRE_MIN)
    t_win_e  = evt_time + pd.Timedelta(minutes=POST_MIN)
    win = scada[(scada['timestamp'] >= t_win_s) & (scada['timestamp'] <= t_win_e)].copy()

    LINE_PAIRS_30 = [
        ('BING', 'BING_ACTIVE_POWER_SUM_S2', 'ACTIVE_POWER_BING', '#ff7f0e'),
        ('DING', 'DING_ACTIVE_POWER_SUM_S2', 'ACTIVE_POWER_DING', '#2ca02c'),
        ('WU',   'WU_ACTIVE_POWER_SUM_S2',   'ACTIVE_POWER_WU',   '#1f77b4'),
    ]
    fig, axes30 = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    for (lname, fan_col, line_col, c_fan), ax in zip(LINE_PAIRS_30, axes30):
        ax.plot(win['timestamp'], win[fan_col],
                color=c_fan, lw=1.6, label=f'{lname} 风机汇总 S2（正值保留，负值取0）')
        ax.plot(win['timestamp'], win[line_col],
                color='gray', lw=1.3, ls='--', alpha=0.9,
                label=f'{lname} 集电线路 CT 测点（独立采集）')
        # Shade anomaly window
        ax.axvspan(evt_time,
                   evt_time + pd.Timedelta(minutes=evt_dur),
                   color='red', alpha=0.10, zorder=0)
        ax.axvline(evt_time,
                   color='red', ls=':', lw=1.2, label=f'异常开始 {evt_time.strftime("%H:%M")}')
        ax.axvline(evt_time + pd.Timedelta(minutes=evt_dur),
                   color='darkorange', ls=':', lw=1.2,
                   label=f'Fan SCADA 恢复 {(evt_time + pd.Timedelta(minutes=evt_dur)).strftime("%H:%M")}')
        # Twin axis: diff
        diff = win[fan_col] - win[line_col]
        ax_r = ax.twinx()
        ax_r.plot(win['timestamp'], diff, color='purple', lw=0.9, alpha=0.55, ls=':')
        ax_r.set_ylabel('Fan−Line (MW)', fontsize=7.5, color='purple', labelpad=2)
        ax_r.tick_params(axis='y', labelcolor='purple', labelsize=7)
        ax.set_ylabel(f'{lname} 功率 (MW)', fontsize=9)
        ax.legend(fontsize=7.5, loc='upper left', ncol=2)
        ax.grid(True, alpha=0.2)
        ax.set_title(
            f'{lname} 线路：Fan_sum_S2 vs 集电线路 CT（红色阴影 = Fan SCADA 冻结期）',
            fontsize=9
        )
    plt.suptitle(
        f'典型第一类异常（全场通讯中断）前后的 Fan vs Line 对比\n'
        f'事件：{evt_time.strftime("%Y-%m-%d %H:%M")}，{evt_fans} 台同时冻结，'
        f'持续约 {evt_dur} 分钟\n'
        f'关键观察：集电线路 CT 与 Fan SCADA 同步冻结，但 CT 更早恢复，'
        f'期间差值（紫色虚线）出现"幻影偏移"',
        fontsize=10, y=1.01
    )
    plt.xticks(rotation=20, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, '30a_anomaly_fan_vs_line_timeseries.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ 30a_anomaly_fan_vs_line_timeseries.png")

    # ── Fig 30b: 正常 vs 第一类异常 时段 Fan−Line 差值分布对比 ─────────────
    print("生成图表 30b ...")
    fig, axes30b = plt.subplots(1, 3, figsize=(16, 5))
    for (lname, fan_col, line_col, c_fan), ax in zip(LINE_PAIRS_30, axes30b):
        diff_series = scada[fan_col] - scada[line_col]
        # Restrict to "generating mode" (both fan_sum > 0 and line > 0)
        gen_mask = (scada[fan_col] > 0) & (scada[line_col] > 0)
        normal_d  = diff_series[gen_mask & ~scada['is_type1']]
        anomaly_d = diff_series[gen_mask & scada['is_type1']]
        if len(anomaly_d) == 0:
            ax.set_title(f'{lname}: 异常时段数据不足', fontsize=9)
            continue
        bins = np.linspace(
            min(normal_d.quantile(0.01), anomaly_d.quantile(0.01)),
            max(normal_d.quantile(0.99), anomaly_d.quantile(0.99)),
            55
        )
        ax.hist(normal_d,  bins=bins, density=True, alpha=0.6,
                color='steelblue', label=f'正常时段 (n={len(normal_d):,})')
        ax.hist(anomaly_d, bins=bins, density=True, alpha=0.6,
                color='#d62728',   label=f'第一类异常 (n={len(anomaly_d):,})')
        ax.axvline(normal_d.mean(),  color='steelblue', ls='--', lw=1.8,
                   label=f'正常均值 {normal_d.mean():.1f} MW')
        ax.axvline(anomaly_d.mean(), color='#d62728',   ls='--', lw=1.8,
                   label=f'异常均值 {anomaly_d.mean():.1f} MW')
        ax.set_xlabel('Fan_sum_S2 − Line (MW)', fontsize=9)
        ax.set_ylabel('密度', fontsize=9)
        ax.set_title(
            f'{lname} 线路：Fan−Line 差值分布\n'
            f'正常均值 {normal_d.mean():.1f} MW  vs  '
            f'异常均值 {anomaly_d.mean():.1f} MW\n'
            f'（差值偏移来自 Fan 冻结而 Line CT 已恢复）',
            fontsize=8.5
        )
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.2)
    plt.suptitle(
        '正常时段 vs 第一类异常时段：Fan_sum_S2 − 集电线路测点 差值分布对比\n'
        '（仅限发电工况：Fan_S2 > 0 且 Line > 0）\n'
        '差值在异常期间均值偏移 → 主因：Line CT 提前恢复，Fan SCADA 仍冻结',
        fontsize=10, y=1.04
    )
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_FIG_DIR, '30b_anomaly_vs_normal_diff_distribution.png'),
                dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✅ 30b_anomaly_vs_normal_diff_distribution.png")

else:
    print("  ⚠️ 未找到 SCADA 合并数据，跳过 Fig 30a/30b（Fan vs Line 异常对比）")
    scada = None


# ══════════════════════════════════════════════════════════════
# 动态传输损耗模型拟合（P = FAN_SUM_S2，过原点）及 fan_repeat_six_types.csv 生成
#
# 数据口径说明：
#   FAN_SUM = *_ACTIVE_POWER_SUM_S2：集电线路所属风机有功功率之和，
#     各风机功率已进行截断处理（正值保留、负值取 0），故 FAN_SUM_S2 ≥ 0。
#     物理含义：风机侧净有效出力汇总，是集电线路损耗计算的"输入侧"功率。
#
#   CT = ACTIVE_POWER_* 经 max(·, 0) 截断后的集电线路有功（"输出侧"功率）：
#     CT < 0 表示厂用电通过集电线路向风机供电（反向潮流），净输出为 0。
#
#   实际线损 L = FAN_SUM_S2 − max(CT, 0)
#     L 代表风机汇总出力与集电线路净送出功率的差值（电缆铜损 + 箱变损耗）。
#
#   模型目标：CT_pred = FAN_SUM_S2 − L̂(FAN_SUM_S2)
#     用于功率预测：已知风机功率和 FAN_SUM_S2，预测集电线路功率 CT。
#
# 拟合流程：
#   1. 对每条集电线路独立排除该线路自身所有重复时段（A~E + Normal）
#   2. 排除真正停电时段（CT 和 FAN_SUM_S2 同时接近 0）
#   3. 自变量 P = FAN_SUM_S2（已 ≥ 0），因变量 L = FAN_SUM_S2 − max(CT, 0)
#   4. 发电工况（P > 0）：按 P 分 15 等频分位箱，取各箱中位数 (P_med, L_med)
#   5. 过原点二次拟合（无截距，强制 L̂(0) = 0）：
#        L̂(P) = a·P² + b·P  （使用 lstsq，不含常数项）
#      物理依据：当 FAN_SUM_S2 = 0 时，无电流流过集电线路，损耗自然为 0。
# ══════════════════════════════════════════════════════════════

# 集电线路列映射
# FAN_SUM 使用 _S2（各风机功率正值保留、负值取 0 后的汇总）
# CT 使用 ACTIVE_POWER_*，在计算实际线损时进行 max(·, 0) 截断
_LINE_CT_COL  = {'BING': 'ACTIVE_POWER_BING',      'DING': 'ACTIVE_POWER_DING',      'WU': 'ACTIVE_POWER_WU'}
_LINE_FAN_COL = {'BING': 'BING_ACTIVE_POWER_SUM_S2','DING': 'DING_ACTIVE_POWER_SUM_S2','WU': 'WU_ACTIVE_POWER_SUM_S2'}
_LINE_NAME_ZH = {'BING': '丙（BING）', 'DING': '丁（DING）', 'WU': '戊（WU）'}
_N_BINS = 15

# 六类分类 → 集电线路映射（用于排除各线路自身异常时段）
# fan_df 的 'line' 列已由前面步骤生成
ANOM_SIX_TYPES = ['第一类-全场通讯中断', '第二类-部分通讯中断',
                   '第三类-单机通讯故障', '第三类-发电状态零值',
                   '第三类-单机非零卡值', '正常停机-保留', '零值-状态待核实']
# 以上涵盖全部七种原始异常类型（含 Normal 对应的"正常停机-保留"及"零值-状态待核实"）

loss_models = {}   # line → numpy polynomial coefficients [a, b, 0] (polyval-compatible)

if scada is not None:
    print("\n" + "=" * 70)
    print("  拟合动态传输损耗模型（P = FAN_SUM_S2，过原点：L̂ = a·P² + b·P）")
    print("=" * 70)

    for line in ['BING', 'DING', 'WU']:
        ct_col  = _LINE_CT_COL[line]
        fan_col = _LINE_FAN_COL[line]

        # 本线路全部异常段（所有七种类型合并）
        anom_segs = (fan_df[fan_df['line'] == line][['开始时间', '结束时间']]
                     .drop_duplicates())
        print(f"\n  {_LINE_NAME_ZH[line]}：异常段 {len(anom_segs)} 个")

        # 向量化区间标注
        _mask = np.zeros(len(scada), dtype=bool)
        _ts_np = scada['timestamp'].values.astype('datetime64[ns]')
        for _t0, _t1 in zip(anom_segs['开始时间'].values.astype('datetime64[ns]'),
                             anom_segs['结束时间'].values.astype('datetime64[ns]')):
            _mask |= (_ts_np >= _t0) & (_ts_np <= _t1)

        _normal = scada[~_mask].copy()
        # 排除真正停电时段（CT 和 FAN_SUM 同时接近 0）
        _normal = _normal[
            ~((_normal[ct_col].abs() < 0.1) & (_normal[fan_col].abs() < 0.1))
        ].copy()
        # P = FAN_SUM_S2（已 ≥ 0），L = FAN_SUM_S2 − max(CT, 0)
        _normal['_P'] = _normal[fan_col].clip(lower=0)    # P = FAN_SUM_S2
        _normal['_L'] = _normal[fan_col] - _normal[ct_col].clip(lower=0)  # L = FAN_SUM − max(CT,0)
        n_gen = (_normal['_P'] > 0).sum()
        print(f"    数据量：{len(_normal):,} 条（发电工况 {n_gen:,}）")

        # 发电工况等频分位数箱（15箱，P > 0）→ 箱中位数拟合
        _gen = _normal[_normal['_P'] > 0].copy()
        _gen['_bin'] = pd.qcut(_gen['_P'], q=_N_BINS, duplicates='drop')
        _bins_gen = _gen.groupby('_bin', observed=True).agg(
            P_med=('_P', 'median'), L_med=('_L', 'median'), n=('_P', 'count')
        ).reset_index()

        # 过原点二次拟合：L̂ = a·P² + b·P（无截距，使用 lstsq）
        _X = np.column_stack([_bins_gen['P_med'].values ** 2, _bins_gen['P_med'].values])
        _ab, _, _, _ = np.linalg.lstsq(_X, _bins_gen['L_med'].values, rcond=None)
        # 以 [a, b, 0] 存储，兼容 np.polyval(coeffs, P) 调用
        _coeffs = np.array([_ab[0], _ab[1], 0.0])
        loss_models[line] = _coeffs

        # 拟合质量（R² 对发电箱，σ 对 P>2MW 的发电记录）
        _L_hat_bins = _X @ _ab
        _r2_bins = (1 - (((_bins_gen['L_med'].values - _L_hat_bins) ** 2).sum()
                         / ((_bins_gen['L_med'].values - _bins_gen['L_med'].mean()) ** 2).sum()))
        _gen2 = _gen[_gen['_P'] > 2]
        _sigma = (_gen2['_L'] - np.polyval(_coeffs, _gen2['_P'])).std()
        sign_b = '+' if _coeffs[1] >= 0 else '-'
        print(f"    拟合结果（过原点）：L̂(P) = {_coeffs[0]:.4e}·P² {sign_b} {abs(_coeffs[1]):.5f}·P  [L̂(0)=0]")
        print(f"    R²(发电箱中位数) = {_r2_bins:.4f}   σ(P>2MW 发电数据) = {_sigma:.4f} MW")

    # ── 生成 fan_repeat_six_types.csv ──────────────────────────────────────
    print("\n  生成 fan_repeat_six_types.csv ...")

    # 读取已分类的风机重复段（fan_df 已包含六类重分类信息，但需映射到 A~E+Normal 标签）
    # 此处重新从 fan_df 出发，利用已有的 anomaly_type 和 mass_fan_count 构建六类标签

    # 六类标签映射
    MASS_THRESH_A = 100   # ≥100 台且持续 [9,12] min → A
    MASS_THRESH_B1 = 80   # ≥80 台（非A）→ B1

    def _map_six_type(row) -> str:
        """将原始三类标注映射到 A/B1/B2/C/D/E/Normal 六类。"""
        atype = row['anomaly_type']
        n     = row['mass_fan_count']
        dur   = row['持续长度']
        pw    = row['active_power']   # 冻结有功（kW），0 表示零值冻结

        if atype == '第一类-全场通讯中断':
            if n >= MASS_THRESH_A and 9 <= dur <= 12:
                return 'A'
            return 'B1'
        if atype == '第二类-部分通讯中断':
            return 'B2'
        if atype == '第三类-单机通讯故障':
            return 'C'
        if atype == '第三类-发电状态零值':
            return 'E'
        if atype == '第三类-单机非零卡值':
            return 'D'
        # 正常停机-保留 / 零值-状态待核实 → Normal
        return 'Normal'

    fan_df['six_type'] = fan_df.apply(_map_six_type, axis=1)

    # 加载 SCADA 数据并计算各重复段期间的集电线路均值
    # 按 开始时间 ~ 结束时间 在 scada 中查询对应时段均值
    scada_ts_idx = scada.set_index('timestamp')

    def _get_line_avg(row):
        """返回该重复段期间集电线路 CT 均值和 FAN_SUM 均值（MW）。"""
        line = row['line']
        t0   = row['开始时间']
        t1   = row['结束时间']
        ct_col_  = _LINE_CT_COL.get(line)
        fan_col_ = _LINE_FAN_COL.get(line)
        if ct_col_ is None or fan_col_ is None:
            return pd.Series({'_ct': np.nan, '_fan': np.nan})
        mask_ = (scada['timestamp'] >= t0) & (scada['timestamp'] <= t1)
        sub_ = scada[mask_]
        if len(sub_) == 0:
            return pd.Series({'_ct': np.nan, '_fan': np.nan})
        return pd.Series({'_ct': sub_[ct_col_].mean(), '_fan': sub_[fan_col_].mean()})

    print("    计算各重复段期间集电线路均值（可能需要几分钟）...")
    _avgs = fan_df.apply(_get_line_avg, axis=1)
    fan_df['CT均值MW']      = _avgs['_ct']
    fan_df['FAN_SUM均值MW'] = _avgs['_fan']

    # 实际线损均值（L = FAN_SUM − max(CT, 0)，CT<0 时按 0 处理）
    fan_df['CT有效均值MW']  = fan_df['CT均值MW'].clip(lower=0)   # P_eff = max(CT, 0)
    fan_df['实际损耗均值MW'] = fan_df['FAN_SUM均值MW'] - fan_df['CT有效均值MW']

    # 预期线损（使用 P = FAN_SUM_S2 代入过原点模型 L̂(P) = a·P² + b·P）
    def _expected_loss(row):
        line   = row['line']
        fan_v  = row['FAN_SUM均值MW']   # P = FAN_SUM_S2（自变量）
        if line not in loss_models or np.isnan(fan_v):
            return np.nan
        p = max(fan_v, 0.0)   # P = FAN_SUM_S2（已 ≥ 0）
        if p <= 2:
            return np.nan
        return float(np.polyval(loss_models[line], p))

    fan_df['预期损耗均值MW'] = fan_df.apply(_expected_loss, axis=1)
    fan_df['超额损耗均值MW'] = fan_df['实际损耗均值MW'] - fan_df['预期损耗均值MW']
    fan_df['超额损耗总量MWmin'] = fan_df['超额损耗均值MW'] * fan_df['持续长度']
    fan_df['超额损耗总量MWh']   = fan_df['超额损耗总量MWmin'] / 60.0

    # 整理输出列
    _out = fan_df[[
        '风机编号', 'mfr', 'line', '开始时间', '结束时间', '持续长度',
        'status', 'active_power', 'windspeed', 'mass_fan_count',
        'six_type',
        'CT均值MW', 'CT有效均值MW', 'FAN_SUM均值MW', '实际损耗均值MW', '预期损耗均值MW',
        '超额损耗均值MW', '超额损耗总量MWmin', '超额损耗总量MWh',
    ]].copy()
    _out.rename(columns={
        'mfr':           '厂商',
        'line':          '集电线路',
        '持续长度':      '持续长度(min)',
        'status':        '状态码',
        'active_power':  '冻结有功kW',
        'windspeed':     '冻结风速ms',
        'mass_fan_count':'同时冻结风机数',
        'six_type':      '异常类型',
    }, inplace=True)
    _out_path = os.path.join(OUT_DATA_DIR, 'fan_repeat_six_types.csv')
    _out.to_csv(_out_path, index=False, encoding='utf-8-sig')
    print(f"    ✅ fan_repeat_six_types.csv 已保存 ({len(_out):,} 行)")
    print(f"       → {_out_path}")

    # 快速摘要
    print("\n  === 超额损耗汇总（每类型，唯一线路事件，MWh）===")
    for _stype in ['A', 'B1', 'B2', 'C', 'D', 'E', 'Normal']:
        _sub = _out[_out['异常类型'] == _stype]
        for _line in ['BING', 'DING', 'WU']:
            _u = _sub[_sub['集电线路'] == _line].drop_duplicates(subset=['开始时间', '持续长度(min)'])
            _tot = (_u['超额损耗均值MW'] * _u['持续长度(min)'] / 60).sum()
            print(f"    {_stype:6s} {_line}: {_tot:+.0f} MWh  ({len(_u)} 唯一事件)")

else:
    print("  ⚠️ 未找到 SCADA 数据，跳过传输损耗模型拟合及 fan_repeat_six_types.csv 生成")


# ══════════════════════════════════════════════════════════════
# 终端汇总输出
# ══════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("  清洗建议终端汇总")
print("=" * 70)
print()
total_segs = len(fan_df)
total_recs = fan_df['持续长度'].sum()

for action in ['删除', '保留', '人工复核']:
    sub = fan_df[fan_df['清洗建议'] == action]
    print(f"  [{action}]  {len(sub):5,} 段 / {sub['持续长度'].sum():8,} 条分钟记录")
    for atype in sub['anomaly_type'].unique():
        s2 = sub[sub['anomaly_type'] == atype]
        print(f"      {atype}: {len(s2)} 段, {s2['持续长度'].sum():,} 条")
print()
print(f"  全部风机重复段合计: {total_segs:,} 段, {total_recs:,} 条")
print()

# Per-line fan anomaly breakdown
print("  === 按集电线路汇总（风机） ===")
for lname in ['BING', 'DING', 'WU']:
    sub_l = fan_df[fan_df['line'] == lname]
    del_l  = sub_l[sub_l['清洗建议'] == '删除']['持续长度'].sum()
    keep_l = sub_l[sub_l['清洗建议'] == '保留']['持续长度'].sum()
    rev_l  = sub_l[sub_l['清洗建议'] == '人工复核']['持续长度'].sum()
    print(f"  {lname}: 删除 {del_l:,} 条 | 保留 {keep_l:,} 条 | 复核 {rev_l:,} 条")
print()

# Per-line collection stats
print("  集电线路非零卡值（确定删除）:")
for col in ['ACTIVE_POWER_BING', 'ACTIVE_POWER_DING', 'ACTIVE_POWER_WU']:
    sub_l = line_focus[(line_focus['字段名'] == col) & (~line_focus['is_zero'])]
    print(f"    {col}: {len(sub_l)} 段, {sub_l['持续长度'].sum()} 条")
print()
print("  集电线路零值（参考风机数据判断）:")
for col in ['ACTIVE_POWER_BING', 'ACTIVE_POWER_DING', 'ACTIVE_POWER_WU']:
    sub_l = line_focus[(line_focus['字段名'] == col) & (line_focus['is_zero'])]
    print(f"    {col}: {len(sub_l)} 段, {sub_l['持续长度'].sum()} 条")

print()
print("=" * 70)
print("全部结果已保存至：")
print(f"  图表: {OUT_FIG_DIR}/25~30*.png")
print(f"  数据: {OUT_DATA_DIR}/")
print("=" * 70)
