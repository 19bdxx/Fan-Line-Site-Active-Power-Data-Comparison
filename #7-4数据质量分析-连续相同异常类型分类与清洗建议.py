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

输入：
    #7-3检查风机数据连续相同情况/峡阳B/联合重复值检测结果.xlsx
    #7-2检查集电线路-全站功率数据连续相同情况/峡阳B/每列连续重复检测结果.csv

输出：
    DATA/峡阳B/analysis_output/
        25_fan_anomaly_type_distribution.png     异常类型分布（段数+记录数双轴）
        26_fan_zero_power_classification.png     零值卡值细分分类
        27_mass_event_timeline.png               第一类事件时间轴
        28_line_repeat_analysis.png              集电线路连续重复分析
    #7-4分析结果/峡阳B/
        fan_repeat_classified.csv                风机卡值分类明细
        fan_repeat_cleaning_summary.csv          清洗建议汇总（按风机）
        line_repeat_classified.csv               集电线路卡值分类明细
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

# ── 路径配置 ──────────────────────────────────────────────────
FAN_EXCEL    = "#7-3检查风机数据连续相同情况/峡阳B/联合重复值检测结果.xlsx"
LINE_CSV     = "#7-2检查集电线路-全站功率数据连续相同情况/峡阳B/每列连续重复检测结果.csv"
OUT_FIG_DIR  = "DATA/峡阳B/analysis_output"
OUT_DATA_DIR = "#7-4分析结果/峡阳B"

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
print(f"  图表: {OUT_FIG_DIR}/25~28_*.png")
print(f"  数据: {OUT_DATA_DIR}/")
print("=" * 70)
