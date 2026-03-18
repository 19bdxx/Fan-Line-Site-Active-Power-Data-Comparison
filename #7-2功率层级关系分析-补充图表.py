"""
峡阳B风场功率层级关系补充图表生成脚本（图15-24）

本脚本读取 DATA/峡阳B/ 下所有 *with_sum*.csv 文件，对风机汇总功率、
集电线路CT测点进行多维度对比分析，并生成图15至图24，保存至
DATA/峡阳B/analysis_output/ 目录。

分析维度包括：
  - 分功率区间传输损耗对比（图15）
  - 各线路散点图（图16）
  - 月度损耗变化（图17）
  - 损耗分布直方图（图18）
  - 总体损耗率对比（图19）
  - FAN_S1 vs LINE_S1 四象限分析（图20）
  - 消耗模式功率构成（图21）
  - 全场消耗功率平衡（图22）
  - 典型全场耗电时段时序示例（图23）
  - 运行状态分布（图24）
"""

import glob
import os
import warnings

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── 字体设置 ────────────────────────────────────────────────────────────────
FONT_PATH = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(FONT_PATH):
    fm.fontManager.addfont(FONT_PATH)
matplotlib.rcParams["font.family"] = "Noto Sans CJK JP"
matplotlib.rcParams["axes.unicode_minus"] = False

# ── 路径 ─────────────────────────────────────────────────────────────────────
DATA_DIR = "DATA/峡阳B"
OUT_DIR = os.path.join(DATA_DIR, "analysis_output")
os.makedirs(OUT_DIR, exist_ok=True)

DPI = 150
CUT_DATE = pd.Timestamp("2024-07-19")

# ── 加载数据 ──────────────────────────────────────────────────────────────────
files = sorted(glob.glob(os.path.join(DATA_DIR, "*with_sum*.csv")))
print(f"找到 {len(files)} 个数据文件，正在加载…")
df = pd.concat(
    [pd.read_csv(f, parse_dates=["timestamp"]) for f in files],
    ignore_index=True,
)
df.sort_values("timestamp", inplace=True)
df.reset_index(drop=True, inplace=True)
print(f"数据加载完成，共 {len(df):,} 行，时间范围 {df['timestamp'].min()} ~ {df['timestamp'].max()}")

# ── 派生列 ────────────────────────────────────────────────────────────────────
# 各线路传输损耗（S2策略：Fan_S2 - Line_CT）
for line, ct in [("BING", "BING"), ("DING", "DING"), ("WU", "WU")]:
    df[f"{line}_LOSS"] = df[f"{line}_ACTIVE_POWER_SUM_S2"] - df[f"ACTIVE_POWER_{ct}"]
    df[f"{line}_LOSS_RATE"] = np.where(
        df[f"{line}_ACTIVE_POWER_SUM_S2"] > 0,
        df[f"{line}_LOSS"] / df[f"{line}_ACTIVE_POWER_SUM_S2"] * 100,
        np.nan,
    )

# 全场汇总（S1 策略）
df["FAN_LINE_SUM_S1"] = (
    df["BING_ACTIVE_POWER_SUM_S1"]
    + df["DING_ACTIVE_POWER_SUM_S1"]
    + df["WU_ACTIVE_POWER_SUM_S1"]
)
df["LINE_TOTAL"] = (
    df["ACTIVE_POWER_BING"] + df["ACTIVE_POWER_DING"] + df["ACTIVE_POWER_WU"]
)

# 发电工况过滤（2024-07-19 后）
gen_mask = {}
for line, ct in [("BING", "BING"), ("DING", "DING"), ("WU", "WU")]:
    gen_mask[line] = (
        (df["timestamp"] > CUT_DATE)
        & (df[f"{line}_ACTIVE_POWER_SUM_S2"] > 0)
        & (df[f"ACTIVE_POWER_{ct}"] > 0)
    )

df_post = df[df["timestamp"] > CUT_DATE].copy()

# ═══════════════════════════════════════════════════════════════════════════════
# 图15：分功率区间传输损耗对比
# ═══════════════════════════════════════════════════════════════════════════════
print("生成图15…")
bins = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 310]
bin_labels = [f"{bins[i]}-{bins[i+1]} MW" for i in range(len(bins) - 1)]

records = []
for line, ct, color in [("BING", "BING", "orange"), ("DING", "DING", "green"), ("WU", "WU", "steelblue")]:
    tmp = df[gen_mask[line]].copy()
    tmp["bin"] = pd.cut(
        tmp[f"{line}_ACTIVE_POWER_SUM_S2"], bins=bins, labels=bin_labels, right=True
    )
    grp = tmp.groupby("bin", observed=True).agg(
        mean_loss=(f"{line}_LOSS", "mean"),
        mean_rate=(f"{line}_LOSS_RATE", "mean"),
    )
    grp["line"] = line
    grp["color"] = color
    records.append(grp)

fig15, (ax15a, ax15b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, dpi=DPI)
fig15.suptitle("三线路传输损耗分功率区间对比（2024-07-19后发电工况）", fontsize=14)

x = np.arange(len(bin_labels))
width = 0.25
for i, (grp, lname, color) in enumerate(
    zip(records, ["BING", "DING", "WU"], ["orange", "green", "steelblue"])
):
    ax15a.bar(x + (i - 1) * width, grp["mean_loss"].values, width, label=lname, color=color, alpha=0.85)
    ax15b.bar(x + (i - 1) * width, grp["mean_rate"].values, width, label=lname, color=color, alpha=0.85)

for ax in (ax15a, ax15b):
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

ax15a.set_ylabel("传输损耗均值 (MW)")
ax15b.set_ylabel("传输损耗率 (%)")
ax15b.set_xlabel("功率区间")
plt.tight_layout()
fig15.savefig(os.path.join(OUT_DIR, "15_line_transmission_loss_comparison.png"), dpi=DPI)
plt.close(fig15)
print("✅ fig15.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 图16：各线路散点图
# ═══════════════════════════════════════════════════════════════════════════════
print("生成图16…")
fig16, axes16 = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)
fig16.suptitle("各线路风机汇总功率 vs 集电线路测点散点图", fontsize=14)

for ax, (line, ct, title) in zip(
    axes16,
    [("BING", "BING", "BING线路"), ("DING", "DING", "DING线路"), ("WU", "WU", "WU线路")],
):
    tmp = df[gen_mask[line]][[f"{line}_ACTIVE_POWER_SUM_S2", f"ACTIVE_POWER_{ct}", f"{line}_LOSS"]].dropna()
    if len(tmp) > 10000:
        tmp = tmp.sample(10000, random_state=42)
    x_vals = tmp[f"{line}_ACTIVE_POWER_SUM_S2"].values
    y_vals = tmp[f"ACTIVE_POWER_{ct}"].values
    c_vals = tmp[f"{line}_LOSS"].values
    vmax = np.percentile(np.abs(c_vals), 95)
    sc = ax.scatter(x_vals, y_vals, c=c_vals, cmap="RdYlGn_r", s=5, alpha=0.5,
                    vmin=-vmax, vmax=vmax)
    lim = max(x_vals.max(), y_vals.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="y=x")
    ax.set_xlabel("风机汇总功率 S2 (MW)", fontsize=9)
    ax.set_ylabel("集电线路测点 (MW)", fontsize=9)
    ax.set_title(title)
    ax.legend(fontsize=8)
    plt.colorbar(sc, ax=ax, label="损耗 (MW)")

plt.tight_layout()
fig16.savefig(os.path.join(OUT_DIR, "16_fan_vs_line_scatter_per_line.png"), dpi=DPI)
plt.close(fig16)
print("✅ fig16.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 图17：月度传输损耗变化
# ═══════════════════════════════════════════════════════════════════════════════
print("生成图17…")
monthly_records = []
for line, ct, color in [("BING", "BING", "orange"), ("DING", "DING", "green"), ("WU", "WU", "steelblue")]:
    tmp = df[gen_mask[line]].copy()
    tmp["month"] = tmp["timestamp"].dt.to_period("M")
    grp = tmp.groupby("month", observed=True).agg(
        mean_loss=(f"{line}_LOSS", "mean"),
        mean_rate=(f"{line}_LOSS_RATE", "mean"),
    )
    monthly_records.append((grp, line, color))

all_months = sorted(
    set().union(*[set(r[0].index) for r in monthly_records])
)
month_labels = [str(m) for m in all_months]
x_m = np.arange(len(all_months))

fig17, (ax17a, ax17b) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, dpi=DPI)
fig17.suptitle("月度传输损耗变化（2024-07-19后发电工况）", fontsize=14)

for i, (grp, lname, color) in enumerate(monthly_records):
    vals_loss = [grp["mean_loss"].get(m, np.nan) for m in all_months]
    vals_rate = [grp["mean_rate"].get(m, np.nan) for m in all_months]
    ax17a.bar(x_m + (i - 1) * 0.25, vals_loss, 0.25, label=lname, color=color, alpha=0.85)
    ax17b.bar(x_m + (i - 1) * 0.25, vals_rate, 0.25, label=lname, color=color, alpha=0.85)

for ax in (ax17a, ax17b):
    ax.set_xticks(x_m)
    ax.set_xticklabels(month_labels, rotation=30, ha="right", fontsize=9)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

ax17a.set_ylabel("损耗均值 (MW)")
ax17b.set_ylabel("传输损耗率 (%)")
ax17b.set_xlabel("月份")
plt.tight_layout()
fig17.savefig(os.path.join(OUT_DIR, "17_monthly_loss_per_line.png"), dpi=DPI)
plt.close(fig17)
print("✅ fig17.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 图18：损耗分布直方图
# ═══════════════════════════════════════════════════════════════════════════════
print("生成图18…")
fig18, axes18 = plt.subplots(1, 3, figsize=(15, 5), dpi=DPI)
fig18.suptitle("各线路传输损耗分布直方图（2024-07-19后发电工况）", fontsize=14)

for ax, (line, ct, title) in zip(
    axes18,
    [("BING", "BING", "BING线路"), ("DING", "DING", "DING线路"), ("WU", "WU", "WU线路")],
):
    vals = df[gen_mask[line]][f"{line}_LOSS"].dropna().values
    mu, sigma = vals.mean(), vals.std()
    ax.hist(vals, bins=60, color="steelblue", alpha=0.7, edgecolor="white")
    ax.axvline(mu, color="red", linestyle="--", lw=1.5, label=f"均值={mu:.2f}")
    ax.set_title(f"{title}\n均值={mu:.2f} MW  标准差={sigma:.2f} MW", fontsize=9)
    ax.set_xlabel("传输损耗 (MW)", fontsize=9)
    ax.set_ylabel("频次", fontsize=9)
    ax.legend(fontsize=8)

plt.tight_layout()
fig18.savefig(os.path.join(OUT_DIR, "18_loss_distribution_per_line.png"), dpi=DPI)
plt.close(fig18)
print("✅ fig18.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 图19：总体损耗率对比
# ═══════════════════════════════════════════════════════════════════════════════
print("生成图19…")
summary_lines = []
for line, ct in [("BING", "BING"), ("DING", "DING"), ("WU", "WU")]:
    tmp = df[gen_mask[line]]
    total_fan = tmp[f"{line}_ACTIVE_POWER_SUM_S2"].sum()
    total_line = tmp[f"ACTIVE_POWER_{ct}"].sum()
    total_loss = total_fan - total_line
    loss_rate = total_loss / total_fan * 100 if total_fan > 0 else 0
    mean_loss = tmp[f"{line}_LOSS"].mean()
    summary_lines.append((line, loss_rate, mean_loss))

fig19, ax19 = plt.subplots(figsize=(8, 5), dpi=DPI)
fig19.suptitle("三线路总体传输损耗率对比（2024-07-19后发电工况）", fontsize=13)

line_names = [s[0] for s in summary_lines]
rates = [s[1] for s in summary_lines]
means = [s[2] for s in summary_lines]
colors = ["orange", "green", "steelblue"]
bars = ax19.bar(line_names, rates, color=colors, alpha=0.85, width=0.5)
for bar, rate, mean in zip(bars, rates, means):
    ax19.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.1,
        f"{rate:.1f}%\n({mean:.2f} MW)",
        ha="center", va="bottom", fontsize=10,
    )
ax19.set_xlabel("集电线路")
ax19.set_ylabel("传输损耗率 (%)")
ax19.grid(axis="y", alpha=0.3)
ax19.text(
    0.5, -0.18,
    "注：损耗包含箱变铁损、电缆线损及明阳风机自耗电，实际线路损耗更低",
    ha="center", va="center", transform=ax19.transAxes,
    fontsize=8, color="gray",
)
plt.tight_layout()
fig19.savefig(os.path.join(OUT_DIR, "19_loss_rate_summary.png"), dpi=DPI)
plt.close(fig19)
print("✅ fig19.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 图20：FAN_S1 vs LINE_S1 四象限分析
# ═══════════════════════════════════════════════════════════════════════════════
print("生成图20…")
df20 = df_post[["FAN_LINE_SUM_S1", "LINE_TOTAL",
                 "BING_ACTIVE_POWER_SUM_S1", "DING_ACTIVE_POWER_SUM_S1",
                 "WU_ACTIVE_POWER_SUM_S1",
                 "ACTIVE_POWER_BING", "ACTIVE_POWER_DING", "ACTIVE_POWER_WU"]].dropna()

# 四象限颜色
def quadrant_color(fan, line):
    if fan > 0 and line > 0:
        return "steelblue"    # Q1 正常发电
    elif fan > 0 and line <= 0:
        return "orange"       # Q2/Q4 混合状态
    elif fan <= 0 and line <= 0:
        return "red"          # Q3 全场耗电
    else:
        return "gray"         # 其他

colors20 = df20.apply(
    lambda r: quadrant_color(r["FAN_LINE_SUM_S1"], r["LINE_TOTAL"]), axis=1
)

sample20 = df20.sample(min(15000, len(df20)), random_state=42)
colors20_s = colors20.loc[sample20.index]

fig20, (ax20a, ax20b) = plt.subplots(1, 2, figsize=(14, 6), dpi=DPI)
fig20.suptitle("FAN_S1 vs LINE_S1 四象限分析（2024-07-19后）", fontsize=14)

ax20a.scatter(sample20["FAN_LINE_SUM_S1"], sample20["LINE_TOTAL"],
              c=colors20_s, s=4, alpha=0.4)
ax20a.axhline(0, color="black", lw=0.8, ls="--")
ax20a.axvline(0, color="black", lw=0.8, ls="--")
ax20a.set_xlabel("风机汇总功率 S1 (MW)")
ax20a.set_ylabel("集电线路总功率 (MW)")
ax20a.set_title("全域四象限分布")

# 添加象限标注
xrange = ax20a.get_xlim()
yrange = ax20a.get_ylim()
ax20a.text(max(xrange) * 0.5, max(yrange) * 0.8, "正常发电", color="steelblue", fontsize=9, ha="center")
ax20a.text(max(xrange) * 0.5, min(yrange) * 0.5, "混合状态", color="orange", fontsize=9, ha="center")
ax20a.text(min(xrange) * 0.5, min(yrange) * 0.5, "全场耗电", color="red", fontsize=9, ha="center")

# 右图：消耗区间（fan≤0 and line≤0）
consume = df20[(df20["FAN_LINE_SUM_S1"] <= 0) & (df20["LINE_TOTAL"] <= 0)]
line_colors = {"BING": "orange", "DING": "green", "WU": "steelblue"}
for line, ct, lcolor in [("BING", "ACTIVE_POWER_BING", "orange"),
                          ("DING", "ACTIVE_POWER_DING", "green"),
                          ("WU", "ACTIVE_POWER_WU", "steelblue")]:
    ax20b.scatter(
        consume[f"{line}_ACTIVE_POWER_SUM_S1"],
        consume[ct],
        c=lcolor, s=6, alpha=0.5, label=f"{line}线", rasterized=True,
    )
ax20b.axhline(0, color="black", lw=0.8, ls="--")
ax20b.axvline(0, color="black", lw=0.8, ls="--")
ax20b.set_xlabel("各线路风机 S1 功率 (MW)")
ax20b.set_ylabel("各线路CT测点 (MW)")
ax20b.set_title("消耗区间（各线路分色）")
ax20b.legend(fontsize=9)
plt.tight_layout()
fig20.savefig(os.path.join(OUT_DIR, "20_fan_vs_line_quadrant.png"), dpi=DPI)
plt.close(fig20)
print("✅ fig20.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 图21：消耗模式功率构成分解
# ═══════════════════════════════════════════════════════════════════════════════
print("生成图21…")
consume_stats = []
for line, ct in [("BING", "ACTIVE_POWER_BING"), ("DING", "ACTIVE_POWER_DING"), ("WU", "ACTIVE_POWER_WU")]:
    mask_c = (df_post[f"{line}_ACTIVE_POWER_SUM_S1"] < 0) & (df_post[ct] < 0)
    sub = df_post[mask_c]
    if len(sub) == 0:
        consume_stats.append((line, 0, 0))
        continue
    fan_scada = sub[f"{line}_ACTIVE_POWER_SUM_S1"].mean()   # 负值
    line_ct = sub[ct].mean()                                  # 负值
    # abs(fan_scada): 风机SCADA耗电；hidden = |fan_scada| - |line_ct| (箱变+电缆)
    # 注意：|fan_scada| < |line_ct|，因线路需要提供额外铁损，故hidden可为负
    # 重新定义：total_supply=|line_ct|，fan_part=|fan_scada|，box_loss=|line_ct|-|fan_scada|
    consume_stats.append((line, abs(fan_scada), abs(line_ct) - abs(fan_scada)))

fig21, ax21 = plt.subplots(figsize=(9, 5), dpi=DPI)
fig21.suptitle("消耗模式功率构成分解（三线路分别）", fontsize=13)

line_names21 = [s[0] for s in consume_stats]
fan_parts = [s[1] for s in consume_stats]
box_parts = [s[2] for s in consume_stats]
x21 = np.arange(len(line_names21))
b1 = ax21.bar(x21, fan_parts, width=0.5, label="风机SCADA耗电", color="steelblue", alpha=0.85)
b2 = ax21.bar(x21, box_parts, width=0.5, bottom=fan_parts, label="箱变铁损+电缆损耗（隐藏）",
              color="orange", alpha=0.85)

for i, (fp, bp) in enumerate(zip(fan_parts, box_parts)):
    total = fp + bp
    ax21.text(i, total + 0.01, f"合计\n{total:.3f} MW", ha="center", va="bottom", fontsize=8)

ax21.set_xticks(x21)
ax21.set_xticklabels(line_names21)
ax21.set_ylabel("平均耗电功率 (MW)")
ax21.set_xlabel("集电线路")
ax21.legend(fontsize=9)
ax21.grid(axis="y", alpha=0.3)
plt.tight_layout()
fig21.savefig(os.path.join(OUT_DIR, "21_consuming_mode_composition.png"), dpi=DPI)
plt.close(fig21)
print("✅ fig21.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 图22：全场消耗状态功率平衡
# ═══════════════════════════════════════════════════════════════════════════════
print("生成图22…")
mask22 = (
    (df["timestamp"] >= pd.Timestamp("2024-10-01"))
    & (df["BING_ACTIVE_POWER_SUM_S1"] < 0)
    & (df["DING_ACTIVE_POWER_SUM_S1"] < 0)
    & (df["WU_ACTIVE_POWER_SUM_S1"] < 0)
)
df22 = df[mask22].copy()

if len(df22) > 0:
    # 各分项均值（绝对值）
    line_total_mean = df22["LINE_TOTAL"].abs().mean()
    fan_scada_mean = df22["FAN_LINE_SUM_S1"].abs().mean()
    box_loss_mean = line_total_mean - fan_scada_mean  # 箱变+电缆

    # 场站测点
    station_mean = df22["ACTIVE_POWER_STATION"].abs().mean() if "ACTIVE_POWER_STATION" in df22.columns else np.nan
    station_aux = (station_mean - line_total_mean) if not np.isnan(station_mean) else np.nan

    labels22 = ["集电线路合计\n(CT测点)", "风机SCADA\n汇总耗电", "箱变铁损+\n电缆损耗"]
    values22 = [line_total_mean, fan_scada_mean, box_loss_mean]
    colors22 = ["steelblue", "orange", "tomato"]

    if not np.isnan(station_mean):
        labels22 = ["场站总供电\n(ACTIVE_POWER_STATION)"] + labels22 + ["场站辅助用电\n(差值)"]
        values22 = [station_mean] + values22 + [station_aux if not np.isnan(station_aux) else 0]
        colors22 = ["purple"] + colors22 + ["gray"]

    fig22, ax22 = plt.subplots(figsize=(10, 5), dpi=DPI)
    fig22.suptitle("全场消耗状态功率平衡（2024-10后 FAN_S1<0 AND LINE<0）\n"
                   f"样本数：{len(df22):,} 条", fontsize=12)
    bars22 = ax22.bar(labels22, values22, color=colors22, alpha=0.85, width=0.5)
    for bar, val in zip(bars22, values22):
        ax22.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                  f"{val:.3f} MW", ha="center", va="bottom", fontsize=9)
    ax22.set_ylabel("均值 (MW)")
    ax22.set_xlabel("功率组成分项")
    ax22.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig22.savefig(os.path.join(OUT_DIR, "22_whole_farm_power_balance.png"), dpi=DPI)
    plt.close(fig22)
    print("✅ fig22.png")
else:
    print("⚠ 图22：无满足条件的数据，跳过")

# ═══════════════════════════════════════════════════════════════════════════════
# 图23：典型全场耗电时段时序示例
# ═══════════════════════════════════════════════════════════════════════════════
print("生成图23…")
df23 = df_post[["timestamp", "FAN_LINE_SUM_S1", "LINE_TOTAL"]].dropna().copy()
df23 = df23.set_index("timestamp").sort_index()

# 找连续 FAN_LINE_SUM_S1 < -1 MW 的窗口（全场整体净耗电）
consuming = (df23["FAN_LINE_SUM_S1"] < -1).astype(int)
# 标记连续段
consuming_diff = consuming.diff().fillna(0)
episode_starts = df23.index[consuming_diff == 1]
episode_ends = df23.index[consuming_diff == -1]

if len(episode_starts) > 0 and len(episode_ends) > 0:
    # 对齐起止
    if episode_ends[0] < episode_starts[0]:
        episode_ends = episode_ends[1:]
    min_len = min(len(episode_starts), len(episode_ends))
    episode_starts = episode_starts[:min_len]
    episode_ends = episode_ends[:min_len]
    durations = [(e - s).total_seconds() / 60 for s, e in zip(episode_starts, episode_ends)]

    # 找最长 >= 30 min 的耗电时段
    qualified = [(s, e, d) for s, e, d in zip(episode_starts, episode_ends, durations) if d >= 30]
    if qualified:
        best = max(qualified, key=lambda x: x[2])
        ep_start, ep_end, ep_dur = best
        center = ep_start + (ep_end - ep_start) / 2
        window_start = center - pd.Timedelta(minutes=30)
        window_end = center + pd.Timedelta(minutes=30)
        sub23 = df23.loc[window_start:window_end]

        fig23, ax23 = plt.subplots(figsize=(12, 5), dpi=DPI)
        ax23.plot(sub23.index, sub23["FAN_LINE_SUM_S1"], color="steelblue", lw=1.5, label="风机汇总 S1")
        ax23.plot(sub23.index, sub23["LINE_TOTAL"], color="orange", lw=1.5, label="集电线路合计")
        shade_start = max(ep_start, window_start)
        shade_end = min(ep_end, window_end)
        ax23.axvspan(shade_start, shade_end, color="red", alpha=0.15, label="耗电时段")
        ax23.axhline(0, color="black", lw=0.8, ls="--")
        ax23.set_xlabel("时间")
        ax23.set_ylabel("有功功率 (MW)")
        ax23.set_title(f"典型全场耗电时段时序图示例\n耗电时段：{ep_start.strftime('%Y-%m-%d %H:%M')} ~ {ep_end.strftime('%H:%M')}，持续 {ep_dur:.0f} 分钟")
        ax23.legend(fontsize=9)
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        fig23.savefig(os.path.join(OUT_DIR, "23_consuming_episode_timeseries.png"), dpi=DPI)
        plt.close(fig23)
        print("✅ fig23.png")
    else:
        print("⚠ 图23：未找到 ≥30 分钟的耗电时段，跳过")
else:
    print("⚠ 图23：未检测到耗电时段，跳过")

# ═══════════════════════════════════════════════════════════════════════════════
# 图24：运行状态分布
# ═══════════════════════════════════════════════════════════════════════════════
print("生成图24…")
df24 = df_post[["FAN_LINE_SUM_S1", "LINE_TOTAL"]].dropna()

cond_gen = (df24["FAN_LINE_SUM_S1"] > 0) & (df24["LINE_TOTAL"] > 0)
cond_mix = (df24["FAN_LINE_SUM_S1"] > 0) & (df24["LINE_TOTAL"] <= 0)
cond_consume = (df24["FAN_LINE_SUM_S1"] <= 0) & (df24["LINE_TOTAL"] <= 0)
cond_other = (df24["FAN_LINE_SUM_S1"] <= 0) & (df24["LINE_TOTAL"] > 0)

counts = {
    "正常发电": cond_gen.sum(),
    "混合状态": cond_mix.sum(),
    "待机/停机耗电": cond_consume.sum(),
    "其他": cond_other.sum(),
}
total24 = sum(counts.values())

fig24, (ax24a, ax24b) = plt.subplots(1, 2, figsize=(13, 5), dpi=DPI)
fig24.suptitle("运行状态分布（2024-07-19后）", fontsize=14)

pie_labels = [f"{k}\n{v:,}条\n({v/total24*100:.1f}%)" for k, v in counts.items()]
pie_colors = ["steelblue", "orange", "red", "gray"]
ax24a.pie(
    list(counts.values()),
    labels=pie_labels,
    colors=pie_colors,
    autopct=None,
    startangle=90,
    textprops={"fontsize": 9},
)
ax24a.set_title("各运行状态占比")

consume_line = df24[cond_consume]["LINE_TOTAL"]
ax24b.hist(consume_line, bins=60, color="red", alpha=0.7, edgecolor="white")
ax24b.axvline(consume_line.mean(), color="navy", lw=1.5, ls="--",
              label=f"均值={consume_line.mean():.3f} MW")
ax24b.set_xlabel("集电线路合计功率 (MW)")
ax24b.set_ylabel("频次")
ax24b.set_title("待机/停机耗电时 LINE_TOTAL 分布")
ax24b.legend(fontsize=9)

plt.tight_layout()
fig24.savefig(os.path.join(OUT_DIR, "24_operational_state_distribution.png"), dpi=DPI)
plt.close(fig24)
print("✅ fig24.png")

print(f"\n全部图表已保存至：{OUT_DIR}")
