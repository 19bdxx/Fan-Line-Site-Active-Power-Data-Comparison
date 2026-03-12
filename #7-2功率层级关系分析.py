"""
功能说明：
本脚本对峡阳B风电场的 SCADA 数据进行探索性数据分析（EDA），
分析风机功率（FAN）、集电线路功率（LINE）与全站功率（STATION）之间的关系。

主要功能：
1. 加载并合并峡阳B的分部数据文件
2. 计算三个层级功率之间的差值：
   - Fan - Line（风机功率 - 集电线路功率）
   - Line - Station（集电线路功率 - 全站功率）
   - Fan - Station（风机功率 - 全站功率）
3. 基本统计分析（均值、标准差、分位数等）
4. 可视化分析（时间序列、散点图、分布图、差值分析）
5. 输出观察结论与可能原因分析

数据来源：DATA/峡阳B/
输出目录：DATA/峡阳B/analysis_output/
"""

import os
import glob
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# 0. 路径配置
# ─────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "DATA", "峡阳B")
OUTPUT_DIR = os.path.join(DATA_DIR, "analysis_output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 中文字体支持（Linux 环境使用 DejaVu，标签用英文以保证渲染）
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "axes.unicode_minus": False,
})


# ─────────────────────────────────────────────
# 1. 加载并合并数据
# ─────────────────────────────────────────────
def load_data(data_dir: str) -> pd.DataFrame:
    pattern = os.path.join(data_dir, "*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No CSV files found in: {data_dir}")

    dfs = []
    for f in files:
        print(f"  Loading: {os.path.basename(f)}")
        dfs.append(pd.read_csv(f, parse_dates=["timestamp"]))

    df = pd.concat(dfs, ignore_index=True)
    df.sort_values("timestamp", inplace=True)
    df.drop_duplicates(subset="timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ─────────────────────────────────────────────
# 2. 特征工程 — 差值列
# ─────────────────────────────────────────────
def add_diff_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    以 Strategy-2（负值置0）结果作为代表值，计算三层功率差值。
    Strategy-2 更贴近实际运营中的统计口径（负功率视为无效输出）。
    """
    df["FAN"] = df["FAN_ACTIVE_POWER_SUM_S2"]
    df["LINE"] = df["LINE_ACTIVE_POWER_SUM_S2"]
    df["STATION"] = pd.to_numeric(df["ACTIVE_POWER_STATION"], errors="coerce")

    df["DIFF_FAN_LINE"] = df["FAN"] - df["LINE"]          # 风机 - 线路
    df["DIFF_LINE_STATION"] = df["LINE"] - df["STATION"]  # 线路 - 全站
    df["DIFF_FAN_STATION"] = df["FAN"] - df["STATION"]    # 风机 - 全站

    return df


# ─────────────────────────────────────────────
# 3. 基本统计
# ─────────────────────────────────────────────
def print_basic_stats(df: pd.DataFrame) -> None:
    cols = ["FAN", "LINE", "STATION",
            "DIFF_FAN_LINE", "DIFF_LINE_STATION", "DIFF_FAN_STATION"]

    print("\n" + "=" * 70)
    print("Section 1: Basic Statistics (MW)")
    print("=" * 70)
    stats = df[cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
    print(stats.to_string())

    print("\n" + "=" * 70)
    print("Section 2: Missing Value Count")
    print("=" * 70)
    print(df[cols].isnull().sum().to_string())

    print("\n" + "=" * 70)
    print("Section 3: Correlation Matrix")
    print("=" * 70)
    print(df[["FAN", "LINE", "STATION"]].corr().to_string())


# ─────────────────────────────────────────────
# 4. 可视化辅助函数
# ─────────────────────────────────────────────
def save_fig(name: str) -> None:
    path = os.path.join(OUTPUT_DIR, name)
    plt.savefig(path, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ─────────────────────────────────────────────
# 5. 可视化 1 — 时间序列（抽样显示）
# ─────────────────────────────────────────────
def plot_time_series(df: pd.DataFrame, sample_days: int = 30) -> None:
    # 取前 sample_days 天数据避免图形过密
    end_ts = df["timestamp"].min() + pd.Timedelta(days=sample_days)
    sub = df[df["timestamp"] <= end_ts].copy()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # --- 上图：三层功率 ---
    ax = axes[0]
    ax.plot(sub["timestamp"], sub["FAN"], label="Fan Power (S2)", alpha=0.7, lw=0.8)
    ax.plot(sub["timestamp"], sub["LINE"], label="Line Power (S2)", alpha=0.7, lw=0.8)
    ax.plot(sub["timestamp"], sub["STATION"], label="Station Power", alpha=0.9, lw=0.8)
    ax.set_ylabel("Active Power (MW)")
    ax.set_title(f"Time Series of Power Levels (first {sample_days} days)")
    ax.legend(loc="upper right", fontsize=8)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.grid(True, alpha=0.3)

    # --- 下图：差值 ---
    ax2 = axes[1]
    ax2.plot(sub["timestamp"], sub["DIFF_FAN_LINE"],
             label="Fan - Line", alpha=0.7, lw=0.8, color="C3")
    ax2.plot(sub["timestamp"], sub["DIFF_LINE_STATION"],
             label="Line - Station", alpha=0.7, lw=0.8, color="C4")
    ax2.plot(sub["timestamp"], sub["DIFF_FAN_STATION"],
             label="Fan - Station", alpha=0.7, lw=0.8, color="C5")
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.set_ylabel("Power Difference (MW)")
    ax2.set_title("Power Differences Between Levels")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.grid(True, alpha=0.3)

    fig.autofmt_xdate()
    plt.tight_layout()
    save_fig("01_time_series.png")


# ─────────────────────────────────────────────
# 6. 可视化 2 — 散点图
# ─────────────────────────────────────────────
def plot_scatter(df: pd.DataFrame, n_sample: int = 20000) -> None:
    sub = df.dropna(subset=["FAN", "LINE", "STATION"])
    if len(sub) > n_sample:
        sub = sub.sample(n_sample, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    pairs = [
        ("FAN", "LINE", "Fan vs Line Power"),
        ("LINE", "STATION", "Line vs Station Power"),
        ("FAN", "STATION", "Fan vs Station Power"),
    ]

    for ax, (xcol, ycol, title) in zip(axes, pairs):
        ax.scatter(sub[xcol], sub[ycol], alpha=0.15, s=5, color="steelblue")
        # 1:1 参考线
        lim_min = min(sub[xcol].min(), sub[ycol].min())
        lim_max = max(sub[xcol].max(), sub[ycol].max())
        ax.plot([lim_min, lim_max], [lim_min, lim_max],
                "r--", lw=1, label="y = x")
        ax.set_xlabel(f"{xcol} (MW)")
        ax.set_ylabel(f"{ycol} (MW)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Scatter Plots: Power Level Comparison", y=1.02)
    plt.tight_layout()
    save_fig("02_scatter_plots.png")


# ─────────────────────────────────────────────
# 7. 可视化 3 — 差值分布（直方图 + KDE）
# ─────────────────────────────────────────────
def plot_diff_distribution(df: pd.DataFrame) -> None:
    diff_cols = {
        "Fan - Line": "DIFF_FAN_LINE",
        "Line - Station": "DIFF_LINE_STATION",
        "Fan - Station": "DIFF_FAN_STATION",
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (label, col) in zip(axes, diff_cols.items()):
        data = df[col].dropna()
        # 截断极值以便展示主体分布
        p1, p99 = data.quantile(0.01), data.quantile(0.99)
        data_clip = data.clip(p1, p99)

        ax.hist(data_clip, bins=100, color="steelblue", edgecolor="none", alpha=0.7)
        ax.axvline(data.mean(), color="red", lw=1.5, ls="--",
                   label=f"Mean={data.mean():.2f}")
        ax.axvline(data.median(), color="orange", lw=1.5, ls="-.",
                   label=f"Median={data.median():.2f}")
        ax.set_xlabel("Difference (MW)")
        ax.set_ylabel("Count")
        ax.set_title(f"Distribution: {label}")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Distribution of Power Differences", y=1.02)
    plt.tight_layout()
    save_fig("03_diff_distribution.png")


# ─────────────────────────────────────────────
# 8. 可视化 4 — 差值 vs 功率水平（箱线图分段）
# ─────────────────────────────────────────────
def plot_diff_vs_power_level(df: pd.DataFrame) -> None:
    """将风机功率按十等分分段，观察不同功率水平下差值特征"""
    sub = df.dropna(subset=["FAN", "DIFF_FAN_LINE", "DIFF_LINE_STATION", "DIFF_FAN_STATION"]).copy()

    # 按 FAN 功率分10段
    sub["FAN_BIN"] = pd.cut(sub["FAN"], bins=10)

    diff_cols = {
        "Fan - Line": "DIFF_FAN_LINE",
        "Line - Station": "DIFF_LINE_STATION",
        "Fan - Station": "DIFF_FAN_STATION",
    }

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    for ax, (label, col) in zip(axes, diff_cols.items()):
        groups = [g[col].dropna().values for _, g in sub.groupby("FAN_BIN", observed=True)]
        bin_labels = [str(b) for b in sub.groupby("FAN_BIN", observed=True).groups.keys()]

        ax.boxplot(groups, labels=bin_labels, patch_artist=True,
                   boxprops=dict(facecolor="lightsteelblue", alpha=0.7),
                   medianprops=dict(color="red", lw=1.5),
                   flierprops=dict(marker=".", markersize=2, alpha=0.3))
        ax.axhline(0, color="black", lw=0.8, ls="--")
        ax.set_xlabel("Fan Power Bin (MW)")
        ax.set_ylabel("Difference (MW)")
        ax.set_title(f"{label} by Fan Power Level")
        ax.tick_params(axis="x", rotation=30, labelsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    plt.suptitle("Power Difference by Fan Power Level", y=1.01)
    plt.tight_layout()
    save_fig("04_diff_by_power_level.png")


# ─────────────────────────────────────────────
# 9. 可视化 5 — 月度平均功率对比
# ─────────────────────────────────────────────
def plot_monthly_avg(df: pd.DataFrame) -> None:
    sub = df.copy()
    sub["month"] = sub["timestamp"].dt.to_period("M")
    monthly = sub.groupby("month")[["FAN", "LINE", "STATION",
                                    "DIFF_FAN_LINE", "DIFF_LINE_STATION",
                                    "DIFF_FAN_STATION"]].mean()

    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    x = range(len(monthly))
    ax = axes[0]
    ax.bar([i - 0.25 for i in x], monthly["FAN"], width=0.25,
           label="Fan", color="C0", alpha=0.8)
    ax.bar([i for i in x], monthly["LINE"], width=0.25,
           label="Line", color="C1", alpha=0.8)
    ax.bar([i + 0.25 for i in x], monthly["STATION"], width=0.25,
           label="Station", color="C2", alpha=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels([str(m) for m in monthly.index], rotation=30, ha="right")
    ax.set_ylabel("Mean Active Power (MW)")
    ax.set_title("Monthly Average Power by Level")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax2 = axes[1]
    ax2.plot(list(x), monthly["DIFF_FAN_LINE"],
             marker="o", label="Fan - Line", color="C3")
    ax2.plot(list(x), monthly["DIFF_LINE_STATION"],
             marker="s", label="Line - Station", color="C4")
    ax2.plot(list(x), monthly["DIFF_FAN_STATION"],
             marker="^", label="Fan - Station", color="C5")
    ax2.axhline(0, color="black", lw=0.8, ls="--")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels([str(m) for m in monthly.index], rotation=30, ha="right")
    ax2.set_ylabel("Mean Difference (MW)")
    ax2.set_title("Monthly Average Power Differences")
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    save_fig("05_monthly_avg.png")


# ─────────────────────────────────────────────
# 10. 可视化 6 — 各线路风机汇总 vs 线路测点对比
# ─────────────────────────────────────────────
def plot_line_vs_fan_group(df: pd.DataFrame) -> None:
    """比较每条集电线路的测点功率 vs 对应风机功率汇总（S2策略）"""
    line_pairs = [
        ("ACTIVE_POWER_BING", "BING_ACTIVE_POWER_SUM_S2", "BING Line"),
        ("ACTIVE_POWER_DING", "DING_ACTIVE_POWER_SUM_S2", "DING Line"),
        ("ACTIVE_POWER_WU",  "WU_ACTIVE_POWER_SUM_S2",  "WU Line"),
    ]

    n_sample = 15000
    sub = df.dropna(subset=[p[0] for p in line_pairs] + [p[1] for p in line_pairs])
    if len(sub) > n_sample:
        sub = sub.sample(n_sample, random_state=42)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, (line_col, fan_col, label) in zip(axes, line_pairs):
        x = pd.to_numeric(sub[line_col], errors="coerce")
        y = pd.to_numeric(sub[fan_col], errors="coerce")
        ax.scatter(x, y, alpha=0.15, s=5, color="teal")
        lim = max(x.max(), y.max())
        ax.plot([0, lim], [0, lim], "r--", lw=1, label="y = x")
        # 差值
        diff = (y - x).dropna()
        ax.set_xlabel(f"Line Measurement (MW)\n[{line_col}]")
        ax.set_ylabel(f"Fan Group Sum (MW)\n[{fan_col}]")
        ax.set_title(f"{label}\nMean diff (Fan-Line)={diff.mean():.2f} MW")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Per-Line: Fan Group Sum vs Line Measurement Power", y=1.02)
    plt.tight_layout()
    save_fig("06_per_line_comparison.png")


# ─────────────────────────────────────────────
# 11. 可视化 7 — 全站功率为0时的异常分析
# ─────────────────────────────────────────────
def plot_station_zero_analysis(df: pd.DataFrame) -> None:
    """当全站功率为0但风机/线路有功率时，分析差异情况（疑似数据质量问题）"""
    station_zero = df[(df["STATION"] == 0) & (df["FAN"] > 1)].copy()
    normal = df[(df["STATION"] > 1) & (df["FAN"] > 1)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    n_zero = len(station_zero)
    n_total = len(df[df["FAN"] > 1])
    ax.pie(
        [n_zero, n_total - n_zero],
        labels=[f"Station=0 but Fan>1MW\n(n={n_zero})",
                f"Station>1 MW\n(n={n_total - n_zero})"],
        autopct="%1.1f%%",
        startangle=90,
        colors=["tomato", "steelblue"],
    )
    ax.set_title("Station=0 Anomaly Proportion\n(when Fan Power > 1 MW)")

    ax2 = axes[1]
    if len(station_zero) > 0:
        ax2.hist(station_zero["FAN"].clip(0, station_zero["FAN"].quantile(0.99)),
                 bins=60, color="tomato", alpha=0.7,
                 label=f"Station=0 (n={len(station_zero)})")
    ax2.hist(normal["FAN"].clip(0, normal["FAN"].quantile(0.99)),
             bins=60, color="steelblue", alpha=0.5,
             label=f"Normal (n={len(normal)})")
    ax2.set_xlabel("Fan Power (MW)")
    ax2.set_ylabel("Count")
    ax2.set_title("Fan Power Distribution:\nStation=0 vs Normal Records")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    save_fig("07_station_zero_anomaly.png")


# ─────────────────────────────────────────────
# 12. 可视化 8 — 策略1 vs 策略2 差异
# ─────────────────────────────────────────────
def plot_strategy_comparison(df: pd.DataFrame) -> None:
    """对比 S1（含负值）与 S2（负值置0）两种统计策略"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (col_s1, col_s2, label) in zip(axes, [
        ("FAN_ACTIVE_POWER_SUM_S1", "FAN_ACTIVE_POWER_SUM_S2", "Fan Power"),
        ("LINE_ACTIVE_POWER_SUM_S1", "LINE_ACTIVE_POWER_SUM_S2", "Line Power"),
    ]):
        diff = df[col_s2] - df[col_s1]
        ax.hist(diff.dropna(), bins=80, color="purple", alpha=0.7)
        ax.axvline(diff.mean(), color="red", lw=1.5, ls="--",
                   label=f"Mean={diff.mean():.3f}")
        ax.set_xlabel("S2 - S1 (MW)")
        ax.set_ylabel("Count")
        ax.set_title(f"{label}: Strategy Difference (S2 - S1)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("Strategy-2 vs Strategy-1: Impact of Clipping Negative Values",
                 y=1.02)
    plt.tight_layout()
    save_fig("08_strategy_comparison.png")


# ─────────────────────────────────────────────
# 13. 输出分析报告（文本）
# ─────────────────────────────────────────────
def print_findings(df: pd.DataFrame) -> None:
    total = len(df)
    fan = df["FAN"]
    line = df["LINE"]
    station = df["STATION"]

    d_fl = df["DIFF_FAN_LINE"]
    d_ls = df["DIFF_LINE_STATION"]
    d_fs = df["DIFF_FAN_STATION"]

    station_zero_fan_positive = ((station == 0) & (fan > 1)).sum()
    large_diff_pct = (d_fs.abs() > 10).sum() / total * 100

    report = f"""
================================================================================
Analysis Report: Fan / Line / Station Power Relationship
Site: 峡阳B (Xia Yang B)  |  Records: {total:,}
================================================================================

[Power Level Statistics]
  FAN   | Mean={fan.mean():.2f} MW  Std={fan.std():.2f} MW  Range=[{fan.min():.2f}, {fan.max():.2f}]
  LINE  | Mean={line.mean():.2f} MW  Std={line.std():.2f} MW  Range=[{line.min():.2f}, {line.max():.2f}]
  STATION | Mean={station.mean():.2f} MW  Std={station.std():.2f} MW  Range=[{station.min():.2f}, {station.max():.2f}]

[Difference Statistics]
  Fan - Line    | Mean={d_fl.mean():.3f}  Median={d_fl.median():.3f}  Std={d_fl.std():.3f}
  Line - Station| Mean={d_ls.mean():.3f}  Median={d_ls.median():.3f}  Std={d_ls.std():.3f}
  Fan - Station | Mean={d_fs.mean():.3f}  Median={d_fs.median():.3f}  Std={d_fs.std():.3f}

[Key Observations]
  1. Fan > Line > Station hierarchy:
     - Fan power is generally higher than line measurement power (mean diff = {d_fl.mean():.2f} MW),
       suggesting turbine-level metering captures slightly more power than line instruments.
     - Line power exceeds station power on average (mean diff = {d_ls.mean():.2f} MW),
       consistent with auxiliary power consumption and line losses before the station meter.

  2. Station=0 anomaly:
     - {station_zero_fan_positive:,} records ({station_zero_fan_positive/total*100:.2f}%) have STATION=0
       while Fan power > 1 MW. This strongly indicates a data quality issue —
       the station active power measurement is missing or zeroed out while turbines
       were actually generating power.

  3. Large discrepancy (|Fan - Station| > 10 MW): {large_diff_pct:.1f}% of all records.
     When present at low production levels, this often reflects auxiliary consumption.
     At high production levels, it may indicate metering instrument errors.

  4. Strategy impact (S1 vs S2):
     - The difference between sum-with-negatives (S1) and clip-at-zero (S2) is small
       for fan power but more notable for line power (negative readings from line
       instruments can occur during low-wind or reverse-flow conditions).

[Possible Causes of Power Differences]
  Fan - Line:
    - Measurement position: Fan meters are at nacelle; line meters are at substation busbar.
    - Cable and transformer losses from turbine to collection line.
    - Timing offsets between fan SCADA and line SCADA.

  Line - Station:
    - Auxiliary (station service) power consumption (lighting, cooling, controls).
    - Line losses in the collection system before the station meter.
    - Differences in measurement locations (high/low voltage side of main transformer).

  Station power = 0 anomalies:
    - Likely SCADA data recording gaps or instrument failure.
    - Could also occur during grid-disconnected tests or forced outages where
      station meter resets while turbines briefly spin.

================================================================================
"""
    print(report)

    report_path = os.path.join(OUTPUT_DIR, "analysis_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  Report saved: {report_path}")


# ─────────────────────────────────────────────
# 主程序
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 70)
    print("Power Level Relationship Analysis — 峡阳B")
    print("=" * 70)

    print("\n[1] Loading data ...")
    df = load_data(DATA_DIR)
    print(f"    Total records loaded: {len(df):,}")

    print("\n[2] Computing difference columns ...")
    df = add_diff_columns(df)

    print("\n[3] Basic statistics ...")
    print_basic_stats(df)

    print("\n[4] Generating visualizations ...")
    print("  4.1 Time series ...")
    plot_time_series(df)

    print("  4.2 Scatter plots ...")
    plot_scatter(df)

    print("  4.3 Difference distributions ...")
    plot_diff_distribution(df)

    print("  4.4 Difference by power level ...")
    plot_diff_vs_power_level(df)

    print("  4.5 Monthly average ...")
    plot_monthly_avg(df)

    print("  4.6 Per-line comparison ...")
    plot_line_vs_fan_group(df)

    print("  4.7 Station=0 anomaly ...")
    plot_station_zero_analysis(df)

    print("  4.8 Strategy comparison ...")
    plot_strategy_comparison(df)

    print("\n[5] Findings & Report ...")
    print_findings(df)

    print("\nAll done. Outputs saved in:", OUTPUT_DIR)
