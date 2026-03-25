#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
集电线路传输损耗拟合工具（含 Dash 交互版）
================================================
新增功能：
    1. Dash 交互页面，可绘制“功率-损耗”散点图
    2. 散点区分为“正常点 / 异常点”
    3. 叠加拟合曲线
    4. 支持选择不同集电线路（丙 / 丁 / 戊）
    5. 支持切换两种拟合方案：
       - 方案 A：P = max(CT, 0)，L̂(P)=a·P²+b·P+c
       - 方案 B：P = FAN_SUM_S2，L̂(P)=a·P²+b·P+c（仅阈值以上参与拟合）

运行方式：
    1) 只生成静态结果（与原脚本类似）
       python 集电线路损耗拟合_DASH版.py

    2) 启动 Dash 页面
       python 集电线路损耗拟合_DASH版.py --dash

    3) 指定端口
       python 集电线路损耗拟合_DASH版.py --dash --port 8051
"""

from __future__ import annotations

import os
import glob
import math
import argparse
import warnings
from typing import Dict, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

warnings.filterwarnings("ignore")

# Dash / Plotly
from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go


# ══════════════════════════════════════════════════════════════
# 0. 字体 & 路径配置
# ══════════════════════════════════════════════════════════════
_fm._load_fontmanager(try_read_cache=False)
_CJK_FONT = next(
    (
        f.name
        for f in _fm.fontManager.ttflist
        if any(
            k in f.name
            for k in (
                "Noto Sans CJK",
                "Noto Serif CJK",
                "WenQuanYi",
                "SimHei",
                "Microsoft YaHei",
            )
        )
    ),
    None,
)
if _CJK_FONT:
    matplotlib.rcParams["font.family"] = _CJK_FONT
    matplotlib.rcParams["axes.unicode_minus"] = False
else:
    matplotlib.rcParams["axes.unicode_minus"] = False
    print("⚠️ 未找到中文字体，Matplotlib 中文可能显示为方块。")

_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
SCADA_DIR = os.path.join(_ROOT_DIR, "DATA", "峡阳B")
OUT_FIG_DIR = os.path.join(_ROOT_DIR, "DATA", "峡阳B", "analysis_output")
OUT_DATA_DIR = os.path.join(_ROOT_DIR, "分析结果", "数据质量分析", "峡阳B")

os.makedirs(OUT_FIG_DIR, exist_ok=True)
os.makedirs(OUT_DATA_DIR, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1. 集电线路列映射
# ══════════════════════════════════════════════════════════════
LINE_CT_COL = {
    "BING": "ACTIVE_POWER_BING",
    "DING": "ACTIVE_POWER_DING",
    "WU": "ACTIVE_POWER_WU",
}
LINE_FAN_COL = {
    "BING": "BING_ACTIVE_POWER_SUM_S2",
    "DING": "DING_ACTIVE_POWER_SUM_S2",
    "WU": "WU_ACTIVE_POWER_SUM_S2",
}
LINE_NAME_ZH = {
    "BING": "丙（BING）",
    "DING": "丁（DING）",
    "WU": "戊（WU）",
}
LINE_COLOR = {
    "BING": "#ff7f0e",
    "DING": "#2ca02c",
    "WU": "#1f77b4",
}

SCADA_COLS = ["timestamp"] + list(LINE_CT_COL.values()) + list(LINE_FAN_COL.values())
N_BINS = 15
MAX_SCATTER_POINTS = 12000

# 方案B：各集电线路仅在阈值以上参与拟合（单位：MW）
LINE_FAN_THRESHOLD_B = {
    "BING": 3.5,
    "DING": 3.0,
    "WU": 4.0,
}

EXCLUDE_TYPES = {
    "第一类-全场通讯中断",
    "第二类-部分通讯中断",
    "第三类-单机通讯故障",
    "第三类-发电状态零值",
    "第三类-单机非零卡值",
    "正常停机-保留",
    "零值-状态待核实",
}


# ══════════════════════════════════════════════════════════════
# 2. 数据加载 / 异常区间 / 拟合
# ══════════════════════════════════════════════════════════════
def get_line_by_fan_num(fan_num):
    if pd.isna(fan_num):
        return "未知"
    fan_num = int(fan_num)
    if 63 <= fan_num <= 109:
        return "WU"
    if 110 <= fan_num <= 152:
        return "DING"
    if 153 <= fan_num <= 199:
        return "BING"
    return "未知"


def build_exclude_mask(scada_ts: np.ndarray, segs_df: pd.DataFrame) -> np.ndarray:
    """返回 bool 数组：True 表示该时间戳属于某个异常段。"""
    mask = np.zeros(len(scada_ts), dtype=bool)
    if segs_df is None or len(segs_df) == 0:
        return mask
    ts_np = scada_ts.astype("datetime64[ns]")
    starts = segs_df["开始时间"].values.astype("datetime64[ns]")
    ends = segs_df["结束时间"].values.astype("datetime64[ns]")
    for t0, t1 in zip(starts, ends):
        mask |= (ts_np >= t0) & (ts_np <= t1)
    return mask


def fit_loss_model_ct(p_vals: np.ndarray, l_vals: np.ndarray, n_bins: int = N_BINS):
    valid = (p_vals > 0) & np.isfinite(p_vals) & np.isfinite(l_vals)
    p = p_vals[valid]
    l = l_vals[valid]
    if len(p) < 5:
        return np.array([np.nan, np.nan, np.nan]), pd.DataFrame(columns=["P_med", "L_med", "n"]), np.nan, np.nan

    tmp = pd.DataFrame({"_P": p, "_L": l})
    tmp["_bin"] = pd.qcut(tmp["_P"], q=min(n_bins, max(2, len(tmp) // 5)), duplicates="drop")
    bins_df = tmp.groupby("_bin", observed=True).agg(
        P_med=("_P", "median"), L_med=("_L", "median"), n=("_P", "count")
    ).reset_index(drop=True)

    if len(bins_df) < 3:
        return np.array([np.nan, np.nan, np.nan]), bins_df, np.nan, np.nan

    coeffs = np.polyfit(bins_df["P_med"].values, bins_df["L_med"].values, deg=2)
    l_hat_bins = np.polyval(coeffs, bins_df["P_med"].values)
    ss_res = np.sum((bins_df["L_med"].values - l_hat_bins) ** 2)
    ss_tot = np.sum((bins_df["L_med"].values - bins_df["L_med"].mean()) ** 2)
    r2_bins = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mask2 = p > 2
    sigma = float(np.std(l[mask2] - np.polyval(coeffs, p[mask2]))) if mask2.sum() > 0 else np.nan
    return coeffs, bins_df, r2_bins, sigma


def fit_loss_model_fan(p_vals: np.ndarray, l_vals: np.ndarray, n_bins: int = N_BINS, p_min: float = 0.0):
    valid = (p_vals > max(0.0, p_min)) & np.isfinite(p_vals) & np.isfinite(l_vals)
    p = p_vals[valid]
    l = l_vals[valid]
    if len(p) < 5:
        return np.array([np.nan, np.nan, np.nan]), pd.DataFrame(columns=["P_med", "L_med", "n"]), np.nan, np.nan

    tmp = pd.DataFrame({"_P": p, "_L": l})
    tmp["_bin"] = pd.qcut(tmp["_P"], q=min(n_bins, max(3, len(tmp) // 5)), duplicates="drop")
    bins_df = tmp.groupby("_bin", observed=True).agg(
        P_med=("_P", "median"), L_med=("_L", "median"), n=("_P", "count")
    ).reset_index(drop=True)

    if len(bins_df) < 3:
        return np.array([np.nan, np.nan, np.nan]), bins_df, np.nan, np.nan

    coeffs = np.polyfit(bins_df["P_med"].values, bins_df["L_med"].values, deg=2)
    l_hat_bins = np.polyval(coeffs, bins_df["P_med"].values)
    ss_res = np.sum((bins_df["L_med"].values - l_hat_bins) ** 2)
    ss_tot = np.sum((bins_df["L_med"].values - bins_df["L_med"].mean()) ** 2)
    r2_bins = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    mask2 = p > max(2, p_min)
    sigma = float(np.std(l[mask2] - np.polyval(coeffs, p[mask2]))) if mask2.sum() > 0 else np.nan
    return coeffs, bins_df, r2_bins, sigma


def load_scada() -> pd.DataFrame:
    print("=" * 65)
    print("  加载 SCADA 合并数据 ...")
    print("=" * 65)
    csv_files = sorted(glob.glob(os.path.join(SCADA_DIR, "*with_sum*.csv")))
    if not csv_files:
        raise FileNotFoundError(
            f"未找到 SCADA 合并 CSV 文件（*with_sum*.csv）\n搜索路径：{SCADA_DIR}"
        )

    parts = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, usecols=SCADA_COLS, parse_dates=["timestamp"])
            parts.append(df)
            print(f"  ✅ {os.path.basename(f)} ({len(df):,} 条)")
        except Exception as e:
            print(f"  ⚠️ 跳过 {os.path.basename(f)}: {e}")

    if not parts:
        raise RuntimeError("SCADA 文件都读取失败，无法继续。")

    scada = pd.concat(parts, ignore_index=True)
    scada.sort_values("timestamp", inplace=True)
    scada.drop_duplicates(subset="timestamp", inplace=True)
    scada.reset_index(drop=True, inplace=True)
    scada["timestamp"] = pd.to_datetime(scada["timestamp"])

    print(
        f"\n  合并后: {len(scada):,} 条 "
        f"({scada['timestamp'].min().date()} ~ {scada['timestamp'].max().date()})"
    )
    return scada


def load_anomaly_segments() -> pd.DataFrame:
    print("\n  加载异常段分类结果 ...")
    classified_csv = os.path.join(OUT_DATA_DIR, "fan_repeat_classified.csv")
    raw_excel = os.path.join(_ROOT_DIR, "联合重复值检测结果.xlsx")

    if os.path.exists(classified_csv):
        anom_df = pd.read_csv(classified_csv, encoding="utf-8-sig")
        anom_df.rename(columns={"异常类型": "anomaly_type"}, inplace=True)
        print(f"  ✅ 已读取 fan_repeat_classified.csv（{len(anom_df):,} 条）")
    elif os.path.exists(raw_excel):
        anom_df = pd.read_excel(raw_excel, engine="openpyxl")
        anom_df["anomaly_type"] = "第三类-单机非零卡值"
        print(f"  ✅ 已读取 联合重复值检测结果.xlsx（{len(anom_df):,} 条）")
    else:
        print("  ⚠️ 未找到异常段分类文件，将不排除异常段。")
        anom_df = pd.DataFrame(columns=["开始时间", "结束时间", "anomaly_type"])

    if len(anom_df) > 0:
        anom_df["开始时间"] = pd.to_datetime(anom_df["开始时间"])
        anom_df["结束时间"] = pd.to_datetime(anom_df["结束时间"])
        if "风机编号" in anom_df.columns:
            anom_df["line"] = anom_df["风机编号"].apply(get_line_by_fan_num)
        elif "line" not in anom_df.columns:
            anom_df["line"] = "未知"

        if "anomaly_type" in anom_df.columns:
            anom_df = anom_df[anom_df["anomaly_type"].isin(EXCLUDE_TYPES)].copy()

    return anom_df


def sample_df(df: pd.DataFrame, n: int = MAX_SCATTER_POINTS, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).sort_values("timestamp")


def prepare_line_data(scada: pd.DataFrame, anom_df: pd.DataFrame) -> Dict[str, dict]:
    print("\n" + "=" * 65)
    print("  开始拟合并准备 Dash 数据")
    print("=" * 65)

    results: Dict[str, dict] = {}
    ts_np = scada["timestamp"].values.astype("datetime64[ns]")

    for line in ["BING", "DING", "WU"]:
        ct_col = LINE_CT_COL[line]
        fan_col = LINE_FAN_COL[line]
        zh_name = LINE_NAME_ZH[line]
        print(f"\n{'─' * 60}\n  {zh_name}\n{'─' * 60}")

        if len(anom_df) > 0 and "line" in anom_df.columns:
            segs = anom_df.loc[anom_df["line"] == line, ["开始时间", "结束时间"]].drop_duplicates()
        else:
            segs = pd.DataFrame(columns=["开始时间", "结束时间"])

        exc_mask = build_exclude_mask(ts_np, segs)
        line_df = scada[["timestamp", ct_col, fan_col]].copy()
        line_df.rename(columns={ct_col: "CT", fan_col: "FAN_SUM_S2"}, inplace=True)
        line_df["is_anomaly"] = exc_mask
        line_df["CT_eff"] = line_df["CT"].clip(lower=0)
        line_df["FAN_eff"] = line_df["FAN_SUM_S2"].clip(lower=0)
        line_df["L"] = line_df["FAN_eff"] - line_df["CT_eff"]
        line_df["is_shutdown"] = (line_df["CT"].abs() < 0.1) & (line_df["FAN_SUM_S2"].abs() < 0.1)

        normal_df = line_df[(~line_df["is_anomaly"]) & (~line_df["is_shutdown"])].copy()
        anomaly_df = line_df[(line_df["is_anomaly"]) & (~line_df["is_shutdown"])].copy()

        p_ct = normal_df["CT_eff"].values
        p_fan = normal_df["FAN_eff"].values
        l = normal_df["L"].values

        fan_threshold_b = float(LINE_FAN_THRESHOLD_B.get(line, 0.0))
        b_fit_df = normal_df[normal_df["FAN_eff"] > fan_threshold_b].copy()
        b_filtered_out_df = normal_df[normal_df["FAN_eff"] <= fan_threshold_b].copy()

        coeffs_a, bins_a, r2_a, sigma_a = fit_loss_model_ct(p_ct, l)
        coeffs_b, bins_b, r2_b, sigma_b = fit_loss_model_fan(p_fan, l, p_min=fan_threshold_b)

        print(f"  正常点: {len(normal_df):,} 条 | 异常点: {len(anomaly_df):,} 条")
        print(f"  方案A: R²={r2_a:.4f} σ={sigma_a:.4f} coeffs={coeffs_a}")
        print(
            f"  方案B: 仅使用 FAN_SUM_S2>{fan_threshold_b:.2f}MW 的正常点参与拟合 "
            f"({len(b_fit_df):,}/{len(normal_df):,} 条)，R²={r2_b:.4f} σ={sigma_b:.4f} coeffs={coeffs_b}"
        )

        results[line] = {
            "all_df": line_df,
            "normal_df": normal_df,
            "anomaly_df": anomaly_df,
            "normal_sample": sample_df(normal_df),
            "anomaly_sample": sample_df(anomaly_df),
            "coeffs_a": coeffs_a,
            "coeffs_b": coeffs_b,
            "fan_threshold_b": fan_threshold_b,
            "b_fit_df": b_fit_df,
            "b_filtered_out_df": b_filtered_out_df,
            "bins_a": bins_a,
            "bins_b": bins_b,
            "r2_a": r2_a,
            "r2_b": r2_b,
            "sigma_a": sigma_a,
            "sigma_b": sigma_b,
        }

    return results


# ══════════════════════════════════════════════════════════════
# 3. 静态输出
# ══════════════════════════════════════════════════════════════
def save_comparison_csv(results: Dict[str, dict]):
    p_refs = [20, 50, 100, 150, 200, 250]
    rows = []
    for line in ["BING", "DING", "WU"]:
        r = results[line]
        ca, cb = r["coeffs_a"], r["coeffs_b"]
        row_a = {
            "集电线路": LINE_NAME_ZH[line],
            "方案": "A（含截距）",
            "自变量P": "max(CT,0)",
            "系数a": ca[0],
            "系数b": ca[1],
            "系数c": ca[2],
            "L̂(0)[MW]": ca[2],
            "R²(箱中位数)": r["r2_a"],
            "σ_P>2MW[MW]": r["sigma_a"],
        }
        row_b = {
            "集电线路": LINE_NAME_ZH[line],
            "方案": "B（含截距，阈值筛选）",
            "自变量P": "FAN_SUM_S2（仅阈值以上参与拟合）",
            "系数a": cb[0],
            "系数b": cb[1],
            "系数c": cb[2],
            "L̂(0)[MW]": cb[2],
            "R²(箱中位数)": r["r2_b"],
            "σ_P>2MW[MW]": r["sigma_b"],
            "拟合阈值P_min[MW]": r["fan_threshold_b"],
            "参与拟合点数": len(r["b_fit_df"]),
            "筛除点数": len(r["b_filtered_out_df"]),
        }
        for p in p_refs:
            row_a[f"L̂({p}MW)"] = float(np.polyval(ca, p)) if np.all(np.isfinite(ca)) else np.nan
            row_b[f"L̂({p}MW)"] = float(np.polyval(cb, p)) if np.all(np.isfinite(cb)) else np.nan
        rows.extend([row_a, row_b])

    out_csv = os.path.join(OUT_DATA_DIR, "loss_fit_comparison.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"\n✅ 对比结果表已保存：{out_csv}")


def save_static_png(results: Dict[str, dict]):
    print("\n生成静态对比图 loss_fit_comparison.png ...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("集电线路传输损耗拟合对比（正常/异常散点 + 拟合曲线）", fontsize=13, y=1.02)

    for ax, line in zip(axes, ["BING", "DING", "WU"]):
        r = results[line]
        normal = r["normal_sample"]
        anomaly = r["anomaly_sample"]

        # 使用方案B的横轴（FAN）作静态展示，更贴近“输入功率-损耗”关系
        ax.scatter(
            normal["FAN_eff"], normal["L"], s=5, alpha=0.18,
            color="#1f77b4", label="正常点"
        )
        if len(anomaly) > 0:
            ax.scatter(
                anomaly["FAN_eff"], anomaly["L"], s=7, alpha=0.35,
                color="#d62728", label="异常点"
            )
        if len(r["b_filtered_out_df"]) > 0:
            filt = sample_df(r["b_filtered_out_df"])
            ax.scatter(
                filt["FAN_eff"], filt["L"], s=5, alpha=0.15,
                color="#7f7f7f", label=f"低于阈值点(P≤{r['fan_threshold_b']:.1f}MW)"
            )
        if len(r["bins_b"]) > 0:
            ax.scatter(
                r["bins_b"]["P_med"], r["bins_b"]["L_med"], s=38,
                color="#ff7f0e", marker="s", zorder=5, label="分箱中位数"
            )

        x_max = max(float(normal["FAN_eff"].max()) if len(normal) else 1.0,
                    float(anomaly["FAN_eff"].max()) if len(anomaly) else 1.0)
        x = np.linspace(0, max(1.0, x_max * 1.05), 300)
        if np.all(np.isfinite(r["coeffs_b"])):
            ax.plot(
                x, np.polyval(r["coeffs_b"], x), lw=2.2, color="#ff7f0e",
                label=f"方案B拟合(阈值>{r['fan_threshold_b']:.1f}MW) R²={r['r2_b']:.3f} σ={r['sigma_b']:.3f}"
            )

        ax.set_title(LINE_NAME_ZH[line])
        ax.set_xlabel("功率 P（MW）")
        ax.set_ylabel("损耗 L（MW）")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_png = os.path.join(OUT_FIG_DIR, "loss_fit_comparison.png")
    plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ 图表已保存：{out_png}")


# ══════════════════════════════════════════════════════════════
# 4. Dash 图表
# ══════════════════════════════════════════════════════════════
def format_formula(coeffs: np.ndarray, scheme: str) -> str:
    if coeffs is None or not np.all(np.isfinite(coeffs)):
        return "拟合失败或数据不足"
    a, b, c = coeffs
    if scheme == "A":
        sign_b = "+" if b >= 0 else "-"
        sign_c = "+" if c >= 0 else "-"
        return f"L̂(P) = {a:.4e}·P² {sign_b} {abs(b):.5f}·P {sign_c} {abs(c):.4f}"
    sign_b = "+" if b >= 0 else "-"
    sign_c = "+" if c >= 0 else "-"
    return f"L̂(P) = {a:.4e}·P² {sign_b} {abs(b):.5f}·P {sign_c} {abs(c):.4f}"


def make_scatter_figure(results: Dict[str, dict], line: str, scheme: str) -> Tuple[go.Figure, str]:
    r = results[line]
    normal = r["normal_sample"]
    anomaly = r["anomaly_sample"]
    b_filtered_out = sample_df(r["b_filtered_out_df"]) if len(r.get("b_filtered_out_df", [])) else pd.DataFrame(columns=normal.columns)

    if scheme == "A":
        x_col = "CT_eff"
        coeffs = r["coeffs_a"]
        bins = r["bins_a"]
        r2 = r["r2_a"]
        sigma = r["sigma_a"]
        x_title = "功率 P = max(CT, 0)（MW）"
        curve_name = "方案A拟合曲线"
    else:
        x_col = "FAN_eff"
        coeffs = r["coeffs_b"]
        bins = r["bins_b"]
        r2 = r["r2_b"]
        sigma = r["sigma_b"]
        x_title = "功率 P = FAN_SUM_S2（MW）"
        curve_name = f"方案B拟合曲线（仅P>{r['fan_threshold_b']:.1f}MW参与拟合）"

    fig = go.Figure()

    n_df = normal[(normal[x_col] > 0) & np.isfinite(normal[x_col]) & np.isfinite(normal["L"])].copy()
    a_df = anomaly[(anomaly[x_col] > 0) & np.isfinite(anomaly[x_col]) & np.isfinite(anomaly["L"])].copy()

    fig.add_trace(go.Scatter(
        x=n_df[x_col], y=n_df["L"], mode="markers", name="正常点",
        marker=dict(size=5, color="#1f77b4", opacity=0.35),
        hovertemplate="时间=%{customdata[0]}<br>P=%{x:.3f} MW<br>L=%{y:.3f} MW<extra>正常点</extra>",
        customdata=np.stack([n_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")], axis=-1) if len(n_df) else None,
    ))

    fig.add_trace(go.Scatter(
        x=a_df[x_col], y=a_df["L"], mode="markers", name="异常点",
        marker=dict(size=6, color="#d62728", opacity=0.55, symbol="circle-open"),
        hovertemplate="时间=%{customdata[0]}<br>P=%{x:.3f} MW<br>L=%{y:.3f} MW<extra>异常点</extra>",
        customdata=np.stack([a_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")], axis=-1) if len(a_df) else None,
    ))

    if scheme == "B" and len(b_filtered_out) > 0:
        bf_df = b_filtered_out[(b_filtered_out[x_col] > 0) & np.isfinite(b_filtered_out[x_col]) & np.isfinite(b_filtered_out["L"])].copy()
        fig.add_trace(go.Scatter(
            x=bf_df[x_col], y=bf_df["L"], mode="markers", name=f"低于阈值点(P≤{r['fan_threshold_b']:.1f}MW)",
            marker=dict(size=5, color="#7f7f7f", opacity=0.22),
            hovertemplate="时间=%{customdata[0]}<br>P=%{x:.3f} MW<br>L=%{y:.3f} MW<extra>低于阈值点</extra>",
            customdata=np.stack([bf_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")], axis=-1) if len(bf_df) else None,
        ))

    if len(bins) > 0:
        fig.add_trace(go.Scatter(
            x=bins["P_med"], y=bins["L_med"], mode="markers",
            name="分箱中位数", marker=dict(size=9, color="#ff7f0e", symbol="diamond")
        ))

    x_max_candidates = []
    if len(n_df):
        x_max_candidates.append(float(n_df[x_col].max()))
    if len(a_df):
        x_max_candidates.append(float(a_df[x_col].max()))
    if len(bins):
        x_max_candidates.append(float(bins["P_med"].max()))
    x_max = max(x_max_candidates) if x_max_candidates else 1.0
    x_curve = np.linspace(0, max(1.0, x_max * 1.05), 300)

    if coeffs is not None and np.all(np.isfinite(coeffs)):
        y_curve = np.polyval(coeffs, x_curve)
        fig.add_trace(go.Scatter(
            x=x_curve, y=y_curve, mode="lines", name=curve_name,
            line=dict(width=3, color="#2ca02c")
        ))

    fig.update_layout(
        title=f"{LINE_NAME_ZH[line]} 功率-损耗散点图（{('方案A' if scheme == 'A' else '方案B')}）",
        template="plotly_white",
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=60, r=30, t=80, b=60),
    )
    fig.update_xaxes(title_text=x_title, rangemode="tozero")
    fig.update_yaxes(title_text="损耗 L（MW）")

    if coeffs is None or not np.all(np.isfinite(coeffs)):
        info = f"{LINE_NAME_ZH[line]}：数据不足，无法完成当前方案拟合。"
    else:
        if scheme == "A":
            info = (
                f"{LINE_NAME_ZH[line]} | 方案A | R²={r2:.4f} | σ={sigma:.4f} MW | "
                f"参与拟合点数={len(r['normal_df']):,} | {format_formula(coeffs, scheme)}"
            )
        else:
            info = (
                f"{LINE_NAME_ZH[line]} | 方案B | 阈值P>{r['fan_threshold_b']:.2f} MW | "
                f"R²={r2:.4f} | σ={sigma:.4f} MW | 参与拟合点数={len(r['b_fit_df']):,} | "
                f"筛除低功率点={len(r['b_filtered_out_df']):,} | {format_formula(coeffs, scheme)}"
            )
    return fig, info


def build_dash_app(results: Dict[str, dict]) -> Dash:
    app = Dash(__name__)
    app.title = "集电线路损耗拟合 DASH"

    app.layout = html.Div(
        style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px", "fontFamily": "Arial, Microsoft YaHei, sans-serif"},
        children=[
            html.H2("集电线路功率-损耗拟合分析 DASH"),
            html.Div(
                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "12px"},
                children=[
                    html.Div([
                        html.Label("选择集电线路"),
                        dcc.Dropdown(
                            id="line-dropdown",
                            options=[
                                {"label": LINE_NAME_ZH[k], "value": k}
                                for k in ["BING", "DING", "WU"]
                            ],
                            value="BING",
                            clearable=False,
                            style={"width": "260px"},
                        ),
                    ]),
                    html.Div([
                        html.Label("选择拟合方案"),
                        dcc.RadioItems(
                            id="scheme-radio",
                            options=[
                                {"label": "方案A：P=max(CT,0)，含截距", "value": "A"},
                                {"label": "方案B：P=FAN_SUM_S2，含截距，且仅阈值以上参与拟合", "value": "B"},
                            ],
                            value="B",
                            inline=False,
                        ),
                    ]),
                ],
            ),
            html.Div(
                id="fit-info",
                style={
                    "background": "#f7f7f7",
                    "padding": "10px 12px",
                    "border": "1px solid #ddd",
                    "borderRadius": "6px",
                    "marginBottom": "12px",
                    "whiteSpace": "pre-wrap",
                },
            ),
            dcc.Graph(id="loss-scatter-graph", style={"height": "760px"}),
            html.Div(
                "说明：蓝色为正常点，红色为异常时段点；方案B下灰色为低于阈值、仅展示不参与拟合的点；橙色为分箱中位数，绿色为拟合曲线。",
                style={"color": "#666", "marginTop": "8px"},
            ),
        ],
    )

    @app.callback(
        Output("loss-scatter-graph", "figure"),
        Output("fit-info", "children"),
        Input("line-dropdown", "value"),
        Input("scheme-radio", "value"),
    )
    def update_graph(line, scheme):
        return make_scatter_figure(results, line, scheme)

    return app


# ══════════════════════════════════════════════════════════════
# 5. 主流程
# ══════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="集电线路损耗拟合工具（支持 Dash）")
    parser.add_argument("--dash", action="store_true", help="启动 Dash 页面")
    parser.add_argument("--host", default="127.0.0.1", help="Dash 监听地址，默认 127.0.0.1")
    parser.add_argument("--port", type=int, default=8050, help="Dash 端口，默认 8050")
    parser.add_argument("--debug", action="store_true", help="Dash debug 模式")
    return parser.parse_args()


def main():
    args = parse_args()
    scada = load_scada()
    anom_df = load_anomaly_segments()
    results = prepare_line_data(scada, anom_df)

    # 始终保留静态输出，便于离线查看
    save_comparison_csv(results)
    save_static_png(results)

    # if args.dash:
    if 1:
        print("\n" + "=" * 65)
        print("  启动 Dash 服务")
        print("=" * 65)
        print(f"访问地址: http://{args.host}:{args.port}")
        app = build_dash_app(results)
        app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
