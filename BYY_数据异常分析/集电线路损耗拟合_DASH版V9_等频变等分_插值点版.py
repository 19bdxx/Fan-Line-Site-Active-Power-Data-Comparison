#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ============================================================
# 脚本说明
# ============================================================
# 本脚本用于对集电线路传输损耗进行分段建模与异常分析，主要针对 BING、DING、WU 三条线路。
#
# 处理逻辑如下：
# 1. 读取 SCADA 合并数据，以及“联合重复值检测结果.xlsx”中的异常时间段；
# 2. 按线路将异常时间段对应的数据先剔除，不参与模型构建；
# 3. 对停机近零段数据进行剔除；
# 4. 定义损耗 L = FAN_SUM_S2 - max(CT, 0)，并将 L < 0 的样本视为明显异常，
#    不参与低功率中位点、高功率拟合及局部 sigma 统计；
# 5. 低功率区采用固定宽度 0.1 MW 分箱，计算各箱中位点并进行插值；
# 6. 高功率区采用固定宽度 5 MW 分箱，计算各箱中位点后进行二次拟合；
# 7. 对样本数过少的稀疏箱自动与相邻箱合并，以提高中位点和局部 sigma 的稳定性；
# 8. 最终输出拟合参数、低功率查表结果、局部 sigma 分箱结果，并可通过 Dash 页面可视化查看。
#
# 注意：
# - 负损耗样本（L < 0）统一按异常处理；
# - 模型拟合和局部 sigma 统计仅基于清洗后的正常样本；
# - 本脚本保留 A / A′ / B 三种方案，便于后续对比和逐时刻损耗计算。
# ============================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
import glob
import argparse
import warnings
from typing import Dict, Optional, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go


# ============================================================
# 路径配置
# ============================================================
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

SCADA_DIR = os.path.join(_ROOT_DIR, "RAW_DATA")
ANOMALY_FILE = os.path.join(_ROOT_DIR, "RAW_DATA", "联合重复值检测结果.xlsx")

FIT_SUMMARY_CSV = os.path.join(_ROOT_DIR, "fit_model_summary.csv")
FIT_SIGMA_BINS_CSV = os.path.join(_ROOT_DIR, "fit_model_sigma_bins.csv")
FIT_LOW_LOOKUP_CSV = os.path.join(_ROOT_DIR, "fit_model_low_power_lookup.csv")
FIT_HIGH_BINS_CSV = os.path.join(_ROOT_DIR, "fit_model_high_power_bins.csv")
FIT_INTERP_POINTS_CSV = os.path.join(_ROOT_DIR, "fit_model_interp_points.csv")


# ============================================================
# 集电线路映射
# ============================================================
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

SCADA_COLS = ["timestamp"] + list(LINE_CT_COL.values()) + list(LINE_FAN_COL.values())

# 固定宽度分箱参数
LOW_BIN_WIDTH = 0.5          # 低功率区固定宽度（MW）
HIGH_BIN_WIDTH = 10.0         # 高功率区固定宽度（MW）
MIN_BIN_SAMPLES = 20         # 单箱样本太少时，自动与相邻箱合并
MODEL_LOSS_MIN = 0.0         # 模型构建时只保留 L >= MODEL_LOSS_MIN 的样本
MAX_SCATTER_POINTS = 12000   # 散点图最大点数（不影响拟合结果）

# 低/高功率切换阈值（MW）
LINE_FAN_THRESHOLD_B = {
    "BING": 10.0,
    "DING": 10.0,
    "WU": 10.0,
}


# ============================================================
# 工具函数
# ============================================================
def get_line_by_fan_num(fan_num) -> str:
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


def sample_df(df: pd.DataFrame, n: int = MAX_SCATTER_POINTS, seed: int = 42) -> pd.DataFrame:
    if len(df) <= n:
        return df.copy()
    return df.sample(n=n, random_state=seed).sort_values("timestamp")


def build_exclude_mask(ts_np: np.ndarray, segs: pd.DataFrame) -> np.ndarray:
    """
    给定 SCADA 时间戳数组和异常时间段，返回是否落入异常段的布尔 mask
    """
    mask = np.zeros(len(ts_np), dtype=bool)
    if segs is None or len(segs) == 0:
        return mask

    t_int = ts_np.astype("datetime64[ns]").astype("int64")

    for _, row in segs.iterrows():
        s = pd.to_datetime(row["开始时间"]).to_datetime64().astype("datetime64[ns]").astype("int64")
        e = pd.to_datetime(row["结束时间"]).to_datetime64().astype("datetime64[ns]").astype("int64")
        mask |= (t_int >= s) & (t_int <= e)

    return mask


def _empty_bin_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "P_left", "P_right", "P_med", "L_med", "n",
        "base_bin_count", "bin_width", "min_bin_samples",
    ])


def _concat_parts(parts: List[pd.DataFrame]) -> pd.DataFrame:
    valid_parts = [x for x in parts if x is not None and len(x) > 0]
    if not valid_parts:
        return pd.DataFrame(columns=["_P", "_L"])
    return pd.concat(valid_parts, ignore_index=True)


def filter_model_xy(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    loss_min: float = MODEL_LOSS_MIN,
):
    """
    模型构建统一过滤：
    - x / y 必须有限
    - y(损耗) 必须 >= loss_min
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_vals, dtype=float)

    keep = np.isfinite(x) & np.isfinite(y)
    if np.isfinite(loss_min):
        keep &= (y >= float(loss_min))

    return x[keep], y[keep]


def build_fixed_width_median_bins(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    bin_width: float,
    min_bin_samples: int,
    range_start: Optional[float] = None,
    range_end: Optional[float] = None,
    loss_min: float = MODEL_LOSS_MIN,
) -> pd.DataFrame:
    """
    固定宽度分箱 + 稀疏箱自动合并。

    返回列：
      P_left, P_right, P_med, L_med, n,
      base_bin_count, bin_width, min_bin_samples

    规则：
    1) 先按固定宽度生成基础箱；
    2) 若某个箱样本数不足 min_bin_samples，则与后续相邻箱连续合并；
    3) 若最后剩余尾箱样本仍不足，则并入前一个已生成箱。
    """
    x, y = filter_model_xy(x_vals, y_vals, loss_min=loss_min)

    if len(x) == 0:
        return _empty_bin_df()

    tmp = pd.DataFrame({"_P": x, "_L": y}).sort_values("_P").reset_index(drop=True)

    if range_start is None:
        range_start = float(np.floor(tmp["_P"].min() / bin_width) * bin_width)
    if range_end is None:
        range_end = float(np.ceil(tmp["_P"].max() / bin_width) * bin_width)

    range_start = float(range_start)
    range_end = float(range_end)

    if not np.isfinite(range_start) or not np.isfinite(range_end) or range_end <= range_start:
        return _empty_bin_df()

    n_steps = int(np.ceil((range_end - range_start) / bin_width))
    edges = range_start + np.arange(n_steps + 1, dtype=float) * bin_width
    if edges[-1] < range_end:
        edges = np.append(edges, range_end)

    if len(edges) < 2:
        return _empty_bin_df()

    # 使用右闭区间：(left, right]。
    # 这里不用 Interval 直接做 key，改用整数 bin_idx，避免浮点边界导致样本漏入箱。
    bin_idx = np.searchsorted(edges, tmp["_P"].values, side="left") - 1
    bin_idx = np.clip(bin_idx, 0, len(edges) - 2)
    tmp["_bin_idx"] = bin_idx
    grouped = {
        int(i): g[["_P", "_L"]].copy()
        for i, g in tmp.groupby("_bin_idx", observed=True)
    }

    merged_rows = []
    acc_left = None
    acc_right = None
    acc_parts: List[pd.DataFrame] = []
    acc_n = 0
    acc_base_bin_count = 0

    def flush_current() -> None:
        nonlocal acc_left, acc_right, acc_parts, acc_n, acc_base_bin_count, merged_rows
        if acc_left is None or acc_right is None:
            return

        merged_df = _concat_parts(acc_parts)
        if len(merged_df) == 0:
            acc_left = None
            acc_right = None
            acc_parts = []
            acc_n = 0
            acc_base_bin_count = 0
            return

        merged_rows.append({
            "P_left": float(acc_left),
            "P_right": float(acc_right),
            "P_med": float(merged_df["_P"].median()),
            "L_med": float(merged_df["_L"].median()),
            "n": int(len(merged_df)),
            "base_bin_count": int(acc_base_bin_count),
            "bin_width": float(bin_width),
            "min_bin_samples": int(min_bin_samples),
            "_samples": merged_df,
        })

        acc_left = None
        acc_right = None
        acc_parts = []
        acc_n = 0
        acc_base_bin_count = 0

    for i in range(len(edges) - 1):
        left = float(edges[i])
        right = float(edges[i + 1])
        g = grouped.get(i, pd.DataFrame(columns=["_P", "_L"]))

        if acc_left is None:
            acc_left = left
        acc_right = right
        acc_base_bin_count += 1

        if len(g) > 0:
            acc_parts.append(g)
            acc_n += len(g)

        if acc_n >= min_bin_samples:
            flush_current()

    # 尾部剩余箱处理：样本不足则并入前一箱；没有前一箱则单独保留
    if acc_left is not None and acc_right is not None:
        tail_df = _concat_parts(acc_parts)
        if len(tail_df) > 0:
            if merged_rows and len(tail_df) < min_bin_samples:
                prev = merged_rows[-1]
                merged_df = pd.concat([prev["_samples"], tail_df], ignore_index=True)
                prev["P_right"] = float(acc_right)
                prev["P_med"] = float(merged_df["_P"].median())
                prev["L_med"] = float(merged_df["_L"].median())
                prev["n"] = int(len(merged_df))
                prev["base_bin_count"] = int(prev["base_bin_count"] + acc_base_bin_count)
                prev["_samples"] = merged_df
            else:
                flush_current()

    if not merged_rows:
        return _empty_bin_df()

    out = pd.DataFrame(merged_rows)
    if "_samples" in out.columns:
        out = out.drop(columns=["_samples"])

    return out.sort_values("P_left").reset_index(drop=True)


# ============================================================
# 数据加载
# ============================================================
def load_scada() -> pd.DataFrame:
    print("=" * 70)
    print("  加载 SCADA 合并数据 ...")
    print("=" * 70)

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
        f"({scada['timestamp'].min()} ~ {scada['timestamp'].max()})"
    )
    return scada


def load_anomaly_segments() -> pd.DataFrame:
    """
    直接读取 联合重复值检测结果.xlsx
    默认其中所有记录都视为异常段
    """
    print("\n  加载异常段结果 ...")

    if not os.path.exists(ANOMALY_FILE):
        raise FileNotFoundError(f"未找到异常段文件：{ANOMALY_FILE}")

    anom_df = pd.read_excel(ANOMALY_FILE, engine="openpyxl")
    print(f"  ✅ 已读取 联合重复值检测结果.xlsx（{len(anom_df):,} 条）")

    required_cols = {"开始时间", "结束时间", "风机编号"}
    missing = required_cols - set(anom_df.columns)
    if missing:
        raise ValueError(f"异常段文件缺少必要列：{missing}")

    anom_df["开始时间"] = pd.to_datetime(anom_df["开始时间"])
    anom_df["结束时间"] = pd.to_datetime(anom_df["结束时间"])
    anom_df["line"] = anom_df["风机编号"].apply(get_line_by_fan_num)

    return anom_df[["开始时间", "结束时间", "风机编号", "line"]].copy()


# ============================================================
# 拟合与分段模型（固定宽度分箱版）
# ============================================================
def fit_quad_from_xy(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    bin_width: float = HIGH_BIN_WIDTH,
    min_bin_samples: int = MIN_BIN_SAMPLES,
    range_start: Optional[float] = None,
    range_end: Optional[float] = None,
    loss_min: float = MODEL_LOSS_MIN,
):
    """
    高功率区二次拟合：
    - 先按固定宽度分箱
    - 稀疏箱自动合并
    - 每个合并箱取中位数
    - 再对中位点做二次拟合

    返回：
      coeffs, bins_df, r2_bins, sigma_global, n_fit
    """
    x, y = filter_model_xy(x_vals, y_vals, loss_min=loss_min)
    n_fit = len(x)

    empty_bins = _empty_bin_df()

    if n_fit < 5:
        return (
            np.array([np.nan, np.nan, np.nan]),
            empty_bins,
            np.nan,
            np.nan,
            n_fit,
        )

    bins_df = build_fixed_width_median_bins(
        x_vals=x,
        y_vals=y,
        bin_width=bin_width,
        min_bin_samples=min_bin_samples,
        range_start=range_start,
        range_end=range_end,
        loss_min=loss_min,
    )

    if len(bins_df) < 3:
        return np.array([np.nan, np.nan, np.nan]), bins_df, np.nan, np.nan, n_fit

    coeffs = np.polyfit(bins_df["P_med"].values, bins_df["L_med"].values, deg=2)

    y_hat_bins = np.polyval(coeffs, bins_df["P_med"].values)
    ss_res = np.sum((bins_df["L_med"].values - y_hat_bins) ** 2)
    ss_tot = np.sum((bins_df["L_med"].values - bins_df["L_med"].mean()) ** 2)
    r2_bins = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    y_hat_raw = np.polyval(coeffs, x)
    sigma_global = float(np.std(y - y_hat_raw)) if len(x) > 0 else np.nan

    return coeffs, bins_df, r2_bins, sigma_global, n_fit


def build_low_power_lookup(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    threshold: float,
    bin_width: float = LOW_BIN_WIDTH,
    min_bin_samples: int = MIN_BIN_SAMPLES,
    loss_min: float = MODEL_LOSS_MIN,
):
    """
    低功率区：固定宽度分箱中位数 lookup 表
    只保留 0 < P <= threshold 的样本
    """
    x, y = filter_model_xy(x_vals, y_vals, loss_min=loss_min)

    mask = (x > 0) & (x <= threshold)
    x = x[mask]
    y = y[mask]

    if len(x) < 5:
        return _empty_bin_df()

    return build_fixed_width_median_bins(
        x_vals=x,
        y_vals=y,
        bin_width=bin_width,
        min_bin_samples=min_bin_samples,
        range_start=0.0,
        range_end=float(threshold),
        loss_min=loss_min,
    )



def _prepare_interp_point_frame(
    low_points_df: Optional[pd.DataFrame],
    high_points_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    parts = []

    if low_points_df is not None and len(low_points_df) > 0:
        low = low_points_df.copy()
        low["region"] = "low"
        parts.append(low)

    if high_points_df is not None and len(high_points_df) > 0:
        high = high_points_df.copy()
        high["region"] = "high"
        parts.append(high)

    if not parts:
        return pd.DataFrame(columns=["region", "P_left", "P_right", "P_med", "L_med"])

    out = pd.concat(parts, ignore_index=True, sort=False)
    out = out.dropna(subset=["P_med"]).sort_values(["P_med", "P_left", "P_right"]).reset_index(drop=True)

    # 同一功率点若重复，优先保留样本数更多的箱；再退化为均值
    if len(out) > 1 and out["P_med"].duplicated().any():
        if "n" in out.columns:
            out = out.sort_values(["P_med", "n"], ascending=[True, False]).drop_duplicates(subset=["P_med"], keep="first")
        else:
            agg_cols = {c: "first" for c in out.columns if c not in ["P_med", "L_med"]}
            agg_cols["L_med"] = "mean"
            out = out.groupby("P_med", as_index=False).agg(agg_cols)

    return out.sort_values(["P_med", "P_left", "P_right"]).reset_index(drop=True)


def interp_1d_with_hold(values: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    xp = np.asarray(xp, dtype=float)
    fp = np.asarray(fp, dtype=float)

    out = np.full(len(values), np.nan)
    keep = np.isfinite(xp) & np.isfinite(fp)
    xp = xp[keep]
    fp = fp[keep]

    if len(xp) == 0:
        return out
    if len(xp) == 1:
        out[np.isfinite(values)] = fp[0]
        return out

    order = np.argsort(xp)
    xp = xp[order]
    fp = fp[order]

    valid_values = np.isfinite(values)
    if np.any(valid_values):
        out[valid_values] = np.interp(values[valid_values], xp, fp, left=fp[0], right=fp[-1])
    return out


def predict_loss_from_interp_points(
    p_vals: np.ndarray,
    low_points_df: Optional[pd.DataFrame],
    high_points_df: Optional[pd.DataFrame],
) -> np.ndarray:
    point_df = _prepare_interp_point_frame(low_points_df, high_points_df)
    if len(point_df) == 0:
        return np.full(len(np.asarray(p_vals, dtype=float)), np.nan)

    return interp_1d_with_hold(
        np.asarray(p_vals, dtype=float),
        point_df["P_med"].values,
        point_df["L_med"].values,
    )


def predict_piecewise_loss(
    p_vals: np.ndarray,
    threshold: float,
    low_lookup_df: Optional[pd.DataFrame],
    high_coeffs: np.ndarray,
) -> np.ndarray:
    """
    分段预测：
    - 低功率区：分箱中位数线性插值
    - 高功率区：二次拟合
    """
    p = np.asarray(p_vals, dtype=float)
    out = np.full(len(p), np.nan)

    if low_lookup_df is not None and len(low_lookup_df) > 0:
        xp = low_lookup_df["P_med"].values.astype(float)
        yp = low_lookup_df["L_med"].values.astype(float)

        low_mask = (p > 0) & (p <= threshold) & np.isfinite(p)
        if len(xp) == 1:
            out[low_mask] = yp[0]
        elif len(xp) >= 2:
            out[low_mask] = np.interp(
                p[low_mask],
                xp,
                yp,
                left=yp[0],
                right=yp[-1],
            )

    high_mask = (p > threshold) & np.isfinite(p)
    if np.all(np.isfinite(high_coeffs)):
        out[high_mask] = np.polyval(high_coeffs, p[high_mask])

    return out



def build_local_sigma_table_piecewise(
    x_vals: np.ndarray,
    y_vals: np.ndarray,
    threshold: float,
    low_lookup_df: Optional[pd.DataFrame],
    high_coeffs: np.ndarray,
    high_bins_df: Optional[pd.DataFrame],
    loss_min: float = MODEL_LOSS_MIN,
):
    """
    按模型实际使用的箱边界计算局部 sigma。
    sigma 的残差基准改为：
    - 低功率区：低功率中位点插值
    - 低/高功率衔接段：低功率最后一个点与高功率第一个点桥接插值
    - 高功率区：高功率中位点插值

    这样导出的 sigma_local 与检测脚本的插值口径保持一致。
    """
    x, y = filter_model_xy(x_vals, y_vals, loss_min=loss_min)

    empty_cols = [
        "region", "bin_source", "sigma_switch_threshold",
        "P_left", "P_right", "P_med", "sigma_local", "n",
        "base_bin_count", "bin_width", "min_bin_samples",
    ]
    if len(x) < 10:
        return pd.DataFrame(columns=empty_cols)

    y_hat = predict_loss_from_interp_points(
        p_vals=x,
        low_points_df=low_lookup_df,
        high_points_df=high_bins_df,
    )

    tmp = pd.DataFrame({"_P": x, "_L": y, "_L_hat": y_hat})
    tmp = tmp[np.isfinite(tmp["_P"]) & np.isfinite(tmp["_L_hat"])].copy()
    if len(tmp) < 5:
        return pd.DataFrame(columns=empty_cols)

    tmp["_resid"] = tmp["_L"] - tmp["_L_hat"]

    rows = []

    def append_sigma_rows(ref_df: Optional[pd.DataFrame], region: str, bin_source: str):
        if ref_df is None or len(ref_df) == 0:
            return
        ref_df_sorted = ref_df.sort_values("P_left").reset_index(drop=True)
        for _, row in ref_df_sorted.iterrows():
            left = float(row["P_left"])
            right = float(row["P_right"])
            p_med = float(row["P_med"])
            g = tmp[(tmp["_P"] > left) & (tmp["_P"] <= right)]
            rows.append({
                "region": region,
                "bin_source": bin_source,
                "sigma_switch_threshold": float(threshold) if np.isfinite(threshold) else np.nan,
                "P_left": left,
                "P_right": right,
                "P_med": p_med,
                "sigma_local": float(np.std(g["_resid"].values)) if len(g) >= 3 else np.nan,
                "n": int(len(g)),
                "base_bin_count": int(row.get("base_bin_count", np.nan)) if pd.notna(row.get("base_bin_count", np.nan)) else np.nan,
                "bin_width": float(row.get("bin_width", np.nan)) if pd.notna(row.get("bin_width", np.nan)) else np.nan,
                "min_bin_samples": int(row.get("min_bin_samples", np.nan)) if pd.notna(row.get("min_bin_samples", np.nan)) else np.nan,
            })

    append_sigma_rows(low_lookup_df, region="low", bin_source="low_lookup")
    append_sigma_rows(high_bins_df, region="high", bin_source="high_fit_bins")

    if not rows:
        return pd.DataFrame(columns=empty_cols)

    return pd.DataFrame(rows).sort_values(["P_left", "P_right"]).reset_index(drop=True)



def lookup_local_sigma(p_vals: np.ndarray, sigma_df: pd.DataFrame) -> np.ndarray:
    """
    按功率查找局部 sigma。
    若含 region / sigma_switch_threshold，则优先按低/高功率区域分别查表。
    """
    p = np.asarray(p_vals, dtype=float)

    if sigma_df is None or len(sigma_df) == 0:
        return np.full(len(p), np.nan)

    def lookup_from_bins(values: np.ndarray, ref_df: pd.DataFrame) -> np.ndarray:
        ref_df = ref_df.sort_values("P_left").reset_index(drop=True)
        lefts = ref_df["P_left"].values.astype(float)
        rights = ref_df["P_right"].values.astype(float)
        sigmas = ref_df["sigma_local"].values.astype(float)

        out_local = np.full(len(values), np.nan)
        for i, val in enumerate(values):
            if not np.isfinite(val):
                continue

            hit = np.where((val > lefts) & (val <= rights))[0]
            if len(hit) > 0:
                out_local[i] = sigmas[hit[0]]
            elif val <= lefts[0]:
                out_local[i] = sigmas[0]
            else:
                out_local[i] = sigmas[-1]
        return out_local

    if {"region", "sigma_switch_threshold"}.issubset(sigma_df.columns):
        thresholds = sigma_df["sigma_switch_threshold"].dropna().astype(float).unique()
        switch_threshold = float(thresholds[0]) if len(thresholds) > 0 else np.nan

        low_df = sigma_df[sigma_df["region"] == "low"].copy()
        high_df = sigma_df[sigma_df["region"] == "high"].copy()

        if np.isfinite(switch_threshold) and (len(low_df) > 0 or len(high_df) > 0):
            out = np.full(len(p), np.nan)
            low_mask = np.isfinite(p) & (p <= switch_threshold)
            high_mask = np.isfinite(p) & (p > switch_threshold)

            if len(low_df) > 0:
                out[low_mask] = lookup_from_bins(p[low_mask], low_df)
            elif np.any(low_mask) and len(high_df) > 0:
                out[low_mask] = lookup_from_bins(p[low_mask], high_df)

            if len(high_df) > 0:
                out[high_mask] = lookup_from_bins(p[high_mask], high_df)
            elif np.any(high_mask) and len(low_df) > 0:
                out[high_mask] = lookup_from_bins(p[high_mask], low_df)

            return out

    return lookup_from_bins(p, sigma_df)


def predict_loss_with_local_sigma_piecewise(
    df: pd.DataFrame,
    threshold: float,
    low_lookup_df: Optional[pd.DataFrame],
    high_coeffs: np.ndarray,
    sigma_df: pd.DataFrame,
    p_col: str,
) -> pd.DataFrame:
    out = df.copy()
    p = out[p_col].values.astype(float)

    out["L_hat"] = predict_piecewise_loss(
        p_vals=p,
        threshold=threshold,
        low_lookup_df=low_lookup_df,
        high_coeffs=high_coeffs,
    )
    out["sigma_local"] = lookup_local_sigma(p, sigma_df)
    out["residual"] = out["L"] - out["L_hat"]

    with np.errstate(divide="ignore", invalid="ignore"):
        out["z_local"] = out["residual"] / out["sigma_local"]

    return out


# ============================================================
# 每条线路准备数据
# ============================================================
def prepare_line_data(scada: pd.DataFrame, anom_df: pd.DataFrame) -> Dict[str, dict]:
    print("\n" + "=" * 70)
    print("  开始拟合并准备 Dash 数据")
    print("=" * 70)
    print(f"  分箱设置：低功率 {LOW_BIN_WIDTH} MW，高功率 {HIGH_BIN_WIDTH} MW，最小样本数 {MIN_BIN_SAMPLES}")

    results: Dict[str, dict] = {}
    ts_np = scada["timestamp"].values.astype("datetime64[ns]")

    for line in ["BING", "DING", "WU"]:
        ct_col = LINE_CT_COL[line]
        fan_col = LINE_FAN_COL[line]
        zh_name = LINE_NAME_ZH[line]

        print(f"\n{'─' * 60}\n  {zh_name}\n{'─' * 60}")

        segs = (
            anom_df.loc[anom_df["line"] == line, ["开始时间", "结束时间"]]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        exc_mask = build_exclude_mask(ts_np, segs)

        line_df = scada[["timestamp", ct_col, fan_col]].copy()
        line_df.rename(columns={ct_col: "CT", fan_col: "FAN_SUM_S2"}, inplace=True)

        line_df["is_anomaly"] = exc_mask
        line_df["CT_eff"] = line_df["CT"].clip(lower=0)
        line_df["FAN_eff"] = line_df["FAN_SUM_S2"].clip(lower=0)
        line_df["L"] = line_df["FAN_eff"] - line_df["CT_eff"]

        line_df["is_negative_loss"] = np.isfinite(line_df["L"]) & (line_df["L"] < MODEL_LOSS_MIN)
        line_df["is_model_abnormal"] = line_df["is_anomaly"] | line_df["is_negative_loss"]

        normal_df = line_df[(~line_df["is_model_abnormal"])].copy()
        anomaly_df = line_df[(line_df["is_model_abnormal"])].copy()
        negative_loss_df = line_df[(line_df["is_negative_loss"])].copy()

        fan_threshold_b = float(LINE_FAN_THRESHOLD_B.get(line, 0.0))

        # -------------------------
        # A：无低功率分段，整体固定宽度高功率箱拟合
        # -------------------------
        fit_df_a = normal_df[
            np.isfinite(normal_df["CT_eff"]) &
            np.isfinite(normal_df["L"]) &
            (normal_df["CT_eff"] > 0)
        ].copy()

        coeffs_a, bins_a, r2_a, sigma_a, n_fit_a = fit_quad_from_xy(
            fit_df_a["CT_eff"].values,
            fit_df_a["L"].values,
            bin_width=HIGH_BIN_WIDTH,
            min_bin_samples=MIN_BIN_SAMPLES,
            range_start=0.0,
        )

        sigma_table_a = build_local_sigma_table_piecewise(
            x_vals=fit_df_a["CT_eff"].values,
            y_vals=fit_df_a["L"].values,
            threshold=0.0,
            low_lookup_df=None,
            high_coeffs=coeffs_a,
            high_bins_df=bins_a,
        )

        # -------------------------
        # A′ / B：低功率区 + 高功率区分段模型
        # 低功率区样本由 FAN_eff 定义
        # -------------------------
        fit_df_low = normal_df[
            np.isfinite(normal_df["FAN_eff"]) &
            np.isfinite(normal_df["L"]) &
            (normal_df["FAN_eff"] > 0) &
            (normal_df["FAN_eff"] <= fan_threshold_b)
        ].copy()

        fit_df_high = normal_df[
            np.isfinite(normal_df["FAN_eff"]) &
            np.isfinite(normal_df["L"]) &
            (normal_df["FAN_eff"] > fan_threshold_b)
        ].copy()

        # A′：低功率区 lookup 用 CT_eff，阈值仍由 FAN_eff 定义
        low_threshold_ap = float(np.nanmax(fit_df_low["CT_eff"].values)) if len(fit_df_low) else fan_threshold_b
        low_lookup_ap = build_low_power_lookup(
            x_vals=fit_df_low["CT_eff"].values,
            y_vals=fit_df_low["L"].values,
            threshold=low_threshold_ap,
            bin_width=LOW_BIN_WIDTH,
            min_bin_samples=MIN_BIN_SAMPLES,
        )

        coeffs_ap, bins_ap, r2_ap, sigma_ap, n_fit_ap = fit_quad_from_xy(
            fit_df_high["CT_eff"].values,
            fit_df_high["L"].values,
            bin_width=HIGH_BIN_WIDTH,
            min_bin_samples=MIN_BIN_SAMPLES,
        )

        ap_threshold_x = (
            float(low_lookup_ap["P_right"].max())
            if low_lookup_ap is not None and len(low_lookup_ap) > 0
            else low_threshold_ap
        )
        sigma_table_ap = build_local_sigma_table_piecewise(
            x_vals=np.concatenate([fit_df_low["CT_eff"].values, fit_df_high["CT_eff"].values]) if (len(fit_df_low) + len(fit_df_high)) > 0 else np.array([]),
            y_vals=np.concatenate([fit_df_low["L"].values, fit_df_high["L"].values]) if (len(fit_df_low) + len(fit_df_high)) > 0 else np.array([]),
            threshold=ap_threshold_x,
            low_lookup_df=low_lookup_ap,
            high_coeffs=coeffs_ap,
            high_bins_df=bins_ap,
        )

        # B：低功率区 lookup 用 FAN_eff，高功率区二次
        low_lookup_b = build_low_power_lookup(
            x_vals=fit_df_low["FAN_eff"].values,
            y_vals=fit_df_low["L"].values,
            threshold=fan_threshold_b,
            bin_width=LOW_BIN_WIDTH,
            min_bin_samples=MIN_BIN_SAMPLES,
        )

        coeffs_b, bins_b, r2_b, sigma_b, n_fit_b = fit_quad_from_xy(
            fit_df_high["FAN_eff"].values,
            fit_df_high["L"].values,
            bin_width=HIGH_BIN_WIDTH,
            min_bin_samples=MIN_BIN_SAMPLES,
            range_start=fan_threshold_b,
        )

        sigma_table_b = build_local_sigma_table_piecewise(
            x_vals=np.concatenate([fit_df_low["FAN_eff"].values, fit_df_high["FAN_eff"].values]) if (len(fit_df_low) + len(fit_df_high)) > 0 else np.array([]),
            y_vals=np.concatenate([fit_df_low["L"].values, fit_df_high["L"].values]) if (len(fit_df_low) + len(fit_df_high)) > 0 else np.array([]),
            threshold=fan_threshold_b,
            low_lookup_df=low_lookup_b,
            high_coeffs=coeffs_b,
            high_bins_df=bins_b,
        )

        print(f"  异常段数: {len(segs):,}")
        print(f"  正常样本: {len(normal_df):,}")
        print(f"  异常样本: {len(anomaly_df):,}")
        print(f"  其中负损耗样本(已剔除出模型): {len(negative_loss_df):,}")
        print(f"  低功率样本(A′/B共用): {len(fit_df_low):,}")
        print(f"  高功率样本(A′/B共用): {len(fit_df_high):,}")
        print(f"  方案 A  拟合点数: {n_fit_a:,} | 高功率箱数: {len(bins_a):,}")
        print(f"  方案 A′ 拟合点数: {n_fit_ap:,} | 低功率箱数: {len(low_lookup_ap):,} | 高功率箱数: {len(bins_ap):,}")
        print(f"  方案 B  拟合点数: {n_fit_b:,} | 低功率箱数: {len(low_lookup_b):,} | 高功率箱数: {len(bins_b):,} | 阈值>{fan_threshold_b} MW")
        print(f"  A : R²={r2_a:.4f},  全局σ={sigma_a:.4f}")
        print(f"  A′: R²={r2_ap:.4f}, 全局σ={sigma_ap:.4f}")
        print(f"  B : R²={r2_b:.4f},  全局σ={sigma_b:.4f}")

        results[line] = {
            "line_name": zh_name,
            "line_code": line,
            "line_df": line_df,
            "normal_df": normal_df,
            "anomaly_df": anomaly_df,
            "negative_loss_df": negative_loss_df,

            "fit_df_a": fit_df_a,
            "fit_df_low": fit_df_low,
            "fit_df_high": fit_df_high,

            # A
            "coeffs_a": coeffs_a,
            "bins_a": bins_a,
            "sigma_table_a": sigma_table_a,
            "r2_a": r2_a,
            "sigma_a": sigma_a,
            "n_fit_a": n_fit_a,

            # A′
            "coeffs_ap": coeffs_ap,
            "bins_ap": bins_ap,
            "low_lookup_ap": low_lookup_ap,
            "sigma_table_ap": sigma_table_ap,
            "r2_ap": r2_ap,
            "sigma_ap": sigma_ap,
            "n_fit_ap": n_fit_ap,
            "ap_threshold_x": ap_threshold_x,

            # B
            "coeffs_b": coeffs_b,
            "bins_b": bins_b,
            "low_lookup_b": low_lookup_b,
            "sigma_table_b": sigma_table_b,
            "r2_b": r2_b,
            "sigma_b": sigma_b,
            "n_fit_b": n_fit_b,

            "fan_threshold_b": fan_threshold_b,
            "scatter_normal": sample_df(normal_df),
            "scatter_anomaly": sample_df(anomaly_df),
        }

    return results



def build_interp_points_output(
    line: str,
    line_name: str,
    scheme: str,
    power_col: str,
    threshold_axis: str,
    switch_threshold: float,
    fan_threshold_b: float,
    low_lookup_df: Optional[pd.DataFrame],
    high_bins_df: Optional[pd.DataFrame],
    sigma_table_df: Optional[pd.DataFrame],
) -> pd.DataFrame:
    parts = []

    if low_lookup_df is not None and len(low_lookup_df) > 0:
        low = low_lookup_df.copy()
        low["region"] = "low"
        low["bin_source"] = "low_lookup"
        parts.append(low)

    if high_bins_df is not None and len(high_bins_df) > 0:
        high = high_bins_df.copy()
        high["region"] = "high"
        high["bin_source"] = "high_fit_bins"
        parts.append(high)

    if not parts:
        return pd.DataFrame()

    point_df = pd.concat(parts, ignore_index=True, sort=False)
    point_df = point_df.sort_values(["P_med", "P_left", "P_right"]).reset_index(drop=True)

    sigma_cols = ["region", "P_left", "P_right", "P_med", "sigma_local", "n"]
    if sigma_table_df is not None and len(sigma_table_df) > 0:
        sigma_use = sigma_table_df[sigma_cols].copy().rename(columns={"n": "sigma_sample_n"})
        point_df = point_df.merge(
            sigma_use,
            on=["region", "P_left", "P_right", "P_med"],
            how="left",
        )
    else:
        point_df["sigma_local"] = np.nan
        point_df["sigma_sample_n"] = np.nan

    point_df["var_local"] = point_df["sigma_local"].astype(float) ** 2
    point_df["line"] = line
    point_df["line_name"] = line_name
    point_df["scheme"] = scheme
    point_df["power_col_for_prediction"] = power_col
    point_df["power_threshold_axis"] = threshold_axis
    point_df["power_switch_threshold"] = switch_threshold
    point_df["fan_threshold_b"] = fan_threshold_b
    point_df["interp_order"] = np.arange(1, len(point_df) + 1)

    return point_df

# ============================================================
# 保存拟合结果
# ============================================================
def save_fit_results(results: Dict[str, dict]):
    summary_rows = []
    sigma_rows = []
    low_rows = []
    high_rows = []
    interp_rows = []

    scheme_map = {
        "A": {
            "coeffs_key": "coeffs_a",
            "r2_key": "r2_a",
            "sigma_key": "sigma_a",
            "nfit_key": "n_fit_a",
            "sigma_table_key": "sigma_table_a",
            "low_lookup_key": None,
            "high_bins_key": "bins_a",
            "model_type": "quadratic_fixed_bins",
            "power_col": "CT_eff",
        },
        "AP": {
            "coeffs_key": "coeffs_ap",
            "r2_key": "r2_ap",
            "sigma_key": "sigma_ap",
            "nfit_key": "n_fit_ap",
            "sigma_table_key": "sigma_table_ap",
            "low_lookup_key": "low_lookup_ap",
            "high_bins_key": "bins_ap",
            "model_type": "piecewise_low_interp_high_quad_fixed_bins",
            "power_col": "CT_eff",
        },
        "B": {
            "coeffs_key": "coeffs_b",
            "r2_key": "r2_b",
            "sigma_key": "sigma_b",
            "nfit_key": "n_fit_b",
            "sigma_table_key": "sigma_table_b",
            "low_lookup_key": "low_lookup_b",
            "high_bins_key": "bins_b",
            "model_type": "piecewise_low_interp_high_quad_fixed_bins",
            "power_col": "FAN_eff",
        },
    }

    for line in ["BING", "DING", "WU"]:
        info = results[line]

        for scheme, meta in scheme_map.items():
            coeffs = info[meta["coeffs_key"]]
            sigma_table = info[meta["sigma_table_key"]]
            low_lookup = info[meta["low_lookup_key"]] if meta["low_lookup_key"] else None
            high_bins = info[meta["high_bins_key"]]

            if scheme == "A":
                switch_threshold = np.nan
                threshold_axis = meta["power_col"]
            elif scheme == "AP":
                switch_threshold = info["ap_threshold_x"]
                threshold_axis = "CT_eff"
            else:
                switch_threshold = info["fan_threshold_b"]
                threshold_axis = "FAN_eff"

            summary_rows.append({
                "line": line,
                "line_name": info["line_name"],
                "scheme": scheme,
                "model_type": meta["model_type"],
                "power_col_for_prediction": meta["power_col"],
                "power_threshold_axis": threshold_axis,
                "power_switch_threshold": switch_threshold,
                "fan_threshold_b": info["fan_threshold_b"],
                "low_threshold": switch_threshold,
                "high_threshold": switch_threshold,
                "low_bin_width": LOW_BIN_WIDTH,
                "high_bin_width": HIGH_BIN_WIDTH,
                "min_bin_samples": MIN_BIN_SAMPLES,
                "normal_samples": len(info["normal_df"]),
                "anomaly_samples": len(info["anomaly_df"]),
                "negative_loss_samples": len(info["negative_loss_df"]),
                "low_power_samples": len(info["fit_df_low"]) if scheme in ["AP", "B"] else np.nan,
                "high_power_samples": len(info["fit_df_high"]) if scheme in ["AP", "B"] else np.nan,
                "fit_samples": info[meta["nfit_key"]],
                "r2_bins": info[meta["r2_key"]],
                "sigma_global": info[meta["sigma_key"]],
                "coef_a2": coeffs[0] if len(coeffs) > 0 else np.nan,
                "coef_a1": coeffs[1] if len(coeffs) > 1 else np.nan,
                "coef_a0": coeffs[2] if len(coeffs) > 2 else np.nan,
                "sigma_mode": "local_by_interp_point_bins",
                "sigma_bins_count": len(sigma_table) if sigma_table is not None else 0,
                "low_bins_count": len(low_lookup) if low_lookup is not None else 0,
                "high_bins_count": len(high_bins) if high_bins is not None else 0,
                "interp_points_count": (len(low_lookup) if low_lookup is not None else 0) + (len(high_bins) if high_bins is not None else 0),
                "model_expr_high": (
                    f"L_hat_high = {coeffs[0]:.12g}*P^2 + {coeffs[1]:.12g}*P + {coeffs[2]:.12g}"
                    if np.all(np.isfinite(coeffs)) else ""
                ),
            })

            if sigma_table is not None and len(sigma_table) > 0:
                tmp_sigma = sigma_table.copy()
                tmp_sigma["line"] = line
                tmp_sigma["line_name"] = info["line_name"]
                tmp_sigma["scheme"] = scheme
                tmp_sigma["power_col_for_prediction"] = meta["power_col"]
                tmp_sigma["power_threshold_axis"] = threshold_axis
                tmp_sigma["power_switch_threshold"] = switch_threshold
                tmp_sigma["fan_threshold_b"] = info["fan_threshold_b"]
                sigma_rows.append(tmp_sigma)

            if low_lookup is not None and len(low_lookup) > 0:
                tmp_low = low_lookup.copy()
                tmp_low["line"] = line
                tmp_low["line_name"] = info["line_name"]
                tmp_low["scheme"] = scheme
                tmp_low["power_col_for_prediction"] = meta["power_col"]
                tmp_low["power_threshold_axis"] = threshold_axis
                tmp_low["power_switch_threshold"] = switch_threshold
                tmp_low["fan_threshold_b"] = info["fan_threshold_b"]
                low_rows.append(tmp_low)

            if high_bins is not None and len(high_bins) > 0:
                tmp_high = high_bins.copy()
                tmp_high["line"] = line
                tmp_high["line_name"] = info["line_name"]
                tmp_high["scheme"] = scheme
                tmp_high["power_col_for_prediction"] = meta["power_col"]
                tmp_high["power_threshold_axis"] = threshold_axis
                tmp_high["power_switch_threshold"] = switch_threshold
                tmp_high["fan_threshold_b"] = info["fan_threshold_b"]
                high_rows.append(tmp_high)

            interp_point_df = build_interp_points_output(
                line=line,
                line_name=info["line_name"],
                scheme=scheme,
                power_col=meta["power_col"],
                threshold_axis=threshold_axis,
                switch_threshold=switch_threshold,
                fan_threshold_b=info["fan_threshold_b"],
                low_lookup_df=low_lookup,
                high_bins_df=high_bins,
                sigma_table_df=sigma_table,
            )
            if len(interp_point_df) > 0:
                interp_rows.append(interp_point_df)

    summary_df = pd.DataFrame(summary_rows)
    sigma_df = pd.concat(sigma_rows, ignore_index=True) if sigma_rows else pd.DataFrame()
    low_df = pd.concat(low_rows, ignore_index=True) if low_rows else pd.DataFrame()
    high_df = pd.concat(high_rows, ignore_index=True) if high_rows else pd.DataFrame()
    interp_df = pd.concat(interp_rows, ignore_index=True) if interp_rows else pd.DataFrame()

    summary_df.to_csv(FIT_SUMMARY_CSV, index=False, encoding="utf-8-sig")
    sigma_df.to_csv(FIT_SIGMA_BINS_CSV, index=False, encoding="utf-8-sig")
    low_df.to_csv(FIT_LOW_LOOKUP_CSV, index=False, encoding="utf-8-sig")
    high_df.to_csv(FIT_HIGH_BINS_CSV, index=False, encoding="utf-8-sig")
    interp_df.to_csv(FIT_INTERP_POINTS_CSV, index=False, encoding="utf-8-sig")

    print("\n已保存拟合结果：")
    print(f"  - {FIT_SUMMARY_CSV}")
    print(f"  - {FIT_SIGMA_BINS_CSV}")
    print(f"  - {FIT_LOW_LOOKUP_CSV}")
    print(f"  - {FIT_HIGH_BINS_CSV}")
    print(f"  - {FIT_INTERP_POINTS_CSV}")


# ============================================================
# Dash 页面
# ============================================================
def build_dash_app(results: Dict[str, dict]) -> Dash:
    app = Dash(__name__)
    line_options = [{"label": LINE_NAME_ZH[k], "value": k} for k in ["BING", "DING", "WU"]]

    app.layout = html.Div(
        style={"fontFamily": "Arial, sans-serif", "padding": "16px"},
        children=[
            html.H2("集电线路传输损耗拟合（固定宽度分箱：低功率0.1 / 高功率5 / 自动合并稀疏箱）"),
            html.Div(
                style={"display": "flex", "gap": "16px", "marginBottom": "12px"},
                children=[
                    html.Div([
                        html.Label("选择集电线路"),
                        dcc.Dropdown(id="line-dd", options=line_options, value="BING", clearable=False),
                    ], style={"width": "240px"}),
                    html.Div([
                        html.Label("选择拟合方案"),
                        dcc.Dropdown(
                            id="scheme-dd",
                            options=[
                                {"label": "方案 A：P = max(CT, 0)（整体二次，固定宽度分箱）", "value": "A"},
                                {"label": "方案 A′：P = max(CT, 0)（低功率插值+高功率二次）", "value": "AP"},
                                {"label": "方案 B：P = FAN_SUM_S2（低功率插值+高功率二次）", "value": "B"},
                            ],
                            value="B",
                            clearable=False,
                        ),
                    ], style={"width": "460px"}),
                ],
            ),
            html.Div(id="metrics-box", style={"marginBottom": "12px", "fontSize": "15px"}),
            dcc.Graph(id="loss-graph", style={"height": "760px"}),
        ],
    )

    @app.callback(
        Output("loss-graph", "figure"),
        Output("metrics-box", "children"),
        Input("line-dd", "value"),
        Input("scheme-dd", "value"),
    )
    def update_graph(line_code: str, scheme: str):
        info = results[line_code]
        normal_sc = info["scatter_normal"]
        anom_sc = info["scatter_anomaly"]

        if scheme == "A":
            x_normal = normal_sc["CT_eff"]
            x_anom = anom_sc["CT_eff"]
            coeffs = info["coeffs_a"]
            bins = info["bins_a"]
            sigma_table = info["sigma_table_a"]
            low_lookup = None
            threshold = None
            display_threshold = np.nan
            threshold_axis = "CT_eff"
            r2 = info["r2_a"]
            sigma = info["sigma_a"]
            n_fit = info["n_fit_a"]
            x_label = "P = max(CT, 0) (MW)"
            scheme_label = "A"
        elif scheme == "AP":
            x_normal = normal_sc["CT_eff"]
            x_anom = anom_sc["CT_eff"]
            coeffs = info["coeffs_ap"]
            bins = info["bins_ap"]
            sigma_table = info["sigma_table_ap"]
            low_lookup = info["low_lookup_ap"]
            threshold = info["ap_threshold_x"]
            display_threshold = info["ap_threshold_x"]
            threshold_axis = "CT_eff"
            r2 = info["r2_ap"]
            sigma = info["sigma_ap"]
            n_fit = info["n_fit_ap"]
            x_label = "P = max(CT, 0) (MW)"
            scheme_label = "A′"
        else:
            x_normal = normal_sc["FAN_eff"]
            x_anom = anom_sc["FAN_eff"]
            coeffs = info["coeffs_b"]
            bins = info["bins_b"]
            sigma_table = info["sigma_table_b"]
            low_lookup = info["low_lookup_b"]
            threshold = info["fan_threshold_b"]
            display_threshold = info["fan_threshold_b"]
            threshold_axis = "FAN_eff"
            r2 = info["r2_b"]
            sigma = info["sigma_b"]
            n_fit = info["n_fit_b"]
            x_label = "P = FAN_SUM_S2 (MW)"
            scheme_label = "B"

        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=x_normal,
            y=normal_sc["L"],
            mode="markers",
            name="正常点",
            marker=dict(size=4, opacity=0.35, color="#1f77b4"),
        ))

        fig.add_trace(go.Scatter(
            x=x_anom,
            y=anom_sc["L"],
            mode="markers",
            name="异常点（含负损耗）",
            marker=dict(size=4, opacity=0.35, color="#d62728"),
        ))

        if len(bins) > 0:
            fig.add_trace(go.Scatter(
                x=bins["P_med"],
                y=bins["L_med"],
                mode="markers+lines",
                name="高功率区分箱中位数",
                marker=dict(size=7, color="#111111"),
                line=dict(color="#111111", width=1),
            ))

        if low_lookup is not None and len(low_lookup) > 0:
            fig.add_trace(go.Scatter(
                x=low_lookup["P_med"],
                y=low_lookup["L_med"],
                mode="markers+lines",
                name="低功率区中位数插值点",
                marker=dict(size=7, color="#9467bd"),
                line=dict(color="#9467bd", width=2, dash="dot"),
            ))

        x_all = pd.concat([x_normal, x_anom], ignore_index=True)
        if len(x_all) > 0:
            xmin = max(0.0, float(np.nanmin(x_all)))
            xmax = float(np.nanmax(x_all))
            if xmax > xmin:
                x_curve = np.linspace(xmin, xmax, 500)

                if scheme == "A":
                    y_curve = np.polyval(coeffs, x_curve) if np.all(np.isfinite(coeffs)) else np.full_like(x_curve, np.nan)
                else:
                    y_curve = predict_piecewise_loss(
                        p_vals=x_curve,
                        threshold=float(threshold),
                        low_lookup_df=low_lookup,
                        high_coeffs=coeffs,
                    )

                fig.add_trace(go.Scatter(
                    x=x_curve,
                    y=y_curve,
                    mode="lines",
                    name="拟合/插值曲线",
                    line=dict(color="#ff7f0e", width=3),
                ))

        fig.update_layout(
            template="plotly_white",
            title=f"{info['line_name']} 线路损耗拟合（方案 {scheme_label}）",
            xaxis_title=x_label,
            yaxis_title="L = FAN_SUM_S2 − max(CT, 0) (MW)",
            legend=dict(orientation="h", y=1.08, x=0),
        )

        if np.all(np.isfinite(coeffs)):
            coeff_txt = f"L_hat_high(P) = {coeffs[0]:.6g}·P² + {coeffs[1]:.6g}·P + {coeffs[2]:.6g}"
        else:
            coeff_txt = "L_hat_high(P) = 拟合失败"

        threshold_txt = (
            f"分段阈值({threshold_axis}) = {display_threshold:.4f} MW | "
            if np.isfinite(display_threshold) else ""
        )

        metrics = (
            f"{info['line_name']} | 方案 {scheme_label} | "
            f"正常样本 = {len(info['normal_df']):,} | "
            f"异常样本 = {len(info['anomaly_df']):,} | "
            f"负损耗样本 = {len(info['negative_loss_df']):,} | "
            f"参与高功率拟合点数 = {n_fit:,} | "
            f"高功率箱宽 = {HIGH_BIN_WIDTH} MW | 低功率箱宽 = {LOW_BIN_WIDTH} MW | 最小样本数 = {MIN_BIN_SAMPLES} | "
            f"R² = {r2:.4f} | 全局σ(参考) = {sigma:.4f} MW | "
            f"局部σ箱数 = {len(sigma_table):,} | "
            f"{threshold_txt}"
            f"参考FAN阈值 = {info['fan_threshold_b']} MW | "
            f"{coeff_txt}"
        )

        return fig, metrics

    return app


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dash", action="store_true", help="启动 Dash 页面")
    parser.add_argument("--port", type=int, default=8050, help="Dash 端口")
    args = parser.parse_args()

    scada = load_scada()
    anom_df = load_anomaly_segments()
    results = prepare_line_data(scada, anom_df)
    save_fit_results(results)

    if args.dash:
        app = build_dash_app(results)
        print(f"\n🚀 Dash 已启动: http://127.0.0.1:{args.port}")
        app.run(debug=False, port=args.port)
    else:
        print("\n已完成拟合。")
        print("未指定 --dash，因此不启动页面。")
        print("可使用：python Pasted_code_fixed_bins.py --dash")


if __name__ == "__main__":
    main()
