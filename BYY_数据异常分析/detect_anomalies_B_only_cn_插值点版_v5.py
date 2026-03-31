import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd


DEFAULT_LINE_CT_COL = {
    "BING": "ACTIVE_POWER_BING",
    "DING": "ACTIVE_POWER_DING",
    "WU": "ACTIVE_POWER_WU",
}
DEFAULT_LINE_FAN_COL = {
    "BING": "BING_ACTIVE_POWER_SUM_S2",
    "DING": "DING_ACTIVE_POWER_SUM_S2",
    "WU": "WU_ACTIVE_POWER_SUM_S2",
}

DEFAULT_INPUT_CSV = r"RAW_DATA\#7-1峡阳B_20240315-20241224_with_sum.csv"
DEFAULT_SUMMARY_CSV = "fit_model_summary.csv"
DEFAULT_SIGMA_CSV = "fit_model_sigma_bins.csv"
DEFAULT_LOW_LOOKUP_CSV = "fit_model_low_power_lookup.csv"
DEFAULT_INTERP_POINTS_CSV = "fit_model_interp_points.csv"
FIXED_SCHEME = "B"

ULTRA_LOW_POWER_THRESHOLDS = {
    "BING": 2.0,
    "DING": 1.5,
    "WU": 3.0,
}
DEFAULT_ULTRA_LOW_LOSS_RATIO = 0.9


DETAIL_RENAME_MAP = {
    "timestamp": "时间",
    "line": "线路",
    "scheme": "方案",
    "CT": "CT功率",
    "FAN_SUM_S2": "风机汇总功率",
    "CT_eff": "CT有效功率",
    "FAN_eff": "风机有效功率",
    "L": "损耗",
    "power_value": "预测功率值",
    "region": "区间",
    "is_valid_for_detection": "是否参与检测",
    "anomaly_rule": "判异规则",
    "L_hat": "预测损耗",
    "sigma_local": "局部sigma",
    "residual": "残差",
    "is_anomaly": "是否异常",
}

SUMMARY_RENAME_MAP = {
    "line": "线路",
    "scheme": "方案",
    "ultra_low_power_threshold": "超低功率阈值",
    "ultra_low_loss_ratio": "超低功率损耗比例阈值",
    "total_rows": "总行数",
    "negative_loss_rows": "负损耗行数",
    "ultra_low_rule_rows": "超低功率比例判异区行数",
    "ultra_low_rule_anomaly_rows": "超低功率比例规则异常行数",
    "valid_rows": "有效检测行数",
    "anomaly_rows": "异常行数",
    "anomaly_rate": "异常率",
}


def localize_output_columns(detail_df: pd.DataFrame, anomaly_only_df: pd.DataFrame, summary_df: pd.DataFrame):
    def _translate_region(df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if "region" in out.columns:
            out["region"] = out["region"].replace({"low": "低功率区", "high": "高功率区"})
        if "anomaly_rule" in out.columns:
            out["anomaly_rule"] = out["anomaly_rule"].replace({
                "ultra_low_ratio_rule": "超低功率比例规则",
                "default_interp_z_rule": "插值+Z分数规则",
                "invalid_or_no_model": "无效点",
            })
        return out

    detail_cn = _translate_region(detail_df).rename(columns=DETAIL_RENAME_MAP)
    anomaly_cn = _translate_region(anomaly_only_df).rename(columns=DETAIL_RENAME_MAP)
    summary_cn = summary_df.rename(columns=SUMMARY_RENAME_MAP)
    return detail_cn, anomaly_cn, summary_cn


def read_csv_auto(path: str) -> pd.DataFrame:
    path_obj = Path(path)
    if not path_obj.exists():
        raise FileNotFoundError(f"文件不存在：{path_obj}")

    last_err = None
    for encoding in ["utf-8-sig", "utf-8", "gbk", "gb18030"]:
        try:
            return pd.read_csv(path_obj, encoding=encoding)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"读取失败：{path_obj}\n最后一次报错：{last_err}")


def ensure_timestamp(df: pd.DataFrame, timestamp_col: str = "timestamp") -> pd.DataFrame:
    out = df.copy()
    if timestamp_col not in out.columns:
        raise ValueError(f"输入数据缺少时间列：{timestamp_col}")
    out[timestamp_col] = pd.to_datetime(out[timestamp_col], errors="coerce")
    return out



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


def prepare_interp_points_df(interp_points_df: Optional[pd.DataFrame]) -> pd.DataFrame:
    if interp_points_df is None or len(interp_points_df) == 0:
        return pd.DataFrame(columns=["P_med", "L_med", "sigma_local", "var_local"])

    out = interp_points_df.copy()
    out = out.dropna(subset=["P_med"]).copy()

    if "var_local" not in out.columns:
        out["var_local"] = pd.to_numeric(out.get("sigma_local"), errors="coerce") ** 2

    out["P_med"] = pd.to_numeric(out["P_med"], errors="coerce")
    out["L_med"] = pd.to_numeric(out["L_med"], errors="coerce")
    out["sigma_local"] = pd.to_numeric(out.get("sigma_local"), errors="coerce")
    out["var_local"] = pd.to_numeric(out["var_local"], errors="coerce")

    sort_cols = ["P_med"]
    if "interp_order" in out.columns:
        sort_cols.append("interp_order")
    out = out.sort_values(sort_cols).reset_index(drop=True)

    # 同一 P_med 若重复，优先保留有 sigma 的点；否则对数值列取均值
    if len(out) > 1 and out["P_med"].duplicated().any():
        agg = {
            "L_med": "mean",
            "sigma_local": "mean",
            "var_local": "mean",
        }
        keep_cols = [c for c in out.columns if c not in agg and c != "P_med"]
        for c in keep_cols:
            agg[c] = "first"
        out = out.groupby("P_med", as_index=False).agg(agg)
        out = out.sort_values(["P_med"]).reset_index(drop=True)

    return out


def predict_loss_sigma_by_interp_points(
    p_vals: np.ndarray,
    interp_points_df: Optional[pd.DataFrame],
) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(p_vals, dtype=float)
    point_df = prepare_interp_points_df(interp_points_df)

    if len(point_df) == 0:
        nan_arr = np.full(len(p), np.nan)
        return nan_arr, nan_arr

    l_hat = interp_1d_with_hold(
        p,
        point_df["P_med"].values,
        point_df["L_med"].values,
    )
    var_hat = interp_1d_with_hold(
        p,
        point_df["P_med"].values,
        point_df["var_local"].values,
    )
    sigma_hat = np.sqrt(np.clip(var_hat, a_min=0.0, a_max=None))
    return l_hat, sigma_hat


def build_line_df(raw_df: pd.DataFrame, line: str, ct_col: str, fan_col: str) -> pd.DataFrame:
    req = {"timestamp", ct_col, fan_col}
    missing = req - set(raw_df.columns)
    if missing:
        raise ValueError(f"{line} 缺少列：{missing}")

    out = raw_df[["timestamp", ct_col, fan_col]].copy()
    out.rename(columns={ct_col: "CT", fan_col: "FAN_SUM_S2"}, inplace=True)
    out["CT_eff"] = pd.to_numeric(out["CT"], errors="coerce").clip(lower=0)
    out["FAN_eff"] = pd.to_numeric(out["FAN_SUM_S2"], errors="coerce").clip(lower=0)
    out["L"] = out["FAN_eff"] - out["CT_eff"]
    out["is_negative_loss"] = np.isfinite(out["L"]) & (out["L"] < 0)
    out["line"] = line
    return out



def detect_for_b_model(
    line_df: pd.DataFrame,
    model_row: pd.Series,
    interp_points_df: pd.DataFrame,
    z_threshold: float,
    sigma_floor: float,
    residual_abs_threshold: float,
    ultra_low_loss_ratio: float,
) -> pd.DataFrame:
    scheme = str(model_row["scheme"])
    line = str(model_row["line"])
    if scheme != FIXED_SCHEME:
        raise ValueError(f"该脚本只支持方案 {FIXED_SCHEME}，当前为 {scheme}")

    p_col = str(model_row["power_col_for_prediction"])
    switch_threshold = pd.to_numeric(model_row.get("power_switch_threshold"), errors="coerce")

    out = line_df.copy()
    p = pd.to_numeric(out[p_col], errors="coerce").values.astype(float)

    ultra_low_power_threshold = float(ULTRA_LOW_POWER_THRESHOLDS.get(line, np.nan))

    out["L_hat"], out["sigma_local"] = predict_loss_sigma_by_interp_points(
        p_vals=p,
        interp_points_df=interp_points_df,
    )
    out["region"] = np.where(
        np.isfinite(p) & np.isfinite(switch_threshold) & (p <= switch_threshold),
        "low",
        "high",
    )

    sigma_local = out["sigma_local"].astype(float)
    out["sigma_used"] = np.maximum(sigma_local, sigma_floor) if sigma_floor > 0 else sigma_local
    out["residual"] = out["L"] - out["L_hat"]
    with np.errstate(divide="ignore", invalid="ignore"):
        out["z_local"] = out["residual"] / out["sigma_used"]

    valid_mask = (
        np.isfinite(p) &
        np.isfinite(out["L"]) &
        np.isfinite(out["L_hat"]) &
        np.isfinite(out["sigma_used"]) &
        (out["sigma_used"] > 0)
    )

    ultra_low_rule_mask = (
        np.isfinite(p) &
        np.isfinite(out["L"]) &
        np.isfinite(ultra_low_power_threshold) &
        (p >= 0) &
        (p <= ultra_low_power_threshold)
    )

    default_rule_mask = valid_mask & (~ultra_low_rule_mask)

    out["scheme"] = scheme
    out["line"] = line
    out["power_col_for_prediction"] = p_col
    out["power_value"] = p
    out["power_switch_threshold"] = switch_threshold
    out["is_valid_for_detection"] = valid_mask | ultra_low_rule_mask
    out["is_ultra_low_rule_zone"] = ultra_low_rule_mask
    out["ultra_low_power_threshold"] = ultra_low_power_threshold
    out["ultra_low_loss_ratio"] = float(ultra_low_loss_ratio)
    out["abs_residual"] = out["residual"].abs()
    out["abs_z_local"] = out["z_local"].abs()

    # 超低功率区完全切换到比例规则：
    # 这些点虽然可以先算出插值结果，但最终不使用 L_hat / sigma / residual / z 来判异，
    # 为避免输出歧义，这些统计量在结果中统一置空。
    cols_to_clear = ["L_hat", "sigma_local", "sigma_used", "residual", "abs_residual", "z_local", "abs_z_local"]
    out.loc[ultra_low_rule_mask, cols_to_clear] = np.nan

    ultra_low_anomaly = ultra_low_rule_mask & (out["L"] < float(ultra_low_loss_ratio) * out["power_value"])
    default_anomaly = (
        default_rule_mask &
        (
            out["is_negative_loss"] |
            (
                (out["abs_z_local"] >= z_threshold) &
                (out["abs_residual"] >= residual_abs_threshold)
            )
        )
    )
    out["is_anomaly"] = ultra_low_anomaly | default_anomaly
    out["anomaly_rule"] = np.where(
        ultra_low_rule_mask,
        "ultra_low_ratio_rule",
        np.where(default_rule_mask, "default_interp_z_rule", "invalid_or_no_model")
    )

    keep_cols = [
        "timestamp", "line", "scheme",
        "CT", "FAN_SUM_S2", "CT_eff", "FAN_eff", "L",
        "power_value", "region",
        "is_valid_for_detection", "anomaly_rule",
        "L_hat", "sigma_local", "residual",
        "is_anomaly",
    ]
    return out[keep_cols].copy()


def infer_available_line_mapping(
    raw_df: pd.DataFrame,
    custom_line: Optional[str],
    custom_ct_col: Optional[str],
    custom_fan_col: Optional[str],
) -> Dict[str, Tuple[str, str]]:
    mapping: Dict[str, Tuple[str, str]] = {}

    for line, ct_col in DEFAULT_LINE_CT_COL.items():
        fan_col = DEFAULT_LINE_FAN_COL[line]
        if ct_col in raw_df.columns and fan_col in raw_df.columns:
            mapping[line] = (ct_col, fan_col)

    if custom_line and custom_ct_col and custom_fan_col:
        if custom_ct_col not in raw_df.columns:
            raise ValueError(f"自定义 CT 列不存在：{custom_ct_col}")
        if custom_fan_col not in raw_df.columns:
            raise ValueError(f"自定义 FAN 列不存在：{custom_fan_col}")
        mapping[custom_line] = (custom_ct_col, custom_fan_col)

    return mapping


def main():
    parser = argparse.ArgumentParser(description="基于 fit_model_* 输出文件做异常检测（仅方案 B，统一中位点插值版）")
    parser.add_argument("--input-csv", default=DEFAULT_INPUT_CSV, help="待检测 CSV 路径")
    parser.add_argument("--summary-csv", default=DEFAULT_SUMMARY_CSV, help="fit_model_summary.csv 路径")
    parser.add_argument("--interp-points-csv", default=DEFAULT_INTERP_POINTS_CSV, help="fit_model_interp_points.csv 路径")
    parser.add_argument("--sigma-csv", default=DEFAULT_SIGMA_CSV, help="兼容保留参数，当前版本不再使用")
    parser.add_argument("--low-lookup-csv", default=DEFAULT_LOW_LOOKUP_CSV, help="兼容保留参数，当前版本不再使用")
    parser.add_argument("--output-dir", default=".", help="输出目录")
    parser.add_argument("--line", default=None, help="仅检测某条线路，例如 BING / DING / WU")
    parser.add_argument("--custom-line", default=None, help="自定义线路编码")
    parser.add_argument("--ct-col", default=None, help="自定义 CT 列名")
    parser.add_argument("--fan-col", default=None, help="自定义 FAN 汇总列名")
    parser.add_argument("--z-threshold", type=float, default=2.0, help="|z_local| 异常阈值，默认 2.0")
    parser.add_argument("--sigma-floor", type=float, default=1e-6, help="sigma 下限，避免除零")
    parser.add_argument("--residual-abs-threshold", type=float, default=0.0, help="|residual| 绝对阈值(MW)")
    parser.add_argument("--ultra-low-loss-ratio", type=float, default=DEFAULT_ULTRA_LOW_LOSS_RATIO, help="超低功率段比例判异阈值，默认 0.9")
    args = parser.parse_args()

    raw_df = ensure_timestamp(read_csv_auto(args.input_csv))
    summary_df = read_csv_auto(args.summary_csv)
    interp_all_df = read_csv_auto(args.interp_points_csv)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    line_mapping = infer_available_line_mapping(
        raw_df=raw_df,
        custom_line=args.custom_line,
        custom_ct_col=args.ct_col,
        custom_fan_col=args.fan_col,
    )
    if not line_mapping:
        raise RuntimeError(
            "输入 CSV 中没有发现可用的线路列。\n"
            "若不是标准列名，请通过 --custom-line / --ct-col / --fan-col 指定。"
        )

    for df in [summary_df, interp_all_df]:
        df["line"] = df["line"].astype(str)
        df["scheme"] = df["scheme"].astype(str)

    model_df = summary_df[
        summary_df["line"].isin(line_mapping.keys()) &
        (summary_df["scheme"] == FIXED_SCHEME)
    ].copy()
    if args.line:
        model_df = model_df[model_df["line"] == args.line].copy()

    if len(model_df) == 0:
        raise RuntimeError(f"在模型汇总文件中没有找到匹配的 {FIXED_SCHEME} 方案。")

    detail_parts = []
    summary_rows = []

    print("=" * 72)
    print(f"开始异常检测（仅方案 {FIXED_SCHEME}）")
    print("=" * 72)
    print(f"输入文件: {args.input_csv}")
    print(f"可用线路: {sorted(line_mapping.keys())}")
    print(f"待检测模型数: {len(model_df)}")
    print(f"插值点文件: {args.interp_points_csv}")
    print(f"超低功率比例规则: BING<=2.0MW, DING<=1.5MW, WU<=3.0MW（含 0 MW）时，若 L < {args.ultra_low_loss_ratio} * 风机功率和 则异常")
    print(f"其余区间判异常阈值: 负损耗直接异常；或 |z_local| >= {args.z_threshold} 且 |residual| >= {args.residual_abs_threshold}")

    for _, model_row in model_df.iterrows():
        line = str(model_row["line"])
        ct_col, fan_col = line_mapping[line]

        line_df = build_line_df(raw_df, line=line, ct_col=ct_col, fan_col=fan_col)
        interp_points_df = interp_all_df[(interp_all_df["line"] == line) & (interp_all_df["scheme"] == FIXED_SCHEME)].copy()
        if len(interp_points_df) == 0:
            raise RuntimeError(f"{line}-{FIXED_SCHEME} 在插值点文件中没有可用记录：{args.interp_points_csv}")

        result_df = detect_for_b_model(
            line_df=line_df,
            model_row=model_row,
            interp_points_df=interp_points_df,
            z_threshold=args.z_threshold,
            sigma_floor=args.sigma_floor,
            residual_abs_threshold=args.residual_abs_threshold,
            ultra_low_loss_ratio=args.ultra_low_loss_ratio,
        )
        detail_parts.append(result_df)

        valid_cnt = int(result_df["is_valid_for_detection"].sum())
        anomaly_cnt = int(result_df["is_anomaly"].sum())
        anomaly_rate = float(anomaly_cnt / valid_cnt) if valid_cnt > 0 else np.nan

        summary_rows.append({
            "line": line,
            "scheme": FIXED_SCHEME,
            "ultra_low_power_threshold": float(ULTRA_LOW_POWER_THRESHOLDS.get(line, np.nan)),
            "ultra_low_loss_ratio": args.ultra_low_loss_ratio,
            "total_rows": len(result_df),
            "negative_loss_rows": int(line_df["is_negative_loss"].sum()),
            "ultra_low_rule_rows": int((result_df["anomaly_rule"] == "ultra_low_ratio_rule").sum()),
            "ultra_low_rule_anomaly_rows": int(((result_df["anomaly_rule"] == "ultra_low_ratio_rule") & result_df["is_anomaly"]).sum()),
            "valid_rows": valid_cnt,
            "anomaly_rows": anomaly_cnt,
            "anomaly_rate": anomaly_rate,
        })

        if valid_cnt > 0:
            neg_cnt = int(line_df["is_negative_loss"].sum())
            ultra_cnt = int((result_df["anomaly_rule"] == "ultra_low_ratio_rule").sum())
            ultra_anom_cnt = int(((result_df["anomaly_rule"] == "ultra_low_ratio_rule") & result_df["is_anomaly"]).sum())
            print(f"  ✅ {line}-{FIXED_SCHEME}: valid={valid_cnt:,}, ultra_low_zone={ultra_cnt:,}, ultra_low_anomaly={ultra_anom_cnt:,}, negative_loss={neg_cnt:,}, anomaly={anomaly_cnt:,}, rate={anomaly_rate:.4%}")
        else:
            print(f"  ✅ {line}-{FIXED_SCHEME}: 无有效点")

    detail_df = pd.concat(detail_parts, ignore_index=True) if detail_parts else pd.DataFrame()
    anomaly_only_df = detail_df[detail_df["is_anomaly"]].copy() if len(detail_df) > 0 else pd.DataFrame()
    summary_out_df = pd.DataFrame(summary_rows)
    detail_cn_df, anomaly_only_cn_df, summary_cn_df = localize_output_columns(
        detail_df=detail_df,
        anomaly_only_df=anomaly_only_df,
        summary_df=summary_out_df,
    )

    stem = Path(args.input_csv).stem
    detail_path = out_dir / f"{stem}_scheme_B_anomaly_detail.csv"
    anomaly_only_path = out_dir / f"{stem}_scheme_B_anomaly_only.csv"
    summary_path = out_dir / f"{stem}_scheme_B_anomaly_summary.csv"

    detail_cn_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    anomaly_only_cn_df.to_csv(anomaly_only_path, index=False, encoding="utf-8-sig")
    summary_cn_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print("\n输出文件：")
    print(f"  - 明细: {detail_path}")
    print(f"  - 仅异常点: {anomaly_only_path}")
    print(f"  - 汇总: {summary_path}")


if __name__ == "__main__":
    main()
