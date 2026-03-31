import argparse
from collections import defaultdict
from pathlib import Path

import pandas as pd

INPUT_FILE = r"RAW_DATA\联合重复值检测结果.xlsx"
ANOMALY_DETAIL_FILE = r"#7-1峡阳B_20240315-20241224_with_sum_scheme_B_anomaly_detail.csv"

SEGMENT_OUTPUT = "freeze_count_segments_v3.csv"
DETAIL_OUTPUT = "freeze_count_segments_detail_v3.csv"
TIMESERIES_OUTPUT = "freeze_count_timeseries_v3.csv"

TARGET_LINES = ["BING", "DING", "WU"]


# =========================
# 厂商 / 集电线路映射
# =========================
MINGYANG = set(range(63, 110)) | set(range(153, 169))
DONGQI = set(range(110, 153))
JINFENG = set(range(169, 200))


def get_manufacturer(fan_num):
    try:
        fan_num = int(fan_num)
    except Exception:
        return "未知"

    if fan_num in MINGYANG:
        return "明阳"
    elif fan_num in DONGQI:
        return "东气"
    elif fan_num in JINFENG:
        return "金风"
    else:
        return "未知"


def get_line(fan_num):
    try:
        fan_num = int(fan_num)
    except Exception:
        return "未知"

    if 63 <= fan_num <= 109:
        return "WU"
    elif 110 <= fan_num <= 152:
        return "DING"
    elif 153 <= fan_num <= 199:
        return "BING"
    else:
        return "未知"


# =========================
# 解析重复值组合
# 预期格式：
# (status, active_power, reactive_power, windspeed, winddirection)
# =========================
def parse_combo(combo):
    try:
        vals = str(combo).strip("()").split(",")
        vals = [v.strip() for v in vals]

        def to_float(x):
            if x is None or x == "":
                return None
            return float(x)

        status = to_float(vals[0]) if len(vals) > 0 else None
        active_power = to_float(vals[1]) if len(vals) > 1 else None
        windspeed = to_float(vals[3]) if len(vals) > 3 else None
        return status, active_power, windspeed
    except Exception:
        return None, None, None


def safe_nonnegative_power(x):
    if pd.isna(x):
        return 0.0
    try:
        return max(float(x), 0.0)
    except Exception:
        return 0.0


def calc_line_power_sums_mw(active_df: pd.DataFrame) -> dict:
    """
    分别统计当前分段内 BING / DING / WU 重复风机的非负功率和（MW）。
    功率口径：max(冻结有功kW, 0) 后再按线路分别求和。
    """
    result = {line: 0.0 for line in TARGET_LINES}
    if active_df is None or len(active_df) == 0:
        return result

    tmp = active_df.copy()
    if "非负冻结有功kW" not in tmp.columns:
        tmp["非负冻结有功kW"] = tmp["冻结有功kW"].apply(safe_nonnegative_power)

    tmp["集电线路"] = tmp["集电线路"].astype(str).str.strip().str.upper()
    grouped = tmp.groupby("集电线路")["非负冻结有功kW"].sum()

    for line in TARGET_LINES:
        result[line] = round(float(grouped.get(line, 0.0)) / 1000.0, 6)

    return result


def format_detail_row(row):
    fan = row["风机编号"]
    manu = row["厂商"]
    line = row["集电线路"]
    status = row["状态码"]
    p = row["冻结有功kW"]
    ws = row["冻结风速ms"]
    return f"{fan}[{manu},{line},status={status},P={p},WS={ws}]"


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


def parse_bool_like(val) -> bool:
    if pd.isna(val):
        return False
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)

    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "y", "是", "异常", "true."}:
        return True
    if s in {"false", "0", "no", "n", "否", "正常", ""}:
        return False
    return False


def build_line_anomaly_minute_table(anomaly_detail_path: str, seg_min: pd.Timestamp, seg_max: pd.Timestamp) -> pd.DataFrame:
    """
    读取异常明细，生成按分钟索引、按线路展开的异常标记表。
    输出列：BING, DING, WU（值为 0/1）
    """
    anomaly_df = read_csv_auto(anomaly_detail_path)

    required_cols = {"时间", "线路", "是否异常"}
    missing = required_cols - set(anomaly_df.columns)
    if missing:
        raise ValueError(f"异常明细文件缺少必要列：{missing}")

    anomaly_df = anomaly_df[["时间", "线路", "是否异常"]].copy()
    anomaly_df["时间"] = pd.to_datetime(anomaly_df["时间"], errors="coerce")
    anomaly_df = anomaly_df.dropna(subset=["时间", "线路"]).copy()
    anomaly_df["线路"] = anomaly_df["线路"].astype(str).str.strip().str.upper()
    anomaly_df = anomaly_df[anomaly_df["线路"].isin(TARGET_LINES)].copy()
    anomaly_df["是否异常"] = anomaly_df["是否异常"].apply(parse_bool_like).astype(int)

    if len(anomaly_df) == 0:
        full_index = pd.date_range(start=seg_min, end=seg_max, freq="min")
        return pd.DataFrame(index=full_index, data={line: 0 for line in TARGET_LINES})

    minute_line_df = (
        anomaly_df.groupby(["时间", "线路"], as_index=False)["是否异常"]
        .max()
        .pivot(index="时间", columns="线路", values="是否异常")
        .sort_index()
        .fillna(0)
    )

    full_index = pd.date_range(start=seg_min, end=seg_max, freq="min")
    minute_line_df = minute_line_df.reindex(full_index, fill_value=0)

    for line in TARGET_LINES:
        if line not in minute_line_df.columns:
            minute_line_df[line] = 0

    return minute_line_df[TARGET_LINES].astype(int)


def append_line_anomaly_stats(segment_df: pd.DataFrame, anomaly_detail_path: str) -> pd.DataFrame:
    """
    对每个“线路独立分段”补充该时间段内三条线路的异常分钟数/异常占比。
    分段是独立生成的，但异常占比仍保留三条线路，方便后续全站视角对比。
    """
    if len(segment_df) == 0:
        return segment_df.copy()

    seg_min = pd.to_datetime(segment_df["开始时间"]).min()
    seg_max = pd.to_datetime(segment_df["结束时间"]).max()
    minute_line_df = build_line_anomaly_minute_table(anomaly_detail_path, seg_min, seg_max)
    cum_df = minute_line_df.cumsum()

    out = segment_df.copy()

    for line in TARGET_LINES:
        out[f"{line}异常分钟数"] = 0
        out[f"{line}异常占比"] = 0.0

    out["三线路异常分钟数之和"] = 0
    out["三线路最大异常占比"] = 0.0

    for idx, row in out.iterrows():
        seg_start = pd.to_datetime(row["开始时间"])
        seg_end = pd.to_datetime(row["结束时间"])
        duration_min = int(row["持续时长(min)"])

        if pd.isna(seg_start) or pd.isna(seg_end) or duration_min <= 0:
            continue

        end_vals = cum_df.loc[seg_end]
        prev_t = seg_start - pd.Timedelta(minutes=1)
        if prev_t in cum_df.index:
            start_vals = cum_df.loc[prev_t]
        else:
            start_vals = pd.Series({line: 0 for line in TARGET_LINES})

        total_abn_min = 0
        max_ratio = 0.0
        for line in TARGET_LINES:
            abn_min = int(end_vals[line] - start_vals[line])
            ratio = abn_min / duration_min if duration_min > 0 else 0.0
            out.at[idx, f"{line}异常分钟数"] = abn_min
            out.at[idx, f"{line}异常占比"] = round(ratio, 6)
            total_abn_min += abn_min
            max_ratio = max(max_ratio, ratio)

        out.at[idx, "三线路异常分钟数之和"] = total_abn_min
        out.at[idx, "三线路最大异常占比"] = round(max_ratio, 6)

    return out


def build_segments_for_one_line(df_line: pd.DataFrame, line_code: str, global_seg_start_id: int = 1):
    """
    对单条集电线路独立做扫描线分段。
    切段边界只来自该线路上的风机重复区间，不再受其他线路影响。
    """
    if df_line is None or len(df_line) == 0:
        return [], [], global_seg_start_id

    events = defaultdict(int)
    start_map = defaultdict(list)
    end_map = defaultdict(list)

    for idx, row in df_line.iterrows():
        start = row["开始时间"]
        end = row["结束时间"]

        events[start] += 1
        events[end + pd.Timedelta(minutes=1)] -= 1

        start_map[start].append(idx)
        end_map[end + pd.Timedelta(minutes=1)].append(idx)

    sorted_times = sorted(events.keys())
    if len(sorted_times) < 2:
        return [], [], global_seg_start_id

    active_set = set()
    segments = []
    detail_rows = []
    next_global_id = global_seg_start_id
    next_local_id = 1

    for i, t in enumerate(sorted_times):
        for idx in start_map.get(t, []):
            active_set.add(idx)

        for idx in end_map.get(t, []):
            active_set.discard(idx)

        current_count = len(active_set)

        if i >= len(sorted_times) - 1:
            continue

        next_t = sorted_times[i + 1]
        if not (t < next_t):
            continue

        seg_start = t
        seg_end = next_t - pd.Timedelta(minutes=1)
        duration_min = int((seg_end - seg_start).total_seconds() / 60) + 1

        active_indices = sorted(active_set, key=lambda x: int(df_line.loc[x, "风机编号"]))

        if active_indices:
            active_df = df_line.loc[active_indices].copy()

            fan_list = [str(x) for x in active_df["风机编号"].tolist()]
            manu_list = sorted(set(active_df["厂商"].tolist()))
            line_list = sorted(set(active_df["集电线路"].tolist()))
            line_count = len(line_list)
            detail_list = [format_detail_row(active_df.loc[idx]) for idx in active_df.index]

            power_sum_mw = round(active_df["非负冻结有功kW"].sum() / 1000.0, 6)
            line_power_sums_mw = calc_line_power_sums_mw(active_df)
        else:
            fan_list = []
            manu_list = []
            line_list = []
            line_count = 0
            detail_list = []
            power_sum_mw = 0.0
            line_power_sums_mw = {line: 0.0 for line in TARGET_LINES}

        segment_row = {
            "分段ID": next_global_id,
            "线路内分段ID": next_local_id,
            "所属集电线路": line_code,
            "开始时间": seg_start,
            "结束时间": seg_end,
            "持续时长(min)": duration_min,
            "冻结风机数": current_count,
            "集电线路列表数量": line_count,
            "各风机功率之和MW": power_sum_mw,
            "BING重复风机功率之和MW": line_power_sums_mw["BING"],
            "DING重复风机功率之和MW": line_power_sums_mw["DING"],
            "WU重复风机功率之和MW": line_power_sums_mw["WU"],
            "风机编号列表": "、".join(fan_list),
            "厂商列表": "、".join(manu_list),
            "集电线路列表": "、".join(line_list),
            "重复详情": " 、 ".join(detail_list),
        }
        segments.append(segment_row)

        if active_indices:
            for idx in active_indices:
                detail_rows.append({
                    "分段ID": next_global_id,
                    "线路内分段ID": next_local_id,
                    "所属集电线路": line_code,
                    "开始时间": seg_start,
                    "结束时间": seg_end,
                    "持续时长(min)": duration_min,
                    "冻结风机数": current_count,
                    "集电线路列表数量": line_count,
                    "各风机功率之和MW": power_sum_mw,
                    "BING重复风机功率之和MW": line_power_sums_mw["BING"],
                    "DING重复风机功率之和MW": line_power_sums_mw["DING"],
                    "WU重复风机功率之和MW": line_power_sums_mw["WU"],
                    "风机编号列表": "、".join(fan_list),
                    "厂商列表": "、".join(manu_list),
                    "集电线路列表": "、".join(line_list),
                    "风机编号": df_line.loc[idx, "风机编号"],
                    "厂商": df_line.loc[idx, "厂商"],
                    "集电线路": df_line.loc[idx, "集电线路"],
                    "状态码": df_line.loc[idx, "状态码"],
                    "冻结有功kW": df_line.loc[idx, "冻结有功kW"],
                    "冻结风速ms": df_line.loc[idx, "冻结风速ms"],
                    "非负冻结有功kW": df_line.loc[idx, "非负冻结有功kW"],
                    "重复值组合": df_line.loc[idx, "重复值组合"],
                    "原始开始时间": df_line.loc[idx, "开始时间"],
                    "原始结束时间": df_line.loc[idx, "结束时间"],
                    "原始持续长度(min)": df_line.loc[idx, "持续长度"],
                })
        else:
            detail_rows.append({
                "分段ID": next_global_id,
                "线路内分段ID": next_local_id,
                "所属集电线路": line_code,
                "开始时间": seg_start,
                "结束时间": seg_end,
                "持续时长(min)": duration_min,
                "冻结风机数": current_count,
                "集电线路列表数量": line_count,
                "各风机功率之和MW": power_sum_mw,
                "BING重复风机功率之和MW": line_power_sums_mw["BING"],
                "DING重复风机功率之和MW": line_power_sums_mw["DING"],
                "WU重复风机功率之和MW": line_power_sums_mw["WU"],
                "风机编号列表": "",
                "厂商列表": "",
                "集电线路列表": "",
                "风机编号": None,
                "厂商": None,
                "集电线路": None,
                "状态码": None,
                "冻结有功kW": None,
                "冻结风速ms": None,
                "非负冻结有功kW": None,
                "重复值组合": None,
                "原始开始时间": None,
                "原始结束时间": None,
                "原始持续长度(min)": None,
            })

        next_global_id += 1
        next_local_id += 1

    return segments, detail_rows, next_global_id


def build_timeseries_from_segments(segment_df: pd.DataFrame) -> pd.DataFrame:
    ts_list = []
    for _, row in segment_df.iterrows():
        rng = pd.date_range(start=row["开始时间"], end=row["结束时间"], freq="min")
        ts_list.append(pd.DataFrame({
            "时间": rng,
            "所属集电线路": row["所属集电线路"],
            "冻结风机数": row["冻结风机数"],
        }))

    if ts_list:
        return pd.concat(ts_list, ignore_index=True)
    return pd.DataFrame(columns=["时间", "所属集电线路", "冻结风机数"])


def main():
    parser = argparse.ArgumentParser(description="生成冻结分段表（BING/DING/WU 三条线路独立分段），并统计每个分段内三条线路异常占比")
    parser.add_argument("--input-file", default=INPUT_FILE, help="联合重复值检测结果.xlsx 路径")
    parser.add_argument(
        "--anomaly-detail-file",
        default=ANOMALY_DETAIL_FILE,
        help="detect_anomalies_B_only_cn_插值点版_v5.py 生成的异常明细 CSV 路径",
    )
    parser.add_argument("--segment-output", default=SEGMENT_OUTPUT, help="分段表输出路径")
    parser.add_argument("--detail-output", default=DETAIL_OUTPUT, help="分段明细表输出路径")
    parser.add_argument("--timeseries-output", default=TIMESERIES_OUTPUT, help="趋势时序表输出路径")
    args = parser.parse_args()

    # 1. 读取
    df = pd.read_excel(args.input_file)
    df["开始时间"] = pd.to_datetime(df["开始时间"])
    df["结束时间"] = pd.to_datetime(df["结束时间"])

    df = df.dropna(subset=["开始时间", "结束时间"]).copy()
    df = df[df["结束时间"] >= df["开始时间"]].copy()
    df = df.reset_index(drop=True)

    # 2. 补字段
    parsed = df["重复值组合"].apply(parse_combo)
    parsed_df = pd.DataFrame(
        parsed.tolist(),
        columns=["状态码", "冻结有功kW", "冻结风速ms"],
        index=df.index,
    )
    df = pd.concat([df, parsed_df], axis=1)
    df["厂商"] = df["风机编号"].apply(get_manufacturer)
    df["集电线路"] = df["风机编号"].apply(get_line)

    # 功率按 max(功率, 0)
    df["非负冻结有功kW"] = df["冻结有功kW"].apply(safe_nonnegative_power)
    df["集电线路"] = df["集电线路"].astype(str).str.strip().str.upper()

    # 3. 对每条线路独立分段
    all_segments = []
    all_detail_rows = []
    next_global_id = 1

    for line_code in TARGET_LINES:
        df_line = df[df["集电线路"] == line_code].copy()
        segs, details, next_global_id = build_segments_for_one_line(
            df_line=df_line,
            line_code=line_code,
            global_seg_start_id=next_global_id,
        )
        all_segments.extend(segs)
        all_detail_rows.extend(details)

    # 4. 输出分段表（补充每段内三条线路的异常分钟数/异常占比）
    segment_df = pd.DataFrame(all_segments)
    if len(segment_df) > 0:
        segment_df = segment_df.sort_values(["所属集电线路", "开始时间", "结束时间", "分段ID"]).reset_index(drop=True)
        segment_df = append_line_anomaly_stats(segment_df, args.anomaly_detail_file)
    segment_df.to_csv(args.segment_output, index=False, encoding="utf-8-sig")

    # 5. 输出分段明细表
    detail_df = pd.DataFrame(all_detail_rows)
    if len(detail_df) > 0:
        detail_df = detail_df.sort_values(["所属集电线路", "开始时间", "结束时间", "分段ID", "风机编号"], na_position="last").reset_index(drop=True)
    detail_df.to_csv(args.detail_output, index=False, encoding="utf-8-sig")

    # 6. 输出按分钟趋势表
    ts_df = build_timeseries_from_segments(segment_df)
    ts_df.to_csv(args.timeseries_output, index=False, encoding="utf-8-sig")

    print("完成：")
    print(f"- 分段表：{args.segment_output}")
    print(f"- 分段明细表：{args.detail_output}")
    print(f"- 趋势时序表：{args.timeseries_output}")
    print(f"- 使用异常明细：{args.anomaly_detail_file}")
    print("说明：当前分段已改为 BING / DING / WU 三条线路各自独立分段。")
    print("\n分段表示例：")
    print(segment_df.head().to_string(index=False))


if __name__ == "__main__":
    main()
