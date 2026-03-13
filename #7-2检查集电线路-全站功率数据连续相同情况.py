import pandas as pd
import os

# 输入文件
file_path = r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7-1提取场站集电线路-全站有功\峡阳B\#7-1峡阳B_20240315-20241224.csv"


# 连续相同最少条数，达到这个值才记录
min_repeat = 5

# 需要提取的列
columns_to_extract = [
    "timestamp",
    "ACTIVE_POWER_BING", 
    "ACTIVE_POWER_DING", 
    "ACTIVE_POWER_WU", 
    "ACTIVE_POWER_STATION",
    "LIMIT_POWER"
]

# 读取CSV
df = pd.read_csv(file_path)

# 清理列名（防止空格/BOM问题）
df.columns = df.columns.astype(str).str.strip().str.replace('\ufeff', '', regex=False)

# 检查缺失列
missing_cols = [c for c in columns_to_extract if c not in df.columns]
if missing_cols:
    print("以下列不存在：", missing_cols)

# 只保留存在的列
cols_exist = [c for c in columns_to_extract if c in df.columns]
df_extract = df[cols_exist].copy()

# 检查 timestamp 列
if "timestamp" not in df_extract.columns:
    raise ValueError("未找到 timestamp 列，无法进行时间排序和连续重复检查。")

# 转换时间列
df_extract["timestamp"] = pd.to_datetime(df_extract["timestamp"], errors="coerce")

# 按时间排序
df_extract = df_extract.sort_values(by="timestamp").reset_index(drop=True)

# =========================
# 检查 timestamp 是否相邻重复
# =========================
dup_timestamp_mask = df_extract["timestamp"].eq(df_extract["timestamp"].shift(1))

if dup_timestamp_mask.any():
    df_dup_timestamp = df_extract[dup_timestamp_mask].copy()
    dup_timestamp_path = "重复时间戳记录-峡沙.csv"
    df_dup_timestamp.to_csv(dup_timestamp_path, index=False, encoding="gbk")
    print(f"⚠️ 发现相邻重复 timestamp，已保存：{dup_timestamp_path}")
else:
    print("✅ 未发现相邻重复 timestamp")

# =========================
# 检查每一列是否存在连续相同
# =========================
repeat_results = []

target_cols = [c for c in cols_exist if c != "timestamp"]

for col in target_cols:
    series = df_extract[col].copy()

    # 缺失值统一处理，避免 NaN != NaN
    series = series.astype(object).where(pd.notna(series), "__MISSING__")

    # 当前值是否与上一行相同
    same_as_prev = series.eq(series.shift(1))

    # 不同则开新组
    group_id = (~same_as_prev).cumsum()

    # 按组遍历
    for _, group in df_extract.groupby(group_id, sort=False):
        group_len = len(group)

        if group_len >= min_repeat:
            start_idx = group.index[0]
            end_idx = group.index[-1]

            repeat_results.append({
                "字段名": col,
                "重复值": df_extract.at[start_idx, col],
                "开始时间": df_extract.at[start_idx, "timestamp"],
                "结束时间": df_extract.at[end_idx, "timestamp"],
                "持续长度": group_len
            })

# 保存连续重复明细
if repeat_results:
    df_repeat = pd.DataFrame(repeat_results)
    repeat_detail_path = "G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7-2检查集电线路-全站功率数据连续相同情况\峡阳B\每列连续重复检测结果.csv"
    df_repeat.to_csv(repeat_detail_path, index=False, encoding="gbk")
    print(f"✅ 每列连续重复明细已保存：{repeat_detail_path}")

    # 汇总
    df_summary = df_repeat.groupby("字段名", as_index=False).agg(
        重复段数量=("字段名", "count"),
        连续重复总长度=("持续长度", "sum")
    )

    repeat_summary_path = "G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7-2检查集电线路-全站功率数据连续相同情况\峡阳B\每列连续重复汇总.csv"
    df_summary.to_csv(repeat_summary_path, index=False, encoding="gbk")
    print(f"✅ 每列连续重复汇总已保存：{repeat_summary_path}")
else:
    print(f"📭 未发现持续长度 >= {min_repeat} 的连续重复段。")