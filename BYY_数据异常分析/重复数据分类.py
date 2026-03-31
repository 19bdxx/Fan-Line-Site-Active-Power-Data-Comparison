import pandas as pd

# =========================
# 参数区
# =========================
INPUT_FILE = "联合重复值检测结果.xlsx"
OUTPUT_FILE = "fan_repeat_six_types.csv"

FAN_COUNT_THRESHOLD = 70   # 大范围阈值
DURATION_THRESHOLD = 15    # 长时阈值（分钟）

# =========================
# 厂商、集电线路映射
# =========================
MINGYANG = set(range(63, 110)) | set(range(153, 169))
DONGQI   = set(range(110, 153))
JINFENG  = set(range(169, 200))

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
# 解析“重复值组合”
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

# =========================
# 六类分类
# =========================
def classify_type(simultaneous_count, duration_min):
    if simultaneous_count == 1:
        return "单机-长时" if duration_min > DURATION_THRESHOLD else "单机-短时"
    elif simultaneous_count > FAN_COUNT_THRESHOLD:
        return "多机-大范围-长时" if duration_min > DURATION_THRESHOLD else "多机-大范围-短时"
    else:
        return "多机-小范围-长时" if duration_min > DURATION_THRESHOLD else "多机-小范围-短时"

# =========================
# 主流程
# =========================
def main():
    df = pd.read_excel(INPUT_FILE)

    # 时间字段
    df["开始时间"] = pd.to_datetime(df["开始时间"])
    df["结束时间"] = pd.to_datetime(df["结束时间"])

    # 解析重复值组合
    parsed = df["重复值组合"].apply(parse_combo)
    parsed_df = pd.DataFrame(
        parsed.tolist(),
        columns=["状态码", "冻结有功kW", "冻结风速ms"],
        index=df.index
    )
    df = pd.concat([df, parsed_df], axis=1)

    # 厂商、集电线路
    df["厂商"] = df["风机编号"].apply(get_manufacturer)
    df["集电线路"] = df["风机编号"].apply(get_line)

    # 同时冻结风机数：按开始时间完全相同统计
    simultaneous_counts = df.groupby("开始时间")["风机编号"].nunique()
    df["同时冻结风机数"] = df["开始时间"].map(simultaneous_counts)

    # 分类
    df["异常类型"] = df.apply(
        lambda row: classify_type(
            simultaneous_count=row["同时冻结风机数"],
            duration_min=row["持续长度"]
        ),
        axis=1
    )

    # 只保留你要的 11 列
    output_df = df[
        [
            "风机编号",
            "厂商",
            "集电线路",
            "开始时间",
            "结束时间",
            "持续长度",
            "状态码",
            "冻结有功kW",
            "冻结风速ms",
            "同时冻结风机数",
            "异常类型"
        ]
    ].copy()

    # 重命名持续长度列
    output_df = output_df.rename(columns={"持续长度": "持续长度(min)"})

    # 排序
    output_df = output_df.sort_values(["开始时间", "风机编号"]).reset_index(drop=True)

    # 输出
    output_df.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("分类完成。")
    print(f"输出文件：{OUTPUT_FILE}")
    print("输出列：")
    print(output_df.columns.tolist())

if __name__ == "__main__":
    main()