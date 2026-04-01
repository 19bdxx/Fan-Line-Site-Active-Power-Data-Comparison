"""
功能说明：
本脚本用于批量处理风电场 SCADA CSV 数据，实现数据提取与功率汇总计算。

主要功能：
1. 提取指定列（如 timestamp、集电线路功率、全站功率、限电功率等）。
2. 自动识别所有风机有功列，列名格式为：ACTIVE_POWER_#数字。
3. 计算所有风机有功总和，并生成两种策略结果：
   - FAN_ACTIVE_POWER_SUM_S1：正值保留，负值保留
   - FAN_ACTIVE_POWER_SUM_S2：正值保留，负值按0处理
4. 利用集电线路测点计算全站线路功率总和，并生成两种策略结果：
   - LINE_ACTIVE_POWER_SUM_S1：正值保留，负值保留
   - LINE_ACTIVE_POWER_SUM_S2：正值保留，负值按0处理
5. 按场站配置的“线路-风机编号映射”，计算各线路对应风机功率总和，并生成两种策略结果：
   - 例如 JIA_ACTIVE_POWER_SUM_S1 / JIA_ACTIVE_POWER_SUM_S2
6. 支持峡阳A对 ACTIVE_POWER_JIA、ACTIVE_POWER_YI 先取反。
7. 批量处理多个场站文件，并自动创建输出目录。
8. 输出新的 CSV 文件，文件名在原有基础上增加后缀 "_with_sum"。
"""

import pandas as pd
import os
import re


def add_suffix_to_filename(file_path, suffix="_with_sum"):
    """在文件名后缀前插入 suffix，例如 xxx.csv -> xxx_with_sum.csv"""
    base, ext = os.path.splitext(file_path)
    return f"{base}{suffix}{ext}"


def expand_ranges(ranges):
    """
    将 [(1, 15), (40, 40), (45, 48)] 展开为 [1,2,3,...,15,40,45,46,47,48]
    """
    result = []
    for start, end in ranges:
        result.extend(range(start, end + 1))
    return result


def fan_num_to_col(fan_num):
    """风机编号转列名，如 63 -> ACTIVE_POWER_#63"""
    return f"ACTIVE_POWER_#{fan_num}"


def calculate_sum_by_strategy(df_part, strategy):
    """
    strategy = 1: 正值保留，负值保留
    strategy = 2: 正值保留，负值按0处理
    """
    df_num = df_part.apply(pd.to_numeric, errors="coerce")

    if strategy == 1:
        return df_num.sum(axis=1, skipna=True)
    elif strategy == 2:
        return df_num.clip(lower=0).sum(axis=1, skipna=True)
    else:
        raise ValueError("strategy 参数只能是 1 或 2")


def get_all_fan_cols(df):
    """自动识别所有风机有功列：ACTIVE_POWER_#数字"""
    return [col for col in df.columns if re.fullmatch(r"ACTIVE_POWER_#\d+", str(col))]


def get_line_measurement_cols(df, selected_columns):
    """
    识别用于‘线路测点求和’的列：
    从 selected_columns 中筛选出线路测点列，排除 timestamp / ACTIVE_POWER_STATION / LIMIT_POWER / 风机列
    """
    line_cols = []
    for col in selected_columns:
        if col not in df.columns:
            continue
        if col in ["timestamp", "ACTIVE_POWER_STATION", "LIMIT_POWER"]:
            continue
        if re.fullmatch(r"ACTIVE_POWER_#\d+", str(col)):
            continue
        if str(col).startswith("ACTIVE_POWER_"):
            line_cols.append(col)
    return line_cols


def apply_invert_lines(df, invert_lines):
    """对指定线路列取反"""
    for col in invert_lines:
        if col in df.columns:
            df[col] = -pd.to_numeric(df[col], errors="coerce")


def add_group_sum_columns(extracted_df, df_source, group_name, fan_numbers):
    """
    按风机编号集合计算线路对应风机功率汇总，并写入：
    {group_name}_ACTIVE_POWER_SUM_S1
    {group_name}_ACTIVE_POWER_SUM_S2
    """
    fan_cols = [fan_num_to_col(n) for n in fan_numbers if fan_num_to_col(n) in df_source.columns]

    col_s1 = f"{group_name}_ACTIVE_POWER_SUM_S1"
    col_s2 = f"{group_name}_ACTIVE_POWER_SUM_S2"

    if fan_cols:
        extracted_df[col_s1] = calculate_sum_by_strategy(df_source[fan_cols], strategy=1)/1000
        extracted_df[col_s2] = calculate_sum_by_strategy(df_source[fan_cols], strategy=2)/1000
    else:
        extracted_df[col_s1] = pd.NA
        extracted_df[col_s2] = pd.NA

    return fan_cols


def process_one_file(site_name, file_path, selected_columns, output_file, site_config):
    df = pd.read_csv(file_path)

    # 1) 场站特殊处理：线路测点取反
    invert_lines = site_config.get("invert_lines", [])
    if invert_lines:
        apply_invert_lines(df, invert_lines)

    # 2) 提取指定列（只提取存在的列，避免报错）
    existing_selected_columns = [col for col in selected_columns if col in df.columns]
    missing_selected_columns = [col for col in selected_columns if col not in df.columns]

    extracted_df = df[existing_selected_columns].copy()

    # 3) 自动识别所有风机列，并计算风机总和
    fan_cols = get_all_fan_cols(df)
    if fan_cols:
        extracted_df["FAN_ACTIVE_POWER_SUM_S1"] = calculate_sum_by_strategy(df[fan_cols], strategy=1)/1000
        extracted_df["FAN_ACTIVE_POWER_SUM_S2"] = calculate_sum_by_strategy(df[fan_cols], strategy=2)/1000
    else:
        extracted_df["FAN_ACTIVE_POWER_SUM_S1"] = pd.NA
        extracted_df["FAN_ACTIVE_POWER_SUM_S2"] = pd.NA

    # 4) 计算线路测点总和
    line_measurement_cols = get_line_measurement_cols(df, selected_columns)
    if line_measurement_cols:
        extracted_df["LINE_ACTIVE_POWER_SUM_S1"] = calculate_sum_by_strategy(df[line_measurement_cols], strategy=1)
        extracted_df["LINE_ACTIVE_POWER_SUM_S2"] = calculate_sum_by_strategy(df[line_measurement_cols], strategy=2)
    else:
        extracted_df["LINE_ACTIVE_POWER_SUM_S1"] = pd.NA
        extracted_df["LINE_ACTIVE_POWER_SUM_S2"] = pd.NA

    # 5) 根据场站配置，计算各线路对应风机的功率总和
    fan_groups = site_config.get("fan_groups", {})
    group_result_info = {}

    for group_name, fan_numbers in fan_groups.items():
        used_fan_cols = add_group_sum_columns(extracted_df, df, group_name, fan_numbers)
        group_result_info[group_name] = used_fan_cols

    # 6) 创建输出目录
    parent = os.path.dirname(output_file)
    if parent and not os.path.exists(parent):
        os.makedirs(parent)

    # 7) 输出 CSV
    extracted_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    # 8) 打印日志
    print("=" * 80)
    print(f"场站: {site_name}")
    print(f"输入文件: {file_path}")
    print(f"输出文件: {output_file}")
    print(f"提取列: {existing_selected_columns}")
    if missing_selected_columns:
        print(f"缺失列(已跳过): {missing_selected_columns}")
    print(f"自动识别风机列数量: {len(fan_cols)}")
    print(f"线路测点列: {line_measurement_cols}")
    if invert_lines:
        print(f"已取反线路列: {invert_lines}")
    if fan_groups:
        print("线路对应风机分组统计:")
        for group_name, used_cols in group_result_info.items():
            print(f"  {group_name}: 使用风机列 {len(used_cols)} 个")
    print("处理完成")


# =========================================================
# 场站配置
# =========================================================
SITE_CONFIG = {
    "峡阳A": {
        # 先对 ACTIVE_POWER_JIA、ACTIVE_POWER_YI 取反
        "invert_lines": ["ACTIVE_POWER_JIA", "ACTIVE_POWER_YI"],
        "fan_groups": {
            # 文档最新要求：
            # JIA：1-15、40、45-48、52-62
            # YI ：16-39、41-44、49-51
            "JIA": expand_ranges([(1, 15), (40, 40), (45, 48), (52, 62)]),
            "YI": expand_ranges([(16, 39), (41, 44), (49, 51)]),
        }
    },
    "峡阳B": {
        "invert_lines": [],
        "fan_groups": {
            # BING：153-199
            # DING：110-152
            # WU  ：63-109
            "BING": expand_ranges([(153, 199)]),
            "DING": expand_ranges([(110, 152)]),
            "WU": expand_ranges([(63, 109)]),
        }
    },
    "峡沙": {
        "invert_lines": [],
        "fan_groups": {
            # JIA：14-18、22-31、37-46、52-59、64-66
            # YI ：1-13、19-21、32-36、47-51、60-63、67-70
            "JIA": expand_ranges([(14, 18), (22, 31), (37, 46), (52, 59), (64, 66)]),
            "YI": expand_ranges([(1, 13), (19, 21), (32, 36), (47, 51), (60, 63), (67, 70)]),
        }
    },
    "蕴阳": {
        "invert_lines": [],
        # 当前文档未给出线路与风机编号映射，因此这里只计算：
        # FAN_ACTIVE_POWER_SUM_S1/S2
        # LINE_ACTIVE_POWER_SUM_S1/S2
        "fan_groups": {}
    },
    "橘子塘": {
        "invert_lines": [],
        # 当前未给出线路-风机映射
        "fan_groups": {}
    }
}


# =========================================================
# 批量处理文件列表
# =========================================================
FILES = [
    (
        "峡沙",
        r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7补齐时间戳\峡沙\#7峡沙_20240315-20241224.csv",
        ["timestamp", "ACTIVE_POWER_JIA", "ACTIVE_POWER_YI", "LIMIT_POWER"],
        r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7-1提取场站集电线路-全站有功\峡沙\#7-1峡沙_20240315-20241224.csv",
    ),
    (
        "峡阳A",
        r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7补齐时间戳\峡阳A\#7峡阳A_20240315-20241224.csv",
        ["timestamp", "ACTIVE_POWER_JIA", "ACTIVE_POWER_YI", "LIMIT_POWER"],
        r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7-1提取场站集电线路-全站有功\峡阳A\#7-1峡阳A_20240315-20241224.csv",
    ),
    (
        "峡阳B",
        r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7补齐时间戳\峡阳B\#7峡阳B_20240315-20241224.csv",
        ["timestamp", "ACTIVE_POWER_BING", "ACTIVE_POWER_DING", "ACTIVE_POWER_WU", "LIMIT_POWER"],
        r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7-1提取场站集电线路-全站有功\峡阳B\#7-1峡阳B_20240315-20241224.csv",
    ),
    (
        "蕴阳",
        r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7补齐时间戳\蕴阳\#7蕴阳_20240315-20241224.csv",
        ["timestamp", "ACTIVE_POWER_JIA", "ACTIVE_POWER_YI", "LIMIT_POWER"],
        r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7-1提取场站集电线路-全站有功\蕴阳\#7-1蕴阳_20240315-20241224.csv",
    ),
    (
        "橘子塘",
        r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7补齐时间戳\橘子塘\#7橘子塘_20240315-20241224.csv",
        ["timestamp", "ACTIVE_POWER_JIA", "LIMIT_POWER"],
        r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7-1提取场站集电线路-全站有功\橘子塘\#7-1橘子塘_20240315-20241224.csv",
    ),
]


# =========================================================
# 主程序
# =========================================================
if __name__ == "__main__":
    for site_name, file_path, selected_columns, output_file in FILES:
        site_config = SITE_CONFIG.get(site_name, {"invert_lines": [], "fan_groups": {}})
        new_output_file = add_suffix_to_filename(output_file, "_with_sum")
        process_one_file(site_name, file_path, selected_columns, new_output_file, site_config)