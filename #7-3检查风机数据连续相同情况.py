import os
import pandas as pd

"""
本脚本用于批量检测风电机组多个字段（如状态、功率、风速等）联合不变的重复段，并输出详细和汇总结果。
功能包括：
1. 自动提取风机编号，根据指定字段组合检测连续相同数据段。
2. 记录每个重复段的风机编号、组合值、起止时间及持续长度。
3. 将详细检测结果保存为Excel文件。
4. 汇总每台风机的联合重复总时长并保存为CSV文件。
"""

def extract_fan_numbers_sorted(file_path, field_prefixes):
    df = pd.read_csv(file_path, encoding='gbk')
    fan_numbers = set()
    for prefix in field_prefixes:
        fan_columns = [col for col in df.columns if prefix in col]
        for col in fan_columns:
            parts = col.split('#')
            if len(parts) > 1:
                try:
                    number = int(parts[1])
                    fan_numbers.add(number)
                except ValueError:
                    continue
    return sorted(fan_numbers)


def detect_joint_repeats(df, fan_numbers, field_prefixes, timestamp_col='timestamp', min_repeat=5):
    results = []
    for fan_num in fan_numbers:
        # 构建五个字段列名
        col_names = [f'{prefix}#{fan_num}' for prefix in field_prefixes]
        if not all(col in df.columns for col in col_names):
            continue

        current_joint_val = None
        count = 0
        start_idx = None

        for idx, row in df[col_names].iterrows():
            joint_val = tuple(row)
            if joint_val == current_joint_val:
                count += 1
            else:
                if count >= min_repeat:
                    results.append({
                        '风机编号': fan_num,
                        '重复值组合': current_joint_val,
                        '开始时间': df[timestamp_col].iloc[start_idx],
                        '结束时间': df[timestamp_col].iloc[idx - 1],
                        '持续长度': count
                    })
                current_joint_val = joint_val
                count = 1
                start_idx = idx

        if count >= min_repeat:
            results.append({
                '风机编号': fan_num,
                '重复值组合': current_joint_val,
                '开始时间': df[timestamp_col].iloc[start_idx],
                '结束时间': df[timestamp_col].iloc[len(df) - 1],
                '持续长度': count
            })
    return results


def save_joint_repeat_results_to_excel(results, save_path):
    if not results:
        print("📭 未检测到任何联合重复值段。")
        return

    df_results = pd.DataFrame(results)
    with pd.ExcelWriter(save_path, engine='openpyxl') as writer:
        df_results.to_excel(writer, sheet_name='联合重复检测结果', index=False)
    print(f"✅ 联合重复检测结果已保存：{save_path}")


def summarize_joint_repeats(results, save_path):
    if not results:
        print("📭 无联合重复值信息，无需汇总。")
        return

    df = pd.DataFrame(results)
    summary = df.groupby('风机编号')['持续长度'].sum().reset_index()
    summary = summary.rename(columns={'持续长度': '联合重复总长度'})

    summary.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"✅ 联合重复时长汇总已保存：{save_path}")


if __name__ == "__main__":
    # ⚠️ 修改为你的实际路径
    file_path = r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7补齐时间戳\峡阳B\#7峡阳B_20240315-20241224.csv"
    save_dir = r"G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7-2检查集电线路-全站功率数据连续相同情况\峡阳B"

    field_prefixes = [
        'STATUS_',
        'ACTIVE_POWER_',
        'REACTIVE_POWER_',
        'WINDSPEED_',
        'WINDDIRECTION_'
    ]

    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
    else:
        os.makedirs(save_dir, exist_ok=True)
        df = pd.read_csv(file_path, encoding='gbk', parse_dates=['timestamp'])

        fan_numbers = extract_fan_numbers_sorted(file_path, field_prefixes)
        print(f"提取的风机编号列表：{fan_numbers}")

        results = detect_joint_repeats(df, fan_numbers, field_prefixes, timestamp_col='timestamp', min_repeat=5)

        # 1. 保存联合检测详细结果到 Excel
        results_excel = os.path.join(save_dir, "联合重复值检测结果.xlsx")
        save_joint_repeat_results_to_excel(results, results_excel)

        # 2. 保存汇总到 CSV
        summary_csv = os.path.join(save_dir, "联合重复值总时长汇总.csv")
        summarize_joint_repeats(results, summary_csv)