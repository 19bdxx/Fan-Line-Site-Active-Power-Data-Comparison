import pandas as pd
import os

def split_csv(file_path):
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])  # 假设有一个'timestamp'列
    df = df.sort_values(by='timestamp')
    
    split_dates = df['timestamp'].quantile([0, 0.25, 0.5, 0.75, 1]).tolist()
    base_name = os.path.splitext(file_path)[0]

    for i in range(4):
        df_part = df[(df['timestamp'] >= split_dates[i]) & (df['timestamp'] < split_dates[i + 1])]
        start_time = split_dates[i].strftime('%Y%m%d_%H%M%S')
        end_time = split_dates[i + 1].strftime('%Y%m%d_%H%M%S')
        df_part.to_csv(f"{base_name}_part{i + 1}_{start_time}_{end_time}.csv", index=False)

# Example usage
file_path = "G:\WindPowerForecast\#1场站数据下载\代码-从日志提取\广东\code_下载\#7-1提取场站集电线路-全站有功\峡阳B\#7-1峡阳B_20240315-20241224_with_sum.csv"
split_csv(file_path)