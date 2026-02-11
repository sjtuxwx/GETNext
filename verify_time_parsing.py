import pandas as pd
from datetime import datetime

# 读取数据
df = pd.read_csv('dataset/NYC/NYC_train.csv')

print("="*80)
print("验证时间戳解析的正确性")
print("="*80)

# 查看前20条数据的时间信息
print("\n【查看原始数据的时间字段】")
print("数据包含的时间相关字段：")
print(df.columns.tolist())

print("\n【前20条数据的时间信息】")
sample = df.head(20)

for idx, row in sample.iterrows():
    print(f"\n记录 {idx}:")
    print(f"  user_id: {row['user_id']}")
    print(f"  trajectory_id: {row['trajectory_id']}")
    print(f"  UTC_time: {row['UTC_time']}")
    print(f"  local_time: {row['local_time']}")
    print(f"  timezone: {row['timezone']}")
    print(f"  day_of_week: {row['day_of_week']}")
    print(f"  norm_in_day_time: {row['norm_in_day_time']}")
    
    # 解析local_time
    local_dt = pd.to_datetime(row['local_time'])
    print(f"  解析后的local_datetime: {local_dt}")
    print(f"  提取的日期: {local_dt.date()}")
    print(f"  提取的小时: {local_dt.hour}")
    print(f"  提取的分钟: {local_dt.minute}")
    print(f"  提取的星期几: {local_dt.dayofweek} (0=周一, 6=周日)")
    
    # 验证norm_in_day_time是否与小时对应
    calculated_norm = (local_dt.hour * 60 + local_dt.minute) / (24 * 60)
    print(f"  根据小时分钟计算的norm_in_day_time: {calculated_norm:.4f}")
    print(f"  数据中的norm_in_day_time: {row['norm_in_day_time']:.4f}")
    print(f"  是否一致: {'✓' if abs(calculated_norm - row['norm_in_day_time']) < 0.001 else '✗'}")

# 验证同一天内的POI识别
print("\n\n" + "="*80)
print("验证同一天内POI的识别逻辑")
print("="*80)

# 查看用户962的数据
user_962 = df[df['user_id'] == 962].sort_values('UTC_time').head(10)

print("\n【用户962的前10条记录】")
user_962['local_datetime'] = pd.to_datetime(user_962['local_time'])
user_962['date'] = user_962['local_datetime'].dt.date
user_962['hour'] = user_962['local_datetime'].dt.hour

for idx, row in user_962.iterrows():
    print(f"\nPOI: {row['POI_catname']:20s}")
    print(f"  local_time: {row['local_time']}")
    print(f"  日期: {row['date']}")
    print(f"  小时: {row['hour']}")
    print(f"  轨迹ID: {row['trajectory_id']}")

# 按日期分组，看看每天有多少POI
print("\n【用户962按日期分组】")
date_groups = user_962.groupby('date').size()
for date, count in date_groups.items():
    print(f"  {date}: {count} 个POI")

# 验证跨天识别
print("\n\n" + "="*80)
print("验证跨天POI的识别")
print("="*80)

# 找一个跨天的例子
user_962_full = df[df['user_id'] == 962].sort_values('UTC_time')
user_962_full['local_datetime'] = pd.to_datetime(user_962_full['local_time'])
user_962_full['date'] = user_962_full['local_datetime'].dt.date
user_962_full['hour'] = user_962_full['local_datetime'].dt.hour

print("\n【用户962的所有记录（按时间排序）】")
for i, (idx, row) in enumerate(user_962_full.iterrows()):
    print(f"\n{i+1}. {row['POI_catname']:20s} | {row['local_time']} | 日期: {row['date']} | 小时: {row['hour']}")
    
    # 检查是否跨天
    if i > 0:
        prev_row = user_962_full.iloc[i-1]
        if row['date'] != prev_row['date']:
            print(f"   ⚠️ 跨天了！从 {prev_row['date']} 到 {row['date']}")
            print(f"   这两个POI之间的距离不应该被计算（不在同一天内）")

# 验证我们的分组逻辑
print("\n\n" + "="*80)
print("验证按(user_id, date)分组的逻辑")
print("="*80)

# 模拟我们的分组逻辑
df_sample = df[df['user_id'].isin([962, 445])].copy()
df_sample['local_datetime'] = pd.to_datetime(df_sample['local_time'])
df_sample['date'] = df_sample['local_datetime'].dt.date
df_sample = df_sample.sort_values(['user_id', 'date', 'local_datetime'])

print("\n【按(user_id, date)分组后的结果】")
for (user_id, date), group in df_sample.groupby(['user_id', 'date']):
    print(f"\n用户 {user_id}, 日期 {date}: {len(group)} 个POI")
    if len(group) > 1:
        print(f"  可以计算 {len(group)-1} 段同一天内的出行距离")
        for i in range(len(group)):
            row = group.iloc[i]
            print(f"    {i+1}. {row['local_datetime']} - {row['POI_catname']}")

print("\n" + "="*80)
print("结论：")
print("="*80)
print("""
我们的时间解析逻辑：
1. 使用 pd.to_datetime() 解析 'local_time' 字段
2. 提取 .date() 作为日期（年-月-日）
3. 提取 .hour 作为小时（0-23）
4. 按 (user_id, date) 分组，确保只计算同一天内的POI距离

这个逻辑是正确的，因为：
- local_time 已经是本地时间（考虑了时区）
- date() 方法提取的是日历日期，可以准确识别跨天
- 只在同一个 (user_id, date) 组内计算相邻POI距离

如果您发现有问题，请告诉我具体哪里不对！
""")
