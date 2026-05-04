import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('../data/data.csv')
df = df.drop(["id"], axis=1, errors='ignore')
print(df.shape)
basic_stats = df.describe().T

basic_stats['偏度'] = round(df.skew(numeric_only=True), 3)
basic_stats['峰度'] = round(df.kurt(numeric_only=True), 3)

basic_stats.rename(columns={
    'count': '样本数', 'mean': '均值', 'std': '标准差',
    'min': '最小值', '25%': '下四分位数(Q1)',
    '50%': '中位数(Q2)', '75%': '上四分位数(Q3)', 'max': '最大值'
}, inplace=True)


print("=== 蓝莓数据集描述性统计结果 ===")
print(round(basic_stats,2))
# basic_stats.to_excel("蓝莓数据描述性统计.xlsx", sheet_name="统计结果")