import pandas as pd
import numpy as np
from scipy.stats import kstest, probplot
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 数据加载 ----------------------
df = pd.read_csv('D:\桌面\毕业论文\项目0227\\train_original.csv')
# 特征顺序（17个变量）
feature_order = [
    'clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia',
    'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
    'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
    'RainingDays', 'AverageRainingDays', 'fruitset', 'fruitmass', 'seeds', 'yield'
]
numeric_cols = [col for col in feature_order if col in df.columns]

# ---------------------- 计算各项指标 ----------------------
results = []
for col in numeric_cols:
    data = df[col].dropna()
    skewness = data.skew()                     # 偏度
    kurtosis = data.kurtosis()                  # 峰度
    mu, std = data.mean(), data.std()           # 均值和标准差（用于K-S检验）
    ks_stat, p_value = kstest(data, 'norm', args=(mu, std))  # K-S检验

    # Q-Q图R²（不绘图，仅计算拟合优度）
    (osm, osr), (slope, intercept, r_squared) = probplot(data, dist='norm', plot=None)

    # Q-Q图拟合程度定性
    if r_squared > 0.95:
        qq_fit = "非常好"
    elif r_squared > 0.9:
        qq_fit = "好"
    elif r_squared > 0.80:
        qq_fit = "中等"
    else:
        qq_fit = "差"

    # 综合正态性判断（可自行调整阈值）
    if abs(skewness) < 1 and abs(kurtosis) < 1 and r_squared > 0.9:
        norm_judge = "正态"
    elif abs(skewness) < 2 and abs(kurtosis) < 2 and r_squared > 0.8:
        norm_judge = "近似正态"
    else:
        norm_judge = "非正态"

    results.append({
        '变量': col,
        '样本量': len(data),
        '偏度': round(skewness, 3),
        '峰度': round(kurtosis, 3),
        'KS统计量': round(ks_stat, 4),
        'K-S p值': round(p_value, 4),
        'QQ图R²': round(r_squared, 4),
        'QQ图拟合': qq_fit,
        '综合判断': norm_judge
    })

# ---------------------- 输出表格 ----------------------
result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

# 可选保存为CSV
result_df.to_csv('正态性综合评价.csv', index=False, encoding='utf_8_sig')