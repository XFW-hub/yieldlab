import pandas as pd
from scipy.stats import kstest
import warnings
warnings.filterwarnings('ignore')

# ---------------------- 数据加载 ----------------------
df = pd.read_csv('D:\桌面\毕业论文\项目0227\\train_original.csv')

# 特征顺序
feature_order = [
    'clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia',
    'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
    'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
    'RainingDays', 'AverageRainingDays', 'fruitset', 'fruitmass', 'seeds', 'yield'
]
numeric_cols = [col for col in feature_order if col in df.columns]

# ---------------------- K-S检验 ----------------------
results = []
for col in numeric_cols:
    data = df[col].dropna()
    # 使用样本均值和标准差作为正态分布的参数
    mu, std = data.mean(), data.std()
    ks_stat, p_value = kstest(data, 'norm', args=(mu, std))
    results.append({
        '变量': col,
        '样本量': len(data),
        'KS统计量': ks_stat,
        'p值': p_value,
        '是否正态 (p≥0.05)': '是' if p_value >= 0.05 else '否'
    })

# 输出结果表格
result_df = pd.DataFrame(results)
print(result_df.to_string(index=False))

# 可选：保存结果到CSV
result_df.to_csv('KS_test_results.csv', index=False, encoding='utf_8_sig')