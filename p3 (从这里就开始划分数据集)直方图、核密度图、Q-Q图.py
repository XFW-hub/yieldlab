import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import warnings
import matplotlib
from sklearn.model_selection import train_test_split

matplotlib.use('TkAgg')
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300
warnings.filterwarnings('ignore')

# df = pd.read_csv('./data/data.csv')
# df = df.drop(["id"], axis=1, errors='ignore')
#
# train_df, test_df = train_test_split(
#     df,
#     test_size=0.3,
#     random_state=42,
#     shuffle=True
# )
# train_df.to_csv("../data/train_original.csv",index=False, encoding='utf_8_sig')
# test_df.to_csv('..data/test_original.csv', index=False, encoding='utf_8_sig')
# train_df.name = "训练集"
# test_df.name = "测试集"
# print(f"划分后 - 训练集: {train_df.shape}, 测试集: {test_df.shape}")



df = pd.read_csv('./data/train_original.csv')


feature_order = [
    'clonesize', 'honeybee',
    # 'bumbles',
    # 'andrena', 'osmia',
    # 'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
    # 'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
    'RainingDays',
    # 'AverageRainingDays',
    'fruitset', 'fruitmass', 'seeds', 'yield'
]

numeric_cols = [col for col in feature_order if col in df.columns]

# ---------------------- 2. 创建整体大图（17行 × 2列） ----------------------
n_features = len(numeric_cols)
fig, axes = plt.subplots(n_features, 2, figsize=(12, 3.5 * n_features))  # 每行高度3.5英寸

# 如果只有一个特征，axes需要调整为2维
if n_features == 1:
    axes = axes.reshape(1, -1)

for idx, col in enumerate(numeric_cols):
    ax1 = axes[idx, 0]   # 左列：直方图+核密度+正态拟合
    ax2 = axes[idx, 1]   # 右列：Q-Q图

    # --- 左图：直方图 + 核密度曲线 + 正态拟合曲线 ---
    # 绘制直方图（密度归一化）
    ax1.hist(df[col].dropna(), bins=30, density=True, alpha=0.6, color='#4CAF50', edgecolor='black', linewidth=0.5)
    # 核密度估计曲线
    df[col].plot(kind='kde', ax=ax1, color='red', linewidth=1.5, label='核密度曲线')
    # 正态拟合曲线
    mu, sigma = df[col].mean(), df[col].std()
    x = np.linspace(df[col].min(), df[col].max(), 200)
    ax1.plot(x, stats.norm.pdf(x, mu, sigma), 'g--', linewidth=1.5, label='正态拟合曲线')
    ax1.set_title(f'{col} 分布直方图', fontsize=11)
    ax1.set_xlabel(col, fontsize=9)
    ax1.set_ylabel('密度', fontsize=9)
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3, linestyle='--')

    # --- 右图：Q-Q图 ---
    stats.probplot(df[col].dropna(), plot=ax2, rvalue=True)
    ax2.set_title(f'{col} Q-Q图', fontsize=11)
    ax2.set_xlabel('理论分位数', fontsize=9)
    ax2.set_ylabel('样本分位数', fontsize=9)
    ax2.grid(alpha=0.3, linestyle='--')



# 调整子图间距
plt.subplots_adjust(hspace=0.5, wspace=0.3)
plt.savefig("正态性检验可视化结果.png", bbox_inches='tight', facecolor='white', dpi=300)
plt.show()