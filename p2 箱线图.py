import pandas as pd
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import matplotlib
matplotlib.use('TkAgg')
import numpy as np

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 300

df = pd.read_csv('..\data\data.csv')
df = df.drop(["id"], axis=1, errors='ignore')

feature_order = [
    'clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia',
    'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
    'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
    'RainingDays', 'AverageRainingDays', 'fruitset', 'fruitmass', 'seeds', 'yield'
]
numeric_cols = [col for col in feature_order if col in df.columns]


n_cols = 3
n_rows = int(np.ceil(len(numeric_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3.6 * n_rows))
axes = axes.flatten()


for idx, col in enumerate(numeric_cols):
    ax = axes[idx]
    bp = ax.boxplot(df[col].dropna(), patch_artist=True,
                     boxprops=dict(facecolor='lightblue', color='blue'),
                     whiskerprops=dict(color='blue'),
                     capprops=dict(color='blue'),
                     medianprops=dict(color='red', linewidth=2),
                     flierprops=dict(marker='o', markerfacecolor='gray', markersize=3, alpha=0.5))
    ax.set_title(col, fontsize=14)
    ax.set_ylabel('数值', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    ax.set_xticklabels([])


for idx in range(len(numeric_cols), len(axes)):
    axes[idx].set_visible(False)


plt.tight_layout()
plt.savefig('箱线图_多子图.png', bbox_inches='tight', facecolor='white')

plt.show()