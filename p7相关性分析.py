import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import matplotlib
matplotlib.use('TkAgg')

# ---------------------- 1. 基础配置（适配中文+高清） ----------------------
plt.rcParams['font.family'] = ["Times New Roman", 'SimSun']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示
plt.rcParams['figure.dpi'] = 300              # 高清分辨率
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['axes.linewidth'] = 0.8          # 坐标轴宽度

# ---------------------- 2. 读取数据（标准化后的数据） ----------------------
# 替换为你标准化后的数据路径
df = pd.read_csv("D:\桌面\毕业论文\项目0227\\train_cleaned.csv")

# 定义需要分析的特征列表（包含yield，分析与产量的相关性）
feature_cols = [
    'clonesize', 'honeybee', 'bumbles', 'andrena', 'osmia',
    'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
    'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
    'RainingDays', 'AverageRainingDays', 'fruitset', 'fruitmass', 'seeds', 'yield'
]

# 筛选存在的特征（避免列名错误）
feature_cols = [col for col in feature_cols if col in df.columns]
print(f"参与相关性分析的特征：{feature_cols}")

# 筛选无空缺值的数据（避免计算报错）
df_corr = df[feature_cols].dropna()
print(f"有效分析数据量：{len(df_corr)} 行")

# ---------------------- 3. 计算相关系数矩阵（皮尔逊相关） ----------------------
# 计算皮尔逊相关系数
corr_matrix = df_corr.corr(method='pearson')
# 保留4位小数，便于查看
corr_matrix = corr_matrix.round(4)

# （可选）计算相关性显著性p值（标记显著相关的特征）
def corr_pvalue(df):
    """计算相关系数的p值矩阵"""
    p_matrix = np.ones_like(corr_matrix)
    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            if i <= j:  # 仅计算下三角（避免重复）
                corr, p = pearsonr(df[corr_matrix.columns[i]], df[corr_matrix.columns[j]])
                p_matrix[i, j] = p
                p_matrix[j, i] = p
    return p_matrix

p_matrix = corr_pvalue(df_corr)
p_matrix = pd.DataFrame(p_matrix, columns=corr_matrix.columns, index=corr_matrix.index)

# ---------------------- 4. 绘制相关性热力图（论文级美观） ----------------------
# 设置画布大小（适配17个特征，可根据特征数量调整）
fig, ax = plt.subplots(figsize=(12, 10))

# 定义颜色映射（蓝红渐变，符合学术审美）
cmap = sns.diverging_palette(240, 10, as_cmap=True)  # 蓝（负相关）→红（正相关）

# 绘制热力图
heatmap = sns.heatmap(
    corr_matrix,
    ax=ax,
    cmap=cmap,          # 颜色映射
    annot=True,         # 显示相关系数数值
    fmt='.4f',          # 数值格式（4位小数）
    vmin=-1, vmax=1,    # 颜色范围（-1到1）
    center=0,           # 中间值（0，白色）
    square=True,        # 单元格正方形
    linewidths=0.5,     # 单元格边框宽度
    cbar_kws={
        'shrink': 0.8,  # 颜色条缩放
        'label': 'Pearson Correlation Coefficient',  # 颜色条标签
        'orientation': 'vertical'  # 颜色条垂直
    },
    # 标记显著性（p<0.05加*，p<0.01加**）
    annot_kws={
        'size': 7,      # 数值字号
        'weight': 'normal'
    }
)

# （可选）添加显著性标记（在相关系数旁加*）
# 遍历每个单元格，添加显著性标记
for i in range(len(corr_matrix.columns)):
    for j in range(len(corr_matrix.columns)):
        text = ax.text(
            j+0.25, i,  # 标记位置（相关系数右侧）
            '*' if p_matrix.iloc[i, j] < 0.05 else ('**' if p_matrix.iloc[i, j] < 0.01 else ''),
            ha='left', va='center', fontsize=6, color='black', fontweight='bold'
        )

# ---------------------- 5. 美化图表（适配论文要求） ----------------------
# 设置标题
# ax.set_title(
#     '蓝莓数据集特征相关性热力图（标准化后）',
#     fontsize=14, fontweight='bold', pad=20
# )

# 设置坐标轴标签
ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, rotation=45, ha='right')
ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)

# 调整颜色条标签字号
cbar = ax.collections[0].colorbar
cbar.ax.set_ylabel('Pearson Correlation Coefficient', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=8)

# 紧凑布局（避免文字重叠）
plt.tight_layout()

# ---------------------- 6. 保存热力图 ----------------------
save_path = r'D:\桌面\毕业论文\项目0227\correlation_heatmap.png'
plt.savefig(
    save_path,
    facecolor='white',   # 背景白色
    bbox_inches='tight', # 紧凑保存（裁剪空白）
    pad_inches=0.1       # 边距
)



# # ---------------------- 7. 输出关键相关性结果（便于论文总结） ----------------------
# # 提取与yield相关性最高的前10个特征
# yield_corr = corr_matrix['yield'].sort_values(ascending=False)
# print("\n与产量（yield）相关性排序（从高到低）：")
# print(yield_corr)
#
# # 输出显著相关的特征（p<0.05）
# significant_corr = yield_corr[yield_corr.index != 'yield'][p_matrix['yield'][yield_corr.index != 'yield'] < 0.05]
# print(f"\n与产量显著相关的特征（p<0.05）：")
# print(significant_corr)