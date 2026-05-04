import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA  # 无监督降维（PCA）
from sklearn.feature_selection import SelectKBest, f_regression  # 有监督特征选择（降维）
from sklearn.ensemble import RandomForestRegressor  # 可选：基于模型的特征重要性降维
import matplotlib
matplotlib.use('TkAgg')
plt.rcParams['font.family'] = ["Times New Roman", 'SimSun']
plt.rcParams['font.sans-serif'] = ['SimHei']  # 中文显示
plt.rcParams['axes.unicode_minus'] = False    # 负号显示
plt.rcParams['figure.dpi'] = 300              # 高清分辨率

# ---------------------- 1. 加载并划分数据 ----------------------
train_df = pd.read_csv(r'D:\桌面\毕业论文\项目0227\\train_cleaned.csv')
test_df = pd.read_csv(r'D:\桌面\毕业论文\项目0227\\test_original.csv')

# 分离特征和标签（yield为目标列）
target_col = 'yield'
X_train = train_df.drop(target_col, axis=1, errors='ignore')
y_train = train_df[target_col]
X_test = test_df.drop(target_col, axis=1, errors='ignore')
y_test = test_df[target_col]


# 标准化（仅用训练集拟合scaler）
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # 训练集：fit+transform
X_test_scaled = scaler.transform(X_test)  # 测试集：仅transform（复用训练集均值/方差）

# ---------------------- 3. 降维（两种方法可选，二选一或对比） ----------------------
# ===== 方法1：无监督降维（PCA，适合高维数据、去除多重共线性） =====
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

print("\n===== 方法1：PCA降维 =====")

# 固定降到4维
pca = PCA(n_components=4, random_state=42)

# 在训练集上拟合PCA
X_train_pca = pca.fit_transform(X_train_scaled)

# 测试集仅做变换
X_test_pca = pca.transform(X_test_scaled)

print(f"PCA降维后 - 训练集: {X_train_pca.shape}, 测试集: {X_test_pca.shape}")
print(f"保留的主成分数: {pca.n_components_}")
print(f"各主成分解释方差比: {pca.explained_variance_ratio_}")
print(f"累计解释方差比: {np.sum(pca.explained_variance_ratio_):.2%}")

# 原始特征名
feature_names = X_train.columns.tolist()   # 如果X是DataFrame

# 主成分系数矩阵
loading_df = pd.DataFrame(
    pca.components_,
    columns=feature_names,
    index=[f'PC{i+1}' for i in range(pca.n_components_)]
)

print("\n===== 主成分系数矩阵 =====")
print(loading_df)

# 输出每个主成分的重要变量（按绝对值排序）
for i in range(pca.n_components_):
    print(f"\n===== 第{i+1}主成分的重要变量 =====")
    comp_series = pd.Series(pca.components_[i], index=feature_names)
    comp_series = comp_series.reindex(comp_series.abs().sort_values(ascending=False).index)
    print(comp_series)

# 绘制累计解释方差曲线
cum_var = np.cumsum(pca.explained_variance_ratio_)
best_k = pca.n_components_

plt.figure(figsize=(8, 5.5))
plt.plot(
    range(1, len(cum_var) + 1),
    cum_var,
    marker='o',
    linewidth=2,
    markersize=5,
    label='累计解释方差比'
)

plt.axvline(x=best_k, linestyle='--', linewidth=1.5, label=f'保留主成分数 = {best_k}')
plt.scatter(best_k, cum_var[best_k - 1], s=60, zorder=3)
plt.text(
    best_k + 0.1,
    cum_var[best_k - 1] - 0.03,
    f'({best_k}, {cum_var[best_k - 1]:.2%})',
    fontsize=10
)

plt.xlabel('主成分数', fontsize=12)
plt.ylabel('累计解释方差比', fontsize=12)
plt.title('PCA累计解释方差曲线', fontsize=13)
plt.xticks(range(1, len(cum_var) + 1))
plt.ylim(0, 1.05)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(frameon=True, fontsize=10)
plt.tight_layout()
plt.savefig("pca_variance_4components.png", dpi=300, bbox_inches='tight')
plt.show()
#
# # ===== 方法2：有监督特征选择（SelectKBest，适合保留可解释的特征） =====
# print("\n===== 方法2：SelectKBest特征选择（有监督降维） =====")
# # 步骤1：在训练集上拟合SelectKBest（选择Top10特征）
# selector = SelectKBest(score_func=f_regression, k=10)
# X_train_selected = selector.fit_transform(X_train_scaled, y_train)  # 训练集：fit+transform（需传入标签）
# # 步骤2：测试集仅用训练集的选择结果转换
# X_test_selected = selector.transform(X_test_scaled)  # 测试集：仅transform
#
# # 查看选中的特征名（可解释性强，毕设加分）
# selected_feature_idx = selector.get_support(indices=True)
# selected_feature_names = X_train.columns[selected_feature_idx].tolist()
# print(f"SelectKBest选中的特征: {selected_feature_names}")
# print(f"特征选择后 - 训练集: {X_train_selected.shape}, 测试集: {X_test_selected.shape}")
#
# # ===== 可选：基于模型的特征重要性降维（随机森林，兼顾效果和可解释） =====
# print("\n===== 方法3：随机森林特征重要性降维 =====")
# rf = RandomForestRegressor(n_estimators=100, random_state=42)
# rf.fit(X_train_scaled, y_train)
# # 提取特征重要性Top10
# importance = pd.Series(rf.feature_importances_, index=X_train.columns)
# top_features = importance.sort_values(ascending=False).head(10).index.tolist()
# # 训练集/测试集仅保留Top10特征（无需拟合，直接选择）
# X_train_rf = X_train_scaled[:, [X_train.columns.get_loc(col) for col in top_features]]
# X_test_rf = X_test_scaled[:, [X_train.columns.get_loc(col) for col in top_features]]
# print(f"随机森林选中的Top10特征: {top_features}")
# print(f"模型特征选择后 - 训练集: {X_train_rf.shape}, 测试集: {X_test_rf.shape}")
#
# # ---------------------- 4. 降维后的数据保存（用于后续建模） ----------------------
# # 保存PCA降维结果
# # pd.DataFrame(X_train_pca).to_csv("X_train_pca.csv", index=False)
# # pd.DataFrame(X_test_pca).to_csv("X_test_pca.csv", index=False)
# # 保存SelectKBest降维结果
# # pd.DataFrame(X_train_selected, columns=selected_feature_names).to_csv("X_train_selected.csv", index=False)
# # pd.DataFrame(X_test_selected, columns=selected_feature_names).to_csv("X_test_selected.csv", index=False)
# # 保存标签
# # y_train.to_csv("y_train.csv", index=False)
# # y_test.to_csv("y_test.csv", index=False)
