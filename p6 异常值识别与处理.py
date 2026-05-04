import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

train_df = pd.read_csv('../data/train_original.csv')
test_df = pd.read_csv('../data/test_original.csv')

normal_approx = [
    'clonesize', 'andrena', 'osmia',
    'MaxOfUpperTRange', 'MinOfUpperTRange', 'AverageOfUpperTRange',
    'MaxOfLowerTRange', 'MinOfLowerTRange', 'AverageOfLowerTRange',
    'RainingDays', 'AverageRainingDays',
    'fruitset', 'fruitmass', 'seeds', 'yield'
]

non_normal = ['honeybee', 'bumbles']


def process_outliers(train_df, test_df, normal_cols, non_normal_cols):
    train_outlier_mask = pd.DataFrame(False, index=train_df.index, columns=train_df.columns)
    train_stats = {}
    for col in normal_cols:
        if col in train_df.columns:
            data = train_df[col].dropna()
            if len(data) > 0:
                mean = data.mean()
                std = data.std()
                train_stats[col] = {"type": "normal", "mean": mean, "std": std}
                lower = mean - 3 * std
                upper = mean + 3 * std
                train_outlier_mask.loc[data.index, col] = (data < lower) | (data > upper)
    for col in non_normal_cols:
        if col in train_df.columns:
            data = train_df[col].dropna()
            if len(data) > 0:
                Q1 = data.quantile(0.25)
                Q3 = data.quantile(0.75)
                IQR = Q3 - Q1
                train_stats[col] = {"type": "non_normal", "Q1": Q1, "Q3": Q3, "IQR": IQR}
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                train_outlier_mask.loc[data.index, col] = (data < lower) | (data > upper)

    train_rows_to_drop = train_outlier_mask.any(axis=1)
    train_clean = train_df[~train_rows_to_drop].copy()
    train_drop_num = train_rows_to_drop.sum()
    train_drop_ratio = train_drop_num / len(train_df) if len(train_df) > 0 else 0

    # ========== 第二步：处理测试集（用训练集统计量标记异常，不删除） ==========
    test_outlier_mask = pd.DataFrame(False, index=test_df.index, columns=test_df.columns)
    for col in test_df.columns:
        if col not in train_stats:
            continue
        data = test_df[col].dropna()
        if len(data) == 0:
            continue

        # 用训练集的统计量判断测试集异常值
        if train_stats[col]["type"] == "normal":
            mean = train_stats[col]["mean"]
            std = train_stats[col]["std"]
            lower = mean - 3 * std
            upper = mean + 3 * std
            test_outlier_mask.loc[data.index, col] = (data < lower) | (data > upper)
        else:  # non_normal
            Q1 = train_stats[col]["Q1"]
            Q3 = train_stats[col]["Q3"]
            IQR = train_stats[col]["IQR"]
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            test_outlier_mask.loc[data.index, col] = (data < lower) | (data > upper)

    # 测试集不删除异常行，仅统计数量
    test_outlier_num = test_outlier_mask.any(axis=1).sum()
    test_outlier_ratio = test_outlier_num / len(test_df) if len(test_df) > 0 else 0

    # ========== 输出结果 ==========
    print("\n==================== 训练集异常值处理 ====================")
    print(f"原始行数: {len(train_df)}, 清洗后行数: {len(train_clean)}, 删除 {train_drop_num} 行（占比 {train_drop_ratio:.2%}）")
    print("各变量异常值数量（训练集）:")
    for col in train_df.columns:
        if col in normal_cols or col in non_normal_cols:
            n_outliers = train_outlier_mask[col].sum()
            if n_outliers > 0:
                non_missing = train_df[col].count()
                ratio = n_outliers / non_missing if non_missing > 0 else 0
                print(f"  {col}: {n_outliers} 个（占非缺失值 {ratio:.2%}）")

    print("\n==================== 测试集异常值统计 ====================")
    print(f"测试集总行数: {len(test_df)}, 异常值行数量: {test_outlier_num}（占比 {test_outlier_ratio:.2%}）")
    print("（测试集仅标记异常值，不删除，保证评估真实性）")
    print("各变量异常值数量（测试集，基于训练集统计量）:")
    for col in test_df.columns:
        if col in normal_cols or col in non_normal_cols:
            n_outliers = test_outlier_mask[col].sum()
            if n_outliers > 0:
                non_missing = test_df[col].count()
                ratio = n_outliers / non_missing if non_missing > 0 else 0
                print(f"  {col}: {n_outliers} 个（占非缺失值 {ratio:.2%}）")

    return train_clean, test_df, train_stats  # 返回清洗后的训练集、原始测试集、训练集统计量


train_clean, test_df, train_stats = process_outliers(
    train_df, test_df, normal_approx, non_normal
)

# train_clean.to_csv('train_cleaned.csv', index=False, encoding='utf_8_sig')
#
# stats_df = pd.DataFrame.from_dict(train_stats, orient='index')
# stats_df.to_csv('train_outlier_stats.csv', encoding='utf_8_sig')


print(f"清洗后训练集: {train_clean.shape}（用于模型训练）")
print(f"原始测试集: {test_df.shape}（用于最终评估）")