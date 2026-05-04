from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

from evaluate_regression_model import evaluate_regression_model
from plot_true_vs_pred import plot_true_vs_pred_advanced


data_path = Path(__file__).resolve().parent
X_train_pca_full = pd.read_csv(data_path / "data" / "X_train_pca.csv")
X_train_full = pd.read_csv(data_path / "data" / "train_cleaned.csv")
X_train_full = X_train_full.drop(columns=["yield"], errors="ignore")
y_train_full = pd.read_csv(data_path / "data" / "y_train.csv").iloc[:, 0]

X_test = pd.read_csv(data_path/ "data" / "test_original.csv")
X_test = X_test.drop(columns=["yield"], errors="ignore")
y_test = pd.read_csv(data_path/ "data" / "y_test.csv").iloc[:, 0]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

train_idx, val_idx = train_test_split(range(len(y_train_full)), test_size=0.2, random_state=42, shuffle=True)

X_train_raw = X_train_full.iloc[list(train_idx)].reset_index(drop=True)
X_val_raw = X_train_full.iloc[list(val_idx)].reset_index(drop=True)
X_train_pca = X_train_pca_full.iloc[list(train_idx)].reset_index(drop=True)
X_val_pca = X_train_pca_full.iloc[list(val_idx)].reset_index(drop=True)
y_train = y_train_full.iloc[list(train_idx)].reset_index(drop=True)
y_val = y_train_full.iloc[list(val_idx)].reset_index(drop=True)


def normalize_result_keys(result, model_name):
    return {
        "模型": result.get("模型", result.get("妯″瀷", model_name)),
        "R2": result["R2"],
        "MAE": result["MAE"],
        "RMSE": result["RMSE"],
        "训练与预测耗时(s)": result.get("训练与预测耗时(s)", result.get("璁粌涓庨娴嬭€楁椂(s)"))
    }


def build_stacking_raw():
    return StackingRegressor(
        estimators=[
            ("xgb", XGBRegressor(subsample=0.6, reg_lambda=5, reg_alpha=1, n_estimators=200, min_child_weight=3, max_depth=5, learning_rate=0.03, gamma=0, colsample_bytree=1.0, objective="reg:squarederror", random_state=42, n_jobs=-1)),
            ("lgb", LGBMRegressor(subsample=0.8, reg_lambda=1, reg_alpha=0, num_leaves=50, n_estimators=300, min_child_samples=20, max_depth=4, learning_rate=0.03, colsample_bytree=0.9, objective="regression", random_state=42, n_jobs=-1, verbose=-1)),
            ("rf", RandomForestRegressor(n_estimators=500, min_samples_split=2, min_samples_leaf=6, max_features=0.6, max_depth=15, bootstrap=True, random_state=42, n_jobs=-1))
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )


def build_stacking_pca():
    return StackingRegressor(
        estimators=[
            ("xgb", XGBRegressor(subsample=0.6, reg_lambda=5, reg_alpha=0.1, n_estimators=200, min_child_weight=1, max_depth=5, learning_rate=0.05, gamma=0.1, colsample_bytree=0.6, objective="reg:squarederror", random_state=42, n_jobs=-1)),
            ("lgb", LGBMRegressor(subsample=0.9, reg_lambda=10, reg_alpha=1, num_leaves=50, n_estimators=300, min_child_samples=30, max_depth=5, learning_rate=0.1, colsample_bytree=0.7, objective="regression", random_state=42, n_jobs=-1, verbose=-1)),
            ("rf", RandomForestRegressor(n_estimators=500, min_samples_split=2, min_samples_leaf=6, max_features=0.6, max_depth=15, bootstrap=True, random_state=42, n_jobs=-1))
        ],
        final_estimator=Ridge(alpha=1.0),
        cv=5,
        n_jobs=-1
    )


def run_stacking_validation(builder, X_train_data, X_val_data, title_prefix, pred_plot_name):
    model = builder()
    result, val_pred = evaluate_regression_model(model, X_train_data, X_val_data, y_train, y_val, model_name=title_prefix)
    result = normalize_result_keys(result, title_prefix)
    final_time = result["训练与预测耗时(s)"]
    plot_path = data_path / pred_plot_name
    plot_true_vs_pred_advanced(y_val, val_pred, title=f"{title_prefix}: Validation True vs Predicted", save_path=plot_path)

    return {
        "模型": result["模型"],
        "交叉验证最佳RMSE": "不适用",
        "验证集RMSE": result["RMSE"],
        "验证集MAE": result["MAE"],
        "验证集R2": result["R2"],
        "调参耗时(s)": 0.0,
        "最终训练与验证耗时(s)": final_time,
        "总耗时(s)": final_time,
        "验证集对比图": str(plot_path)
    }


def run_stacking_test(builder, X_train_val_data, X_test_data, title_prefix, pred_plot_name):
    final_model = builder()
    result, test_pred = evaluate_regression_model(
        final_model,
        X_train_val_data,
        X_test_data,
        pd.concat([y_train, y_val], ignore_index=True),
        y_test,
        model_name=title_prefix
    )
    result = normalize_result_keys(result, title_prefix)
    plot_path = data_path / pred_plot_name
    plot_true_vs_pred_advanced(y_test, test_pred, title=f"{title_prefix}: Test True vs Predicted", save_path=plot_path)
    result["测试集对比图"] = str(plot_path)
    return result


result_stacking_raw = run_stacking_validation(build_stacking_raw, X_train_raw, X_val_raw, "Stacking（PCA前）", "stacking_before_pca_val_pred.png")
result_stacking_pca = run_stacking_validation(build_stacking_pca, X_train_pca, X_val_pca, "Stacking（PCA后）", "stacking_after_pca_val_pred.png")

compare_stacking_df = pd.DataFrame([result_stacking_raw, result_stacking_pca])
print("\n==============================")
print("Stacking 验证集结果汇总")
print("==============================")
print(compare_stacking_df[["模型", "交叉验证最佳RMSE", "验证集RMSE", "验证集MAE", "验证集R2", "调参耗时(s)", "最终训练与验证耗时(s)", "总耗时(s)"]])
print("\n【PCA前详细结果】")
print(result_stacking_raw)
print("\n【PCA后详细结果】")
print(result_stacking_pca)

# 仅对 PCA前 进行最终测试集评估
X_train_val_raw = pd.concat([X_train_raw, X_val_raw], ignore_index=True)
final_test_result = run_stacking_test(
    build_stacking_raw,
    X_train_val_raw,
    X_test,
    "最终模型：Stacking（PCA前）",
    "stacking_before_pca_test_pred.png"
)

print("\n==============================")
print("PCA前 Stacking 最终测试集结果")
print("==============================")
print(final_test_result)
