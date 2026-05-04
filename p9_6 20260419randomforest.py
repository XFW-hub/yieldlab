import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from joblib import parallel_backend
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split

from evaluate_regression_model import evaluate_regression_model
from plot_true_vs_pred import plot_true_vs_pred_advanced


data_path = Path(__file__).resolve().parent
X_train_pca_full = pd.read_csv(data_path / "data" / "X_train_pca.csv")
X_train_full = pd.read_csv(data_path / "data" / "train_cleaned.csv")
X_train_full = X_train_full.drop(columns=["yield"], errors="ignore")
y_train_full = pd.read_csv(data_path / "data" / "y_train.csv").iloc[:, 0]

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

cv = KFold(n_splits=5, shuffle=True, random_state=42)
train_idx, val_idx = train_test_split(range(len(y_train_full)), test_size=0.2, random_state=42, shuffle=True)

X_train_raw = X_train_full.iloc[list(train_idx)].reset_index(drop=True)
X_val_raw = X_train_full.iloc[list(val_idx)].reset_index(drop=True)
X_train_pca = X_train_pca_full.iloc[list(train_idx)].reset_index(drop=True)
X_val_pca = X_train_pca_full.iloc[list(val_idx)].reset_index(drop=True)
y_train = y_train_full.iloc[list(train_idx)].reset_index(drop=True)
y_val = y_train_full.iloc[list(val_idx)].reset_index(drop=True)

param_dist_rf = {
    "n_estimators": [100, 200, 300, 500],
    "max_depth": [None, 5, 10, 15, 20],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4, 6],
    "max_features": ["sqrt", "log2", 0.6, 0.8, 1.0],
    "bootstrap": [True]
}


def normalize_result_keys(result, model_name):
    return {
        "模型": result.get("模型", result.get("妯″瀷", model_name)),
        "R2": result["R2"],
        "MAE": result["MAE"],
        "RMSE": result["RMSE"],
        "训练与预测耗时(s)": result.get("训练与预测耗时(s)", result.get("璁粌涓庨娴嬭€楁椂(s)"))
    }


def build_rf_model(**params):
    return RandomForestRegressor(random_state=42, n_jobs=-1, **params)


def tune_rf_model(X_train_data, y_train_data, n_iter=15):
    random_search = RandomizedSearchCV(
        estimator=build_rf_model(),
        param_distributions=param_dist_rf,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        verbose=0,
        random_state=42,
        n_jobs=-1
    )
    start_time = time.time()
    with parallel_backend("threading"):
        random_search.fit(X_train_data, y_train_data)
    tune_time = time.time() - start_time
    return random_search.best_params_, -random_search.best_score_, tune_time


def run_rf_experiment(X_tr, X_val, title_prefix, plot_name):
    best_params, best_cv_rmse, tune_time = tune_rf_model(X_tr, y_train)
    best_model = build_rf_model(**best_params)
    result, y_pred = evaluate_regression_model(best_model, X_tr, X_val, y_train, y_val, model_name=title_prefix)
    result = normalize_result_keys(result, title_prefix)
    final_time = result["训练与预测耗时(s)"]
    plot_path = data_path / plot_name
    plot_true_vs_pred_advanced(y_val, y_pred, title=f"{title_prefix} 验证集真实值与预测值", save_path=plot_path)
    return {
        "模型": result["模型"],
        "交叉验证最佳RMSE": best_cv_rmse,
        "验证集RMSE": result["RMSE"],
        "验证集MAE": result["MAE"],
        "验证集R2": result["R2"],
        "调参耗时(s)": tune_time,
        "最终训练与验证耗时(s)": final_time,
        "总耗时(s)": tune_time + final_time,
        "best_params": best_params,
        "验证集对比图": str(plot_path)
    }


result_rf_raw = run_rf_experiment(X_train_raw, X_val_raw, "RandomForest（PCA前）", "randomforest_before_pca_val_pred.png")
result_rf_pca = run_rf_experiment(X_train_pca, X_val_pca, "RandomForest（PCA后）", "randomforest_after_pca_val_pred.png")
compare_rf_df = pd.DataFrame([result_rf_raw, result_rf_pca])
print("\n==============================")
print("RandomForest 验证集结果汇总")
print("==============================")
print(compare_rf_df[["模型", "交叉验证最佳RMSE", "验证集RMSE", "验证集MAE", "验证集R2", "调参耗时(s)", "最终训练与验证耗时(s)", "总耗时(s)"]])
print("\n【PCA前详细结果】")
print(result_rf_raw)
print("\n【PCA后详细结果】")
print(result_rf_pca)
