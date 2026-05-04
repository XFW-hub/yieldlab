import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import parallel_backend
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from plot_true_vs_pred import plot_true_vs_pred_advanced


data_path = Path(__file__).resolve().parent

warnings.filterwarnings("ignore", category=ConvergenceWarning)
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS"
]
plt.rcParams["axes.unicode_minus"] = False

X_train_pca_full = pd.read_csv(data_path / "data" / "X_train_pca.csv")
X_train_full = pd.read_csv(data_path / "data" / "train_cleaned.csv")
X_train_full = X_train_full.drop(columns=["yield"], errors="ignore")
y_train_full = pd.read_csv(data_path / "data" / "y_train.csv").iloc[:, 0]

train_idx, val_idx = train_test_split(
    np.arange(len(y_train_full)),
    test_size=0.2,
    random_state=42,
    shuffle=True
)

X_train_raw = X_train_full.iloc[train_idx].reset_index(drop=True)
X_val_raw = X_train_full.iloc[val_idx].reset_index(drop=True)
y_train = y_train_full.iloc[train_idx].reset_index(drop=True)
y_val = y_train_full.iloc[val_idx].reset_index(drop=True)
X_train_pca = X_train_pca_full.iloc[train_idx].reset_index(drop=True)
X_val_pca = X_train_pca_full.iloc[val_idx].reset_index(drop=True)

scaler_mlp = StandardScaler()
X_train_raw_scaled = scaler_mlp.fit_transform(X_train_raw)
X_val_raw_scaled = scaler_mlp.transform(X_val_raw)

cv = KFold(n_splits=5, shuffle=True, random_state=42)
param_dist_mlp = {
    "hidden_layer_sizes": [(32,), (64,), (128,), (64, 32), (128, 64)],
    "activation": ["relu", "tanh"],
    "solver": ["adam"],
    "alpha": [0.0001, 0.001, 0.01, 0.1],
    "learning_rate_init": [0.001, 0.005, 0.01],
    "batch_size": [32, 64, 128]
}
N_SEARCH_ITER = 15
MAX_EPOCHS_FOR_CURVES = 100


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def search_best_params(X_train_data, y_train_data):
    base_model = MLPRegressor(random_state=42, max_iter=1000)
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist_mlp,
        n_iter=N_SEARCH_ITER,
        scoring="neg_root_mean_squared_error",
        cv=cv,
        random_state=42,
        n_jobs=-1,
        verbose=0,
        return_train_score=False
    )
    start_time = time.time()
    with parallel_backend("threading"):
        random_search.fit(X_train_data, y_train_data)
    tune_time = time.time() - start_time
    return random_search.best_params_, -random_search.best_score_, tune_time


def train_mlp_with_val_selection(X_tr, y_tr, X_val, y_val, best_params, max_epochs):
    model = MLPRegressor(
        random_state=42,
        max_iter=1,
        warm_start=True,
        shuffle=True,
        **best_params
    )

    train_curve = []
    val_curve = []
    best_state = None
    best_val_rmse = float("inf")
    best_epoch = 0

    start_time = time.time()
    for epoch in range(1, max_epochs + 1):
        model.fit(X_tr, y_tr)
        train_pred = model.predict(X_tr)
        val_pred = model.predict(X_val)
        train_rmse = rmse(y_tr, train_pred)
        val_rmse = rmse(y_val, val_pred)
        train_curve.append(train_rmse)
        val_curve.append(val_rmse)
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            best_epoch = epoch
            best_state = {
                "val_pred": val_pred.copy(),
                "val_rmse": val_rmse,
                "val_mae": mean_absolute_error(y_val, val_pred),
                "val_r2": r2_score(y_val, val_pred)
            }
    elapsed = time.time() - start_time

    return {
        "train_curve": np.array(train_curve),
        "val_curve": np.array(val_curve),
        "best_epoch": best_epoch,
        "val_rmse": best_state["val_rmse"],
        "val_mae": best_state["val_mae"],
        "val_r2": best_state["val_r2"],
        "val_pred": best_state["val_pred"],
        "elapsed": elapsed
    }


def plot_curve(curve, ylabel, title, output_path, best_epoch=None):
    epochs = np.arange(1, len(curve) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, curve, linewidth=2.2)
    if best_epoch is not None:
        plt.axvline(best_epoch, color="red", linestyle="--", linewidth=1.4, label=f"最佳Epoch={best_epoch}")
        plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()


def run_mlp_experiment(X_train_data, X_val_data, y_train_data, y_val_data, title_prefix, output_prefix):
    best_params, best_cv_rmse, tune_time = search_best_params(X_train_data, y_train_data)
    training_result = train_mlp_with_val_selection(
        X_tr=X_train_data,
        y_tr=y_train_data,
        X_val=X_val_data,
        y_val=y_val_data,
        best_params=best_params,
        max_epochs=MAX_EPOCHS_FOR_CURVES
    )

    train_plot_path = data_path / f"{output_prefix}_Train_Curve.png"
    val_plot_path = data_path / f"{output_prefix}_Validation_Curve.png"
    pred_plot_path = data_path / f"{output_prefix}_Validation_Pred.png"

    plot_curve(training_result["train_curve"], "训练集 RMSE", f"{title_prefix} 训练集曲线", train_plot_path, training_result["best_epoch"])
    plot_curve(training_result["val_curve"], "验证集 RMSE", f"{title_prefix} 验证集曲线", val_plot_path, training_result["best_epoch"])

    plot_true_vs_pred_advanced(
        y_val,
        training_result["val_pred"],
        title=f"{title_prefix} 验证集真实值与预测值",
        save_path=pred_plot_path
    )

    final_time = training_result["elapsed"]
    total_time = tune_time + final_time
    return {
        "模型": title_prefix,
        "交叉验证最佳RMSE": best_cv_rmse,
        "验证集RMSE": training_result["val_rmse"],
        "验证集MAE": training_result["val_mae"],
        "验证集R2": training_result["val_r2"],
        "调参耗时(s)": tune_time,
        "最终训练与验证耗时(s)": final_time,
        "总耗时(s)": total_time,
        "最佳Epoch": training_result["best_epoch"],
        "best_params": best_params,
        "训练曲线图": str(train_plot_path),
        "验证曲线图": str(val_plot_path),
        "验证集对比图": str(pred_plot_path)
    }


def print_final_summary(result_raw, result_pca):
    compare_df = pd.DataFrame([
        {key: result_raw[key] for key in ["模型", "交叉验证最佳RMSE", "验证集RMSE", "验证集MAE", "验证集R2", "调参耗时(s)", "最终训练与验证耗时(s)", "总耗时(s)"]},
        {key: result_pca[key] for key in ["模型", "交叉验证最佳RMSE", "验证集RMSE", "验证集MAE", "验证集R2", "调参耗时(s)", "最终训练与验证耗时(s)", "总耗时(s)"]}
    ])
    print("\n==============================")
    print("MLP 验证集结果汇总")
    print("==============================")
    print(compare_df)
    print("\n【PCA前详细结果】")
    print(result_raw)
    print("\n【PCA后详细结果】")
    print(result_pca)


result_mlp_raw = run_mlp_experiment(X_train_raw_scaled, X_val_raw_scaled, y_train, y_val, "MLP（PCA前）", "MLP_Before_PCA")
result_mlp_pca = run_mlp_experiment(X_train_pca.values, X_val_pca.values, y_train, y_val, "MLP（PCA后）", "MLP_After_PCA")
print_final_summary(result_mlp_raw, result_mlp_pca)
