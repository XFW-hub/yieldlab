import base64
import io
import json
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from joblib import dump, load
from scipy import stats
from scipy.stats import kstest, pearsonr, probplot
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.base import clone

warnings.filterwarnings("ignore")

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:
    LGBMRegressor = None


plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Source Han Sans SC",
    "Arial Unicode MS",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False

FEATURE_ORDER = [
    "clonesize",
    "honeybee",
    "bumbles",
    "andrena",
    "osmia",
    "MaxOfUpperTRange",
    "MinOfUpperTRange",
    "AverageOfUpperTRange",
    "MaxOfLowerTRange",
    "MinOfLowerTRange",
    "AverageOfLowerTRange",
    "RainingDays",
    "AverageRainingDays",
    "fruitset",
    "fruitmass",
    "seeds",
    "yield",
]

DEFAULT_XGB_PARAMS = {
    False: {
        "n_estimators": 300,
        "max_depth": 4,
        "learning_rate": 0.03,
        "subsample": 0.9,
        "colsample_bytree": 0.6,
        "min_child_weight": 3,
        "gamma": 1,
        "reg_alpha": 0.01,
        "reg_lambda": 1,
    },
    True: {
        "n_estimators": 200,
        "max_depth": 5,
        "learning_rate": 0.05,
        "subsample": 1.0,
        "colsample_bytree": 0.6,
        "min_child_weight": 1,
        "gamma": 0.1,
        "reg_alpha": 0.1,
        "reg_lambda": 5,
    },
}

DEFAULT_LGB_PARAMS = {
    False: {
        "n_estimators": 500,
        "max_depth": 4,
        "learning_rate": 0.03,
        "num_leaves": 100,
        "min_child_samples": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "reg_alpha": 0.1,
        "reg_lambda": 10,
    },
    True: {
        "n_estimators": 200,
        "max_depth": 8,
        "learning_rate": 0.03,
        "num_leaves": 100,
        "min_child_samples": 50,
        "subsample": 0.6,
        "colsample_bytree": 0.9,
        "reg_alpha": 0.01,
        "reg_lambda": 10,
    },
}

DEFAULT_RF_PARAMS = {
    False: {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 6,
        "max_features": 0.6,
        "bootstrap": True,
    },
    True: {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_split": 2,
        "min_samples_leaf": 6,
        "max_features": 0.6,
        "bootstrap": True,
    },
}

NORMAL_APPROX_COLS = [
    "clonesize",
    "andrena",
    "osmia",
    "MaxOfUpperTRange",
    "MinOfUpperTRange",
    "AverageOfUpperTRange",
    "MaxOfLowerTRange",
    "MinOfLowerTRange",
    "AverageOfLowerTRange",
    "RainingDays",
    "AverageRainingDays",
    "fruitset",
    "fruitmass",
    "seeds",
    "yield",
]

NON_NORMAL_COLS = ["honeybee", "bumbles"]
MODEL_DIR = Path("models")


@dataclass
class PipelineArtifacts:
    train_original: pd.DataFrame
    test_original: pd.DataFrame
    train_cleaned: pd.DataFrame
    X_train_pca: pd.DataFrame
    X_test_pca: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def load_reference_training_data():
    train_path = "data/data.csv"
    raw_df = pd.read_csv(train_path)
    train_df, feature_columns, _ = prepare_training_data(raw_df)
    return train_df, feature_columns


def reference_training_summary(train_df: pd.DataFrame):
    raw_path = Path("data/data.csv")
    raw_rows = int(len(pd.read_csv(raw_path))) if raw_path.exists() else int(len(train_df))
    return {
        "raw_training_rows": raw_rows,
        "train_split_rows": None,
        "test_split_rows": None,
        "cleaned_training_rows": int(len(train_df)),
        "removed_rows": raw_rows - int(len(train_df)) if raw_rows >= len(train_df) else None,
        "removed_ratio": round((raw_rows - int(len(train_df))) / raw_rows, 4) if raw_rows else None,
    }


def prepare_training_data(training_df: pd.DataFrame):
    df = training_df.copy()
    df = df.drop(columns=["id"], errors="ignore")
    if "yield" not in df.columns:
        raise ValueError("训练数据必须包含 `yield` 列。")
    if len(df) < 20:
        raise ValueError("训练数据行数太少，至少需要 20 行以上。")

    train_original = df.reset_index(drop=True)
    empty_prediction_df = train_original.iloc[0:0].copy()
    train_cleaned, outlier_summary = process_outliers(train_original, empty_prediction_df)
    feature_columns = [col for col in train_cleaned.columns if col != "yield"]
    return train_cleaned, feature_columns, {
        "raw_training_rows": int(len(df)),
        "train_split_rows": None,
        "test_split_rows": None,
        "cleaned_training_rows": int(len(train_cleaned)),
        "removed_rows": outlier_summary["train_rows_removed"],
        "removed_ratio": outlier_summary["train_removed_ratio"],
    }


def normalize_uploaded_features(df: pd.DataFrame, feature_columns: List[str], fill_values: pd.Series):
    df = df.copy()
    df = df.drop(columns=["id", "yield"], errors="ignore")
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        raise ValueError(f"新数据缺少这些字段: {', '.join(missing)}")
    normalized = df[feature_columns].copy()
    normalized = normalized.apply(pd.to_numeric, errors="coerce")
    normalized = normalized.fillna(fill_values)
    return normalized


def parse_params(params_text: str, label: str):
    try:
        parsed = json.loads(params_text) if params_text.strip() else {}
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} 参数不是合法 JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label} 参数必须是 JSON 对象。")
    return parsed


def default_stacking_params(use_pca: bool):
    return {
        "cv": 5,
        "ridge": {"alpha": 1.0},
        "estimators": {
            "xgboost": DEFAULT_XGB_PARAMS[use_pca].copy(),
            "lightgbm": DEFAULT_LGB_PARAMS[use_pca].copy(),
            "random_forest": DEFAULT_RF_PARAMS[use_pca].copy(),
        },
    }


def fig_to_data_url(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def df_to_records(df: pd.DataFrame, limit: Optional[int] = None) -> Dict[str, List[Dict[str, object]]]:
    if limit is not None:
        df = df.head(limit)
    safe_df = df.replace({np.nan: None})
    return {
        "columns": safe_df.columns.tolist(),
        "rows": safe_df.to_dict(orient="records"),
    }


def ordered_numeric_columns(df: pd.DataFrame) -> List[str]:
    ordered = [col for col in FEATURE_ORDER if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    others = [
        col
        for col in df.columns
        if col not in ordered and pd.api.types.is_numeric_dtype(df[col])
    ]
    return ordered + others


def compute_basic_stats(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    basic_stats = df[numeric_cols].describe().T.round(4)
    basic_stats["skew"] = df[numeric_cols].skew(numeric_only=True).round(4)
    basic_stats["kurtosis"] = df[numeric_cols].kurt(numeric_only=True).round(4)
    return basic_stats.reset_index().rename(columns={"index": "feature"})


def build_boxplot(df: pd.DataFrame, numeric_cols: List[str]) -> str:
    n_cols = 3
    n_rows = int(np.ceil(len(numeric_cols) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, max(4, 3.8 * n_rows)))
    axes = np.array(axes).reshape(-1)

    for idx, col in enumerate(numeric_cols):
        ax = axes[idx]
        ax.boxplot(
            df[col].dropna(),
            patch_artist=True,
            boxprops=dict(facecolor="#bcd7ff", color="#3465d9"),
            whiskerprops=dict(color="#3465d9"),
            capprops=dict(color="#3465d9"),
            medianprops=dict(color="#ef4444", linewidth=2),
            flierprops=dict(marker="o", markerfacecolor="#64748b", markersize=3, alpha=0.55),
        )
        ax.set_title(col, fontsize=11)
        ax.grid(axis="y", alpha=0.25, linestyle="--")
        ax.set_xticklabels([])

    for idx in range(len(numeric_cols), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    return fig_to_data_url(fig)


def build_distribution_plots(df: pd.DataFrame, numeric_cols: List[str], max_features: int = 6) -> str:
    plot_cols = numeric_cols[:max_features]
    fig, axes = plt.subplots(len(plot_cols), 2, figsize=(12, max(4, 3.4 * len(plot_cols))))
    axes = np.array(axes).reshape(len(plot_cols), 2)

    for idx, col in enumerate(plot_cols):
        ax1, ax2 = axes[idx]
        series = df[col].dropna()
        ax1.hist(series, bins=30, density=True, alpha=0.65, color="#5fb878", edgecolor="#1f2937", linewidth=0.5)
        series.plot(kind="kde", ax=ax1, color="#dc2626", linewidth=1.6, label="KDE")
        mu, sigma = series.mean(), series.std()
        x = np.linspace(series.min(), series.max(), 200)
        if sigma > 0:
            ax1.plot(x, stats.norm.pdf(x, mu, sigma), linestyle="--", color="#0f766e", linewidth=1.6, label="Normal fit")
        ax1.set_title(f"{col} Histogram / KDE")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.25, linestyle="--")

        stats.probplot(series, plot=ax2, rvalue=True)
        ax2.set_title(f"{col} Q-Q Plot")
        ax2.grid(alpha=0.25, linestyle="--")

    fig.tight_layout()
    return fig_to_data_url(fig)


def build_variable_boxplot(df: pd.DataFrame, column: str) -> str:
    series = df[column].dropna()
    fig, ax = plt.subplots(figsize=(12, 5.2))
    ax.boxplot(
        series,
        vert=False,
        patch_artist=True,
        boxprops=dict(facecolor="#dbeafe", color="#2563eb"),
        whiskerprops=dict(color="#2563eb"),
        capprops=dict(color="#2563eb"),
        medianprops=dict(color="#dc2626", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="#64748b", markersize=4, alpha=0.55),
    )
    ax.set_title(f"{column} boxplot")
    ax.set_xlabel(column)
    ax.set_yticklabels([""])
    ax.grid(axis="x", alpha=0.28, linestyle="--")
    fig.tight_layout()
    return fig_to_data_url(fig)


def build_variable_distribution_plot(df: pd.DataFrame, column: str) -> str:
    series = df[column].dropna()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax1, ax2 = axes
    ax1.hist(series, bins=30, density=True, alpha=0.65, color="#13a06f", edgecolor="#17212b", linewidth=0.45)
    if len(series) > 1:
        series.plot(kind="kde", ax=ax1, color="#c6533d", linewidth=1.8, label="KDE")
    mu, sigma = series.mean(), series.std()
    if sigma and sigma > 0:
        x = np.linspace(series.min(), series.max(), 200)
        ax1.plot(x, stats.norm.pdf(x, mu, sigma), linestyle="--", color="#2f6df6", linewidth=1.6, label="Normal fit")
    ax1.set_title(f"{column} histogram / KDE")
    ax1.grid(alpha=0.28, linestyle="--")
    ax1.legend(fontsize=8)

    stats.probplot(series, plot=ax2, rvalue=True)
    ax2.set_title(f"{column} Q-Q plot")
    ax2.grid(alpha=0.28, linestyle="--")
    fig.tight_layout()
    return fig_to_data_url(fig)


def analyze_variable(df: pd.DataFrame, column: str) -> Dict[str, object]:
    cleaned = df.drop(columns=["id"], errors="ignore")
    numeric_cols = ordered_numeric_columns(cleaned)
    if column not in numeric_cols:
        raise ValueError(f"变量 `{column}` 不是可分析的数值列。")
    stats_df = compute_basic_stats(cleaned, [column])
    normality_df = compute_normality(cleaned, [column])
    return {
        "variable": column,
        "numeric_columns": numeric_cols,
        "stats": df_to_records(stats_df),
        "normality": df_to_records(normality_df),
        "images": {
            "boxplot": build_variable_boxplot(cleaned, column),
            "distribution": build_variable_distribution_plot(cleaned, column),
        },
    }


def describe_dataset(df: pd.DataFrame) -> Dict[str, object]:
    cleaned = df.copy().drop(columns=["id"], errors="ignore")
    numeric_cols = ordered_numeric_columns(cleaned)
    if not numeric_cols:
        raise ValueError("当前数据没有可描述的数值变量。")

    basic_stats = compute_basic_stats(cleaned, numeric_cols)
    return {
        "dataset": {
            "rows": int(len(cleaned)),
            "columns": int(cleaned.shape[1]),
            "numeric_columns": numeric_cols,
            "has_target": "yield" in cleaned.columns,
        },
        "basic_stats": df_to_records(basic_stats),
    }


def compute_normality(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    rows = []
    for col in numeric_cols:
        data = df[col].dropna()
        if data.empty:
            continue
        skewness = float(data.skew())
        kurtosis = float(data.kurtosis())
        mu, std = data.mean(), data.std()
        if std and std > 0:
            ks_stat, p_value = kstest(data, "norm", args=(mu, std))
        else:
            ks_stat, p_value = 0.0, 1.0
        (_, _), (_, _, r_value) = probplot(data, dist="norm", plot=None)
        if r_value > 0.95:
            qq_fit = "very_good"
        elif r_value > 0.9:
            qq_fit = "good"
        elif r_value > 0.8:
            qq_fit = "medium"
        else:
            qq_fit = "weak"

        if abs(skewness) < 1 and abs(kurtosis) < 1 and r_value > 0.9:
            judge = "normal"
        elif abs(skewness) < 2 and abs(kurtosis) < 2 and r_value > 0.8:
            judge = "approximately_normal"
        else:
            judge = "non_normal"

        rows.append(
            {
                "feature": col,
                "sample_size": int(len(data)),
                "skew": round(skewness, 4),
                "kurtosis": round(kurtosis, 4),
                "ks_stat": round(float(ks_stat), 4),
                "ks_p_value": round(float(p_value), 4),
                "qq_r": round(float(r_value), 4),
                "qq_fit": qq_fit,
                "judgement": judge,
            }
        )

    return pd.DataFrame(rows)


def process_outliers(train_df: pd.DataFrame, test_df: pd.DataFrame):
    train_outlier_mask = pd.DataFrame(False, index=train_df.index, columns=train_df.columns)
    train_stats = {}

    for col in NORMAL_APPROX_COLS:
        if col in train_df.columns:
            data = train_df[col].dropna()
            if not data.empty:
                mean = data.mean()
                std = data.std()
                lower = mean - 3 * std
                upper = mean + 3 * std
                train_stats[col] = {"type": "normal", "lower": lower, "upper": upper}
                train_outlier_mask.loc[data.index, col] = (data < lower) | (data > upper)

    for col in NON_NORMAL_COLS:
        if col in train_df.columns:
            data = train_df[col].dropna()
            if not data.empty:
                q1 = data.quantile(0.25)
                q3 = data.quantile(0.75)
                iqr = q3 - q1
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                train_stats[col] = {"type": "non_normal", "lower": lower, "upper": upper}
                train_outlier_mask.loc[data.index, col] = (data < lower) | (data > upper)

    train_rows_to_drop = train_outlier_mask.any(axis=1)
    train_clean = train_df.loc[~train_rows_to_drop].copy()

    test_outlier_mask = pd.DataFrame(False, index=test_df.index, columns=test_df.columns)
    for col, stats_info in train_stats.items():
        if col not in test_df.columns:
            continue
        data = test_df[col].dropna()
        if data.empty:
            continue
        test_outlier_mask.loc[data.index, col] = (data < stats_info["lower"]) | (data > stats_info["upper"])

    train_counts = []
    for col in train_df.columns:
        if col in train_outlier_mask.columns:
            count = int(train_outlier_mask[col].sum())
            if count:
                stats_info = train_stats.get(col, {})
                method_type = stats_info.get("type")
                if method_type == "normal":
                    method = "拉依达准则"
                    basis = "正态或近似正态"
                elif method_type == "non_normal":
                    method = "IQR"
                    basis = "非正态"
                else:
                    method = "-"
                    basis = "-"
                train_counts.append(
                    {
                        "feature": col,
                        "removed_rows": count,
                        "method": method,
                        "basis": basis,
                        "lower": round(float(stats_info["lower"]), 6) if "lower" in stats_info else None,
                        "upper": round(float(stats_info["upper"]), 6) if "upper" in stats_info else None,
                    }
                )

    test_counts = []
    for col in test_df.columns:
        if col in test_outlier_mask.columns:
            count = int(test_outlier_mask[col].sum())
            if count:
                test_counts.append({"feature": col, "outliers": count})

    summary = {
        "train_rows_before": int(len(train_df)),
        "train_rows_after": int(len(train_clean)),
        "train_rows_removed": int(train_rows_to_drop.sum()),
        "train_removed_ratio": round(float(train_rows_to_drop.mean()) if len(train_rows_to_drop) else 0.0, 4),
        "test_rows": int(len(test_df)),
        "test_outlier_rows": int(test_outlier_mask.any(axis=1).sum()),
        "test_outlier_ratio": round(float(test_outlier_mask.any(axis=1).mean()) if len(test_outlier_mask) else 0.0, 4),
        "train_feature_outliers": train_counts,
        "test_feature_outliers": test_counts,
    }
    return train_clean, summary


def build_correlation_heatmap(df: pd.DataFrame, feature_cols: List[str]) -> str:
    corr_df = df[feature_cols].dropna()
    corr_matrix = corr_df.corr(method="pearson").round(4)
    p_matrix = np.ones_like(corr_matrix, dtype=float)

    for i in range(len(corr_matrix.columns)):
        for j in range(i, len(corr_matrix.columns)):
            corr, p_value = pearsonr(corr_df[corr_matrix.columns[i]], corr_df[corr_matrix.columns[j]])
            p_matrix[i, j] = p_value
            p_matrix[j, i] = p_value

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = sns.diverging_palette(240, 10, as_cmap=True)
    sns.heatmap(
        corr_matrix,
        ax=ax,
        cmap=cmap,
        annot=True,
        fmt=".2f",
        vmin=-1,
        vmax=1,
        center=0,
        square=True,
        linewidths=0.5,
        annot_kws={"size": 7},
        cbar_kws={"shrink": 0.82},
    )

    for i in range(len(corr_matrix.columns)):
        for j in range(len(corr_matrix.columns)):
            mark = "**" if p_matrix[i, j] < 0.01 else "*" if p_matrix[i, j] < 0.05 else ""
            if mark:
                ax.text(j + 0.26, i + 0.15, mark, fontsize=6, color="black", fontweight="bold")

    ax.set_xticklabels(ax.get_xticklabels(), fontsize=9, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=9, rotation=0)
    fig.tight_layout()
    return fig_to_data_url(fig)


def run_pca(train_cleaned: pd.DataFrame, test_original: pd.DataFrame):
    target_col = "yield"
    X_train = train_cleaned.drop(columns=[target_col], errors="ignore")
    y_train = train_cleaned[target_col].copy()
    X_test = test_original.drop(columns=[target_col], errors="ignore")
    y_test = test_original[target_col].copy()

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_components = min(4, X_train.shape[1], len(X_train))
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)

    loading_df = pd.DataFrame(
        pca.components_,
        columns=X_train.columns,
        index=[f"PC{i + 1}" for i in range(pca.n_components_)],
    ).reset_index().rename(columns={"index": "component"})

    cum_var = np.cumsum(pca.explained_variance_ratio_)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(range(1, len(cum_var) + 1), cum_var, marker="o", linewidth=2.2, color="#1d4ed8")
    ax.axvline(x=pca.n_components_, linestyle="--", linewidth=1.5, color="#dc2626")
    ax.scatter(pca.n_components_, cum_var[pca.n_components_ - 1], s=60, zorder=3, color="#dc2626")
    ax.set_xlabel("Principal Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Variance Curve")
    ax.set_xticks(range(1, len(cum_var) + 1))
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()

    summary = {
        "n_components": int(pca.n_components_),
        "explained_variance_ratio": [round(float(x), 4) for x in pca.explained_variance_ratio_],
        "cumulative_explained_variance": round(float(cum_var[-1]), 4),
    }

    artifacts = PipelineArtifacts(
        train_original=train_cleaned.copy(),
        test_original=test_original.copy(),
        train_cleaned=train_cleaned.copy(),
        X_train_pca=pd.DataFrame(X_train_pca, columns=[f"PC{i + 1}" for i in range(pca.n_components_)]),
        X_test_pca=pd.DataFrame(X_test_pca, columns=[f"PC{i + 1}" for i in range(pca.n_components_)]),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
    )
    return summary, loading_df, fig_to_data_url(fig), artifacts


def evaluate_model(model, X_train, X_eval, y_train, y_eval, model_name: str):
    start = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_eval)
    elapsed = time.time() - start
    return {
        "model": model_name,
        "r2": round(float(r2_score(y_eval, predictions)), 4),
        "mae": round(float(mean_absolute_error(y_eval, predictions)), 4),
        "rmse": round(float(np.sqrt(mean_squared_error(y_eval, predictions))), 4),
        "time_seconds": round(float(elapsed), 3),
        "predictions": predictions,
    }


def build_pred_plot(y_true, y_pred, title: str) -> str:
    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.scatter(y_true, y_pred, alpha=0.62, s=25, edgecolors="k", linewidths=0.3, color="#2563eb")
    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=2, color="#ef4444", label="Ideal fit")
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.text(0.05, 0.95, f"$R^2={r2_score(y_true, y_pred):.4f}$", transform=ax.transAxes, fontsize=11, va="top")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    return fig_to_data_url(fig)


def run_random_search(model, param_dist, X_train, y_train, n_iter=4):
    search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_dist,
        n_iter=n_iter,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=3, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    start = time.time()
    search.fit(X_train, y_train)
    return search.best_estimator_, round(float(-search.best_score_), 4), round(float(time.time() - start), 3)


def run_mlp_training(X_train, X_val, y_train, y_val, use_scaled_raw: bool):
    scaler = StandardScaler() if use_scaled_raw else None
    X_train_model = scaler.fit_transform(X_train) if scaler is not None else X_train.values
    X_val_model = scaler.transform(X_val) if scaler is not None else X_val.values

    param_dist = {
        "hidden_layer_sizes": [(32,), (64,), (64, 32)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.005],
        "batch_size": [32, 64],
    }
    search_model = MLPRegressor(random_state=42, solver="adam", max_iter=500)
    best_model, best_cv_rmse, tune_time = run_random_search(search_model, param_dist, X_train_model, y_train, n_iter=4)

    result = evaluate_model(best_model, X_train_model, X_val_model, y_train, y_val, "MLP")
    result["cv_rmse"] = best_cv_rmse
    result["tune_time_seconds"] = tune_time
    result["best_params"] = best_model.get_params()
    result["plot"] = build_pred_plot(y_val, result.pop("predictions"), f"MLP {'Raw' if use_scaled_raw else 'PCA'} Validation")
    return result


def maybe_run_lightgbm(X_train, X_val, y_train, y_val):
    if LGBMRegressor is None:
        return None
    param_dist = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [-1, 4, 6],
        "num_leaves": [15, 31, 50],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
    }
    best_model, best_cv_rmse, tune_time = run_random_search(
        LGBMRegressor(objective="regression", random_state=42, n_jobs=-1, verbose=-1),
        param_dist,
        X_train,
        y_train,
        n_iter=4,
    )
    result = evaluate_model(best_model, X_train, X_val, y_train, y_val, "LightGBM")
    result["cv_rmse"] = best_cv_rmse
    result["tune_time_seconds"] = tune_time
    result["best_params"] = best_model.get_params()
    result["plot"] = build_pred_plot(y_val, result.pop("predictions"), "LightGBM Validation")
    return result


def maybe_run_xgboost(X_train, X_val, y_train, y_val):
    if XGBRegressor is None:
        return None
    param_dist = {
        "n_estimators": [100, 200, 300],
        "learning_rate": [0.03, 0.05, 0.1],
        "max_depth": [3, 4, 5],
        "min_child_weight": [1, 3, 5],
        "subsample": [0.7, 0.9, 1.0],
        "colsample_bytree": [0.7, 0.9, 1.0],
    }
    best_model, best_cv_rmse, tune_time = run_random_search(
        XGBRegressor(objective="reg:squarederror", random_state=42, n_jobs=-1),
        param_dist,
        X_train,
        y_train,
        n_iter=4,
    )
    result = evaluate_model(best_model, X_train, X_val, y_train, y_val, "XGBoost")
    result["cv_rmse"] = best_cv_rmse
    result["tune_time_seconds"] = tune_time
    result["best_params"] = best_model.get_params()
    result["plot"] = build_pred_plot(y_val, result.pop("predictions"), "XGBoost Validation")
    return result


def run_random_forest(X_train, X_val, y_train, y_val):
    param_dist = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 5, 10, 15],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "max_features": ["sqrt", "log2", 0.8],
    }
    best_model, best_cv_rmse, tune_time = run_random_search(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_dist,
        X_train,
        y_train,
        n_iter=4,
    )
    result = evaluate_model(best_model, X_train, X_val, y_train, y_val, "RandomForest")
    result["cv_rmse"] = best_cv_rmse
    result["tune_time_seconds"] = tune_time
    result["best_params"] = best_model.get_params()
    result["plot"] = build_pred_plot(y_val, result.pop("predictions"), "RandomForest Validation")
    return result


def run_stacking(X_train, X_val, y_train, y_val):
    estimators = [("rf", RandomForestRegressor(n_estimators=200, max_depth=12, random_state=42, n_jobs=-1))]
    if XGBRegressor is not None:
        estimators.append(
            ("xgb", XGBRegressor(
                objective="reg:squarederror",
                n_estimators=200,
                max_depth=4,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                n_jobs=-1,
            ))
        )
    if LGBMRegressor is not None:
        estimators.append(
            ("lgb", LGBMRegressor(
                n_estimators=200,
                learning_rate=0.05,
                num_leaves=31,
                random_state=42,
                n_jobs=-1,
                verbose=-1,
            ))
        )

    model = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(alpha=1.0),
        cv=3,
        n_jobs=-1,
    )
    result = evaluate_model(model, X_train, X_val, y_train, y_val, "Stacking")
    result["cv_rmse"] = None
    result["tune_time_seconds"] = 0.0
    result["best_params"] = {"estimators": [name for name, _ in estimators], "final_estimator": "Ridge(alpha=1.0)"}
    result["plot"] = build_pred_plot(y_val, result.pop("predictions"), "Stacking Validation")
    return result


def run_models(artifacts: PipelineArtifacts, selected_models: List[str]):
    train_idx, val_idx = train_test_split(
        np.arange(len(artifacts.y_train)),
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    raw_features = artifacts.train_cleaned.drop(columns=["yield"], errors="ignore").reset_index(drop=True)
    raw_train = raw_features.iloc[train_idx].reset_index(drop=True)
    raw_val = raw_features.iloc[val_idx].reset_index(drop=True)
    pca_train = artifacts.X_train_pca.iloc[train_idx].reset_index(drop=True)
    pca_val = artifacts.X_train_pca.iloc[val_idx].reset_index(drop=True)
    y_train = artifacts.y_train.iloc[train_idx].reset_index(drop=True)
    y_val = artifacts.y_train.iloc[val_idx].reset_index(drop=True)

    experiments = []
    skipped = []

    def append_result(name: str, mode: str, result: Optional[Dict[str, object]]):
        if result is None:
            skipped.append({"model": name, "reason": "package_not_available"})
            return
        experiments.append(
            {
                "model": name,
                "mode": mode,
                "cv_rmse": result["cv_rmse"],
                "rmse": result["rmse"],
                "mae": result["mae"],
                "r2": result["r2"],
                "tune_time_seconds": result["tune_time_seconds"],
                "time_seconds": result["time_seconds"],
                "best_params": result["best_params"],
                "plot": result["plot"],
            }
        )

    if "random_forest" in selected_models:
        append_result("RandomForest", "raw", run_random_forest(raw_train, raw_val, y_train, y_val))
        append_result("RandomForest", "pca", run_random_forest(pca_train, pca_val, y_train, y_val))

    if "xgboost" in selected_models:
        append_result("XGBoost", "raw", maybe_run_xgboost(raw_train, raw_val, y_train, y_val))
        append_result("XGBoost", "pca", maybe_run_xgboost(pca_train, pca_val, y_train, y_val))

    if "lightgbm" in selected_models:
        append_result("LightGBM", "raw", maybe_run_lightgbm(raw_train, raw_val, y_train, y_val))
        append_result("LightGBM", "pca", maybe_run_lightgbm(pca_train, pca_val, y_train, y_val))

    if "mlp" in selected_models:
        append_result("MLP", "raw", run_mlp_training(raw_train, raw_val, y_train, y_val, use_scaled_raw=True))
        append_result("MLP", "pca", run_mlp_training(pca_train, pca_val, y_train, y_val, use_scaled_raw=False))

    if "stacking" in selected_models:
        append_result("Stacking", "raw", run_stacking(raw_train, raw_val, y_train, y_val))
        append_result("Stacking", "pca", run_stacking(pca_train, pca_val, y_train, y_val))

    experiments.sort(key=lambda item: (item["rmse"], -item["r2"]))
    return experiments, skipped


def run_full_pipeline(df: pd.DataFrame, selected_models: Optional[List[str]] = None) -> Dict[str, object]:
    selected_models = selected_models or ["random_forest", "xgboost", "mlp", "stacking"]
    df = df.copy()
    df = df.drop(columns=["id"], errors="ignore")

    numeric_cols = ordered_numeric_columns(df)
    if "yield" not in df.columns:
        raise ValueError("数据中缺少目标列 `yield`，无法完成与项目一致的建模流程。")
    if len(df) < 20:
        raise ValueError("数据行数太少，至少需要 20 行以上才能稳定完成拆分和建模。")

    basic_stats = compute_basic_stats(df, numeric_cols)
    boxplot_url = build_boxplot(df, numeric_cols)

    train_original, test_original = train_test_split(
        df,
        test_size=0.3,
        random_state=42,
        shuffle=True,
    )
    train_original = train_original.reset_index(drop=True)
    test_original = test_original.reset_index(drop=True)

    distribution_url = build_distribution_plots(train_original, [col for col in numeric_cols if col in train_original.columns])
    normality_df = compute_normality(train_original, [col for col in numeric_cols if col in train_original.columns])

    train_cleaned, outlier_summary = process_outliers(train_original, test_original)
    correlation_features = [col for col in ordered_numeric_columns(train_cleaned) if col in train_cleaned.columns]
    correlation_url = build_correlation_heatmap(train_cleaned, correlation_features)

    pca_summary, pca_loading_df, pca_curve_url, artifacts = run_pca(train_cleaned, test_original)
    model_results, skipped_models = run_models(artifacts, selected_models)

    return {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "numeric_columns": numeric_cols,
        },
        "split": {
            "train_rows": int(len(train_original)),
            "test_rows": int(len(test_original)),
        },
        "basic_stats": df_to_records(basic_stats),
        "normality": df_to_records(normality_df),
        "pca_loadings": df_to_records(pca_loading_df, limit=12),
        "outlier_summary": outlier_summary,
        "model_results": model_results,
        "skipped_models": skipped_models,
        "images": {
            "boxplot": boxplot_url,
            "distribution": distribution_url,
            "correlation": correlation_url,
            "pca_curve": pca_curve_url,
        },
        "pca_summary": pca_summary,
        "preview": {
            "raw_head": df_to_records(df, limit=8),
            "train_cleaned_head": df_to_records(train_cleaned, limit=8),
        },
    }


def analyze_uploaded_data(df: pd.DataFrame) -> Dict[str, object]:
    df = df.copy()
    df = df.drop(columns=["id"], errors="ignore")
    numeric_cols = ordered_numeric_columns(df)
    if not numeric_cols:
        raise ValueError("上传数据中没有可分析的数值列。")

    basic_stats = compute_basic_stats(df, numeric_cols)
    normality_df = compute_normality(df, numeric_cols)
    boxplot_url = build_boxplot(df, numeric_cols)
    distribution_url = build_distribution_plots(df, numeric_cols)
    preprocessing = None
    cleaned_preview = None
    correlation_url = None

    if "yield" in df.columns and len(df) >= 20:
        train_original = df.reset_index(drop=True)
        empty_prediction_df = train_original.iloc[0:0].copy()
        train_cleaned, outlier_summary = process_outliers(train_original, empty_prediction_df)
        preprocessing = {
            "train_rows": int(len(train_original)),
            "test_rows": 0,
            "cleaned_train_rows": int(len(train_cleaned)),
            "removed_rows": outlier_summary["train_rows_removed"],
            "removed_ratio": outlier_summary["train_removed_ratio"],
            "test_outlier_rows": 0,
            "test_outlier_ratio": 0.0,
            "train_feature_outliers": outlier_summary["train_feature_outliers"],
            "test_feature_outliers": [],
        }
        cleaned_preview = df_to_records(train_cleaned, limit=8)
        corr_cols = [col for col in ordered_numeric_columns(train_cleaned) if col in train_cleaned.columns]
        if len(corr_cols) >= 2:
            correlation_url = build_correlation_heatmap(train_cleaned, corr_cols)

    return {
        "dataset": {
            "rows": int(len(df)),
            "columns": int(df.shape[1]),
            "numeric_columns": numeric_cols,
        },
        "basic_stats": df_to_records(basic_stats),
        "normality": df_to_records(normality_df),
        "preprocessing": preprocessing,
        "cleaned_preview": cleaned_preview,
        "images": {
            "boxplot": boxplot_url,
            "distribution": distribution_url,
            "correlation": correlation_url,
        },
    }


def train_xgboost_with_params(X_train, y_train, X_new, params: Dict[str, object], use_pca: bool):
    if XGBRegressor is None:
        raise ValueError("当前环境未安装 xgboost，无法运行 XGBoost 模型。")

    model_params = {
        "objective": "reg:squarederror",
        "random_state": 42,
        "n_jobs": 1,
    }
    model_params.update(params)
    model = XGBRegressor(**model_params)
    start = time.time()
    model.fit(X_train, y_train)
    predictions = model.predict(X_new)
    elapsed = time.time() - start
    return {
        "model": "XGBoost",
        "mode": "pca" if use_pca else "raw",
        "time_seconds": round(float(elapsed), 3),
        "params": model_params,
        "predictions": [round(float(x), 6) for x in predictions],
    }


def fit_xgboost_model(X_train, y_train, params: Dict[str, object]):
    if XGBRegressor is None:
        raise ValueError("当前环境未安装 xgboost，无法运行 XGBoost 模型。")
    model_params = {"objective": "reg:squarederror", "random_state": 42, "n_jobs": 1}
    model_params.update(params)
    model = XGBRegressor(**model_params)
    model.fit(X_train, y_train)
    return model, model_params


def train_stacking_with_params(X_train, y_train, X_new, params: Dict[str, object], use_pca: bool):
    base_params = params.copy()
    ridge_params = base_params.pop("ridge", {"alpha": 1.0})
    estimator_params = base_params.pop("estimators", {})
    cv = int(base_params.pop("cv", 3))

    estimators = []

    rf_params = estimator_params.get("random_forest")
    if rf_params is not None:
        config = {"random_state": 42, "n_jobs": 1}
        config.update(rf_params)
        estimators.append(("rf", RandomForestRegressor(**config)))

    xgb_params = estimator_params.get("xgboost")
    if xgb_params is not None:
        if XGBRegressor is None:
            raise ValueError("当前环境未安装 xgboost，无法构建包含 XGBoost 的 Stacking。")
        config = {"objective": "reg:squarederror", "random_state": 42, "n_jobs": 1}
        config.update(xgb_params)
        estimators.append(("xgb", XGBRegressor(**config)))

    lgb_params = estimator_params.get("lightgbm")
    if lgb_params is not None:
        if LGBMRegressor is None:
            raise ValueError("当前环境未安装 lightgbm，无法构建包含 LightGBM 的 Stacking。")
        config = {"objective": "regression", "random_state": 42, "n_jobs": 1, "verbose": -1}
        config.update(lgb_params)
        estimators.append(("lgb", LGBMRegressor(**config)))

    if not estimators:
        raise ValueError("Stacking 参数里至少需要提供一个基础学习器。")

    start = time.time()
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    X_train_np = np.asarray(X_train)
    X_new_np = np.asarray(X_new)
    y_train_np = np.asarray(y_train)

    meta_train = np.zeros((len(X_train_np), len(estimators)))
    for est_idx, (_, estimator) in enumerate(estimators):
        for train_idx, valid_idx in kf.split(X_train_np):
            fold_model = clone(estimator)
            fold_model.fit(X_train_np[train_idx], y_train_np[train_idx])
            meta_train[valid_idx, est_idx] = fold_model.predict(X_train_np[valid_idx])

    meta_model = Ridge(**ridge_params)
    meta_model.fit(meta_train, y_train_np)

    fitted_estimators = []
    meta_new = np.zeros((len(X_new_np), len(estimators)))
    for est_idx, (name, estimator) in enumerate(estimators):
        fitted_model = clone(estimator)
        fitted_model.fit(X_train_np, y_train_np)
        fitted_estimators.append((name, fitted_model))
        meta_new[:, est_idx] = fitted_model.predict(X_new_np)

    predictions = meta_model.predict(meta_new)
    elapsed = time.time() - start
    return {
        "model": "Stacking",
        "mode": "pca" if use_pca else "raw",
        "time_seconds": round(float(elapsed), 3),
        "params": {
            "estimators": list(estimator_params.keys()),
            "ridge": ridge_params,
            "cv": cv,
        },
        "predictions": [round(float(x), 6) for x in predictions],
    }


def build_stacking_estimators(params: Dict[str, object]):
    base_params = params.copy()
    ridge_params = base_params.pop("ridge", {"alpha": 1.0})
    estimator_params = base_params.pop("estimators", {})
    cv = int(base_params.pop("cv", 3))
    estimators = []

    rf_params = estimator_params.get("random_forest")
    if rf_params is not None:
        config = {"random_state": 42, "n_jobs": 1}
        config.update(rf_params)
        estimators.append(("rf", RandomForestRegressor(**config)))

    xgb_params = estimator_params.get("xgboost")
    if xgb_params is not None:
        if XGBRegressor is None:
            raise ValueError("当前环境未安装 xgboost，无法构建包含 XGBoost 的 Stacking。")
        config = {"objective": "reg:squarederror", "random_state": 42, "n_jobs": 1}
        config.update(xgb_params)
        estimators.append(("xgb", XGBRegressor(**config)))

    lgb_params = estimator_params.get("lightgbm")
    if lgb_params is not None:
        if LGBMRegressor is None:
            raise ValueError("当前环境未安装 lightgbm，无法构建包含 LightGBM 的 Stacking。")
        config = {"objective": "regression", "random_state": 42, "n_jobs": 1, "verbose": -1}
        config.update(lgb_params)
        estimators.append(("lgb", LGBMRegressor(**config)))

    if not estimators:
        raise ValueError("Stacking 参数里至少需要提供一个基础学习器。")
    return estimators, ridge_params, cv


def fit_stacking_model(X_train, y_train, params: Dict[str, object]):
    estimators, ridge_params, cv = build_stacking_estimators(params)
    kf = KFold(n_splits=cv, shuffle=True, random_state=42)
    X_train_np = np.asarray(X_train)
    y_train_np = np.asarray(y_train)
    meta_train = np.zeros((len(X_train_np), len(estimators)))

    for est_idx, (_, estimator) in enumerate(estimators):
        for train_idx, valid_idx in kf.split(X_train_np):
            fold_model = clone(estimator)
            fold_model.fit(X_train_np[train_idx], y_train_np[train_idx])
            meta_train[valid_idx, est_idx] = fold_model.predict(X_train_np[valid_idx])

    meta_model = Ridge(**ridge_params)
    meta_model.fit(meta_train, y_train_np)

    fitted_estimators = []
    for name, estimator in estimators:
        fitted_model = clone(estimator)
        fitted_model.fit(X_train_np, y_train_np)
        fitted_estimators.append((name, fitted_model))

    return {"estimators": fitted_estimators, "meta_model": meta_model, "cv": cv, "ridge": ridge_params}


def predict_stacking_bundle(bundle: Dict[str, object], X_new):
    X_new_np = np.asarray(X_new)
    meta_new = np.zeros((len(X_new_np), len(bundle["estimators"])))
    for est_idx, (_, fitted_model) in enumerate(bundle["estimators"]):
        meta_new[:, est_idx] = fitted_model.predict(X_new_np)
    return bundle["meta_model"].predict(meta_new)


def builtin_model_path(model_name: str):
    MODEL_DIR.mkdir(exist_ok=True)
    return MODEL_DIR / f"builtin_{model_name}_website_raw.joblib"


def get_builtin_model(model_name: str, params: Dict[str, object], X_train, y_train):
    path = builtin_model_path(model_name)
    if path.exists():
        return load(path), True
    if model_name == "xgboost":
        model, _ = fit_xgboost_model(X_train, y_train, params)
    elif model_name == "stacking":
        model = fit_stacking_model(X_train, y_train, params)
    else:
        raise ValueError("未知模型。")
    dump(model, path)
    return model, False


def train_models_for_website(
    training_data_df: Optional[pd.DataFrame] = None,
    use_builtin_model: bool = False,
    model_choice: str = "both",
    xgb_params_text: str = "",
    stacking_params_text: str = "",
    run_xgb: bool = True,
    run_stacking: bool = True,
):
    if use_builtin_model or training_data_df is None:
        train_df, feature_columns = load_reference_training_data()
        training_summary = reference_training_summary(train_df)
        source = "builtin_cached"
    else:
        train_df, feature_columns, training_summary = prepare_training_data(training_data_df)
        source = "trained_from_uploaded_data"

    X_train_raw = train_df[feature_columns].copy()
    y_train = train_df["yield"].copy()
    run_xgb = model_choice == "xgboost" or (model_choice == "both" and run_xgb)
    run_stacking = model_choice == "stacking" or (model_choice == "both" and run_stacking)

    results = []
    if run_xgb:
        xgb_params = parse_params(xgb_params_text, "XGBoost")
        if not xgb_params:
            xgb_params = DEFAULT_XGB_PARAMS[False].copy()
        start = time.time()
        if use_builtin_model:
            _, cache_hit = get_builtin_model("xgboost", xgb_params, X_train_raw, y_train)
            result_source = "builtin_cached" if cache_hit else "builtin_cache_created"
        else:
            fit_xgboost_model(X_train_raw, y_train, xgb_params)
            result_source = source
            cache_hit = False
        results.append(
            {
                "model": "XGBoost",
                "mode": "raw",
                "time_seconds": round(float(time.time() - start), 3),
                "source": result_source,
                "cache_hit": cache_hit,
                "params": xgb_params,
            }
        )

    if run_stacking:
        stacking_params = parse_params(stacking_params_text, "Stacking")
        if not stacking_params:
            stacking_params = default_stacking_params(False)
        start = time.time()
        if use_builtin_model:
            model, cache_hit = get_builtin_model("stacking", stacking_params, X_train_raw, y_train)
            result_source = "builtin_cached" if cache_hit else "builtin_cache_created"
            params = {"estimators": [name for name, _ in model["estimators"]], "ridge": model["ridge"], "cv": model["cv"]}
        else:
            fit_stacking_model(X_train_raw, y_train, stacking_params)
            result_source = source
            cache_hit = False
            estimators, ridge_params, cv = build_stacking_estimators(stacking_params.copy())
            params = {"estimators": [name for name, _ in estimators], "ridge": ridge_params, "cv": cv}
        results.append(
            {
                "model": "Stacking",
                "mode": "raw",
                "time_seconds": round(float(time.time() - start), 3),
                "source": result_source,
                "cache_hit": cache_hit,
                "params": params,
            }
        )

    return {
        "summary": {
            "training_rows": int(len(train_df)),
            "feature_count": int(len(feature_columns)),
            "feature_columns": feature_columns,
            "training": training_summary,
        },
        "results": results,
    }


def predict_new_data(
    new_data_df: pd.DataFrame,
    training_data_df: Optional[pd.DataFrame] = None,
    use_builtin_model: bool = False,
    model_choice: str = "stacking",
    xgb_params_text: str = "",
    stacking_params_text: str = "",
    run_xgb: bool = True,
    run_stacking: bool = True,
    xgb_use_pca: bool = False,
    stacking_use_pca: bool = False,
):
    if use_builtin_model:
        train_df, feature_columns = load_reference_training_data()
        training_summary = reference_training_summary(train_df)
    elif training_data_df is None:
        train_df, feature_columns = load_reference_training_data()
        training_summary = reference_training_summary(train_df)
    else:
        train_df, feature_columns, training_summary = prepare_training_data(training_data_df)

    fill_values = train_df[feature_columns].median(numeric_only=True)
    new_features = normalize_uploaded_features(new_data_df, feature_columns, fill_values)

    X_train_raw = train_df[feature_columns].copy()
    y_train = train_df["yield"].copy()

    needs_pca = (
        (model_choice in {"xgboost", "both"} and xgb_use_pca)
        or (model_choice in {"stacking", "both"} and stacking_use_pca)
    )
    X_train_pca = None
    X_new_pca = None
    n_components = 0
    pca_variance = []
    if needs_pca:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)
        X_new_scaled = scaler.transform(new_features)

        n_components = min(4, X_train_raw.shape[1], len(X_train_raw))
        pca = PCA(n_components=n_components, random_state=42)
        X_train_pca = pca.fit_transform(X_train_scaled)
        X_new_pca = pca.transform(X_new_scaled)
        pca_variance = [round(float(x), 4) for x in pca.explained_variance_ratio_]

    summary = {
        "training_rows": int(len(train_df)),
        "training": training_summary,
        "feature_count": int(len(feature_columns)),
        "new_rows": int(len(new_features)),
        "feature_columns": feature_columns,
        "pca_components": int(n_components),
        "pca_variance": pca_variance,
    }

    results = []
    run_xgb = model_choice == "xgboost" or (model_choice == "both" and run_xgb)
    run_stacking = model_choice == "stacking" or (model_choice == "both" and run_stacking)

    if run_xgb:
        xgb_params = parse_params(xgb_params_text, "XGBoost")
        if not xgb_params:
            xgb_params = DEFAULT_XGB_PARAMS[xgb_use_pca].copy()
        if use_builtin_model and not xgb_use_pca:
            start = time.time()
            model, cache_hit = get_builtin_model("xgboost", xgb_params, X_train_raw, y_train)
            predictions = model.predict(new_features)
            xgb_result = {
                "model": "XGBoost",
                "mode": "raw",
                "time_seconds": round(float(time.time() - start), 3),
                "params": xgb_params,
                "cache_hit": cache_hit,
                "source": "builtin_cached" if cache_hit else "builtin_cache_created",
                "predictions": [round(float(x), 6) for x in predictions],
            }
        else:
            xgb_result = train_xgboost_with_params(
                X_train_pca if xgb_use_pca else X_train_raw,
                y_train,
                X_new_pca if xgb_use_pca else new_features,
                xgb_params,
                use_pca=xgb_use_pca,
            )
            xgb_result["cache_hit"] = False
            xgb_result["source"] = "trained_from_uploaded_data" if training_data_df is not None else "trained_from_reference_data"
        results.append(xgb_result)

    if run_stacking:
        stacking_params = parse_params(stacking_params_text, "Stacking")
        if not stacking_params:
            stacking_params = default_stacking_params(stacking_use_pca)
        if use_builtin_model and not stacking_use_pca:
            start = time.time()
            model, cache_hit = get_builtin_model("stacking", stacking_params, X_train_raw, y_train)
            predictions = predict_stacking_bundle(model, new_features)
            stacking_result = {
                "model": "Stacking",
                "mode": "raw",
                "time_seconds": round(float(time.time() - start), 3),
                "params": {"estimators": [name for name, _ in model["estimators"]], "ridge": model["ridge"], "cv": model["cv"]},
                "cache_hit": cache_hit,
                "source": "builtin_cached" if cache_hit else "builtin_cache_created",
                "predictions": [round(float(x), 6) for x in predictions],
            }
        else:
            stacking_result = train_stacking_with_params(
                X_train_pca if stacking_use_pca else X_train_raw,
                y_train,
                X_new_pca if stacking_use_pca else new_features,
                stacking_params,
                use_pca=stacking_use_pca,
            )
            stacking_result["cache_hit"] = False
            stacking_result["source"] = "trained_from_uploaded_data" if training_data_df is not None else "trained_from_reference_data"
        results.append(stacking_result)

    prediction_table = pd.DataFrame({"row_id": np.arange(1, len(new_features) + 1)})
    for result in results:
        prediction_table[f"{result['model']}_{result['mode']}"] = result["predictions"]

    return {
        "summary": summary,
        "new_data_preview": df_to_records(new_data_df, limit=8),
        "normalized_preview": df_to_records(new_features, limit=8),
        "results": results,
        "prediction_table": df_to_records(prediction_table),
    }
