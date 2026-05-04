import numpy as np
from sklearn.metrics import r2_score
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
def plot_true_vs_pred_advanced(y_true, y_pred, title="真实值与预测值对比", save_path=None):
    r2 = r2_score(y_true, y_pred)

    plt.figure(figsize=(6.5, 6))

    plt.scatter(
        y_true, y_pred,
        alpha=0.6,
        s=25,
        edgecolors='k',
        linewidths=0.3
    )

    min_val = min(np.min(y_true), np.min(y_pred))
    max_val = max(np.max(y_true), np.max(y_pred))

    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        linestyle='--',
        linewidth=2,
        label='理想预测'
    )

    plt.xlabel('真实值', fontsize=12)
    plt.ylabel('预测值', fontsize=12)
    plt.title(title, fontsize=13)

    plt.text(
        0.05, 0.95,
        f"$R^2 = {r2:.4f}$",
        transform=plt.gca().transAxes,
        fontsize=11,
        verticalalignment='top'
    )

    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()