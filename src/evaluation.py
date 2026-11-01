import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def binary_metrics(name, model, x_test, y_test):
    pred = model.predict(x_test)
    return {"method": name, "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0)}


def save_plot(metrics: pd.DataFrame, output_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    metric_cols = ["accuracy", "precision", "recall", "f1"]
    ax = metrics.set_index("method")[metric_cols].plot(kind="bar", ylim=(0, 1), figsize=(9, 5))
    ax.set_ylabel("Score")
    ax.set_title("Centralized baseline vs. federated averaging")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
