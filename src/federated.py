import copy

import numpy as np
from sklearn.linear_model import SGDClassifier


def new_model(seed: int = 42) -> SGDClassifier:
    return SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4, max_iter=1,
                         tol=None, learning_rate="optimal", random_state=seed)


def fedavg(x: np.ndarray, y: np.ndarray, partitions: list[np.ndarray], rounds: int = 5,
           local_epochs: int = 1, seed: int = 42):
    if rounds < 1 or local_epochs < 1:
        raise ValueError("rounds and local_epochs must be positive")
    classes = np.unique(y)
    global_model = new_model(seed)
    history = []
    for round_index in range(rounds):
        client_models, client_sizes = [], []
        for client_id, indices in enumerate(partitions):
            model = copy.deepcopy(global_model) if round_index else new_model(seed + client_id)
            for epoch in range(local_epochs):
                model.partial_fit(x[indices], y[indices], classes=classes if epoch == 0 else None)
            client_models.append(model)
            client_sizes.append(len(indices))
        weights = np.asarray(client_sizes, dtype=float) / sum(client_sizes)
        global_model.classes_ = classes
        global_model.coef_ = sum(w * m.coef_ for w, m in zip(weights, client_models))
        global_model.intercept_ = sum(w * m.intercept_ for w, m in zip(weights, client_models))
        global_model.n_features_in_ = x.shape[1]
        global_model.n_iter_ = np.asarray([local_epochs])
        global_model.t_ = float(sum(w * m.t_ for w, m in zip(weights, client_models)))
        history.append({"round": round_index + 1, "n_clients": len(partitions), "train_samples": int(sum(client_sizes))})
    parameter_count = int(global_model.coef_.size + global_model.intercept_.size)
    bytes_per_model = parameter_count * np.dtype(np.float64).itemsize
    per_round_bytes = bytes_per_model * len(partitions) * 2
    return global_model, history, {"parameters": parameter_count, "bytes_per_model": bytes_per_model,
                                  "bytes_per_round": per_round_bytes, "total_bytes": per_round_bytes * rounds}
