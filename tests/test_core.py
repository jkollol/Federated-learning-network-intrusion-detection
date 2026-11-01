import numpy as np
import pandas as pd

from src.data import prepare_data
from src.federated import fedavg
from src.partition import dirichlet_partition


def test_dirichlet_partition_covers_data_and_is_reproducible():
    y = np.array([0] * 60 + [1] * 40)
    a = dirichlet_partition(y, n_clients=4, alpha=0.4, seed=9)
    b = dirichlet_partition(y, n_clients=4, alpha=0.4, seed=9)
    assert sorted(np.concatenate(a).tolist()) == list(range(len(y)))
    assert all(np.array_equal(x, z) for x, z in zip(a, b))


def test_prepare_data_binary_labels_and_finite_features():
    frame = pd.DataFrame({" f1 ": np.arange(100), "Flow ID": [f"id{i}" for i in range(100)],
                          " Label ": ["BENIGN"] * 50 + ["DoS"] * 50})
    x_train, x_test, y_train, y_test, features = prepare_data(frame, seed=3)
    assert features == ["f1"]
    assert set(np.unique(np.concatenate([y_train, y_test]))) == {0, 1}
    assert np.isfinite(x_train).all() and np.isfinite(x_test).all()


def test_fedavg_returns_finite_model_and_communication_cost():
    rng = np.random.default_rng(4)
    x = rng.normal(size=(100, 4))
    y = (x[:, 0] > 0).astype(int)
    parts = dirichlet_partition(y, 3, 0.8, 4)
    model, history, comm = fedavg(x, y, parts, rounds=2)
    assert len(history) == 2
    assert np.isfinite(model.coef_).all()
    assert comm["total_bytes"] == 2 * comm["bytes_per_round"] > 0
