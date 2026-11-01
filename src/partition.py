import numpy as np


def dirichlet_partition(y: np.ndarray, n_clients: int = 5, alpha: float = 0.5, seed: int = 42,
                        min_samples: int = 2) -> list[np.ndarray]:
    """Split indices with class proportions drawn from a Dirichlet distribution."""
    if n_clients < 2:
        raise ValueError("n_clients must be at least 2")
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    for _ in range(100):
        buckets: list[list[int]] = [[] for _ in range(n_clients)]
        for label in np.unique(y):
            indices = np.flatnonzero(y == label)
            rng.shuffle(indices)
            proportions = rng.dirichlet(np.full(n_clients, alpha))
            cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(int)
            for bucket, part in zip(buckets, np.split(indices, cuts)):
                bucket.extend(part.tolist())
        if min(map(len, buckets)) >= min_samples:
            return [np.asarray(rng.permutation(bucket), dtype=np.int64) for bucket in buckets]
    raise ValueError("Could not create non-empty client partitions; use more records or a larger alpha.")
