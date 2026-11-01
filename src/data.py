from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LABEL_CANDIDATES = {"label", "class", "attack"}
DROP_NAME_PARTS = ("flow id", "source ip", "destination ip", "timestamp", "src ip", "dst ip")


def _clean_name(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).replace("\ufeff", "")).strip()


def load_csv_folder(folder: str | Path, max_rows: int | None = None, seed: int = 42) -> pd.DataFrame:
    paths = sorted(Path(folder).glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV files found in {folder}. Download CICIDS2017 flow CSVs and place them there.")
    frames = []
    for path in paths:
        frame = pd.read_csv(path, low_memory=False)
        frame.columns = [_clean_name(c) for c in frame.columns]
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if max_rows is not None and len(data) > max_rows:
        data = data.sample(n=max_rows, random_state=seed)
    return data


def prepare_data(raw: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    raw = raw.copy()
    raw.columns = [_clean_name(c) for c in raw.columns]
    label_col = next((c for c in raw if c.lower() in LABEL_CANDIDATES), None)
    if label_col is None:
        raise ValueError("Could not find a label column (expected Label, Class, or Attack).")
    labels = raw.pop(label_col).astype(str).str.strip().str.upper()
    y = (labels != "BENIGN").astype(np.int64).to_numpy()

    drop_cols = [c for c in raw if any(part in c.lower() for part in DROP_NAME_PARTS)]
    raw = raw.drop(columns=drop_cols)
    raw = raw.apply(pd.to_numeric, errors="coerce")
    raw = raw.replace([np.inf, -np.inf], np.nan)
    raw = raw.dropna(axis=1, how="all")
    if raw.shape[1] == 0:
        raise ValueError("No usable numeric feature columns remain after cleaning.")
    if len(np.unique(y)) < 2:
        raise ValueError("The dataset must contain both BENIGN and attack records.")
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=test_size, random_state=seed, stratify=y
    )
    train_frame, test_frame = raw.iloc[train_idx].copy(), raw.iloc[test_idx].copy()
    medians = train_frame.median().fillna(0)
    train_frame = train_frame.fillna(medians)
    test_frame = test_frame.fillna(medians)
    x_train, x_test = train_frame.to_numpy(dtype=np.float64), test_frame.to_numpy(dtype=np.float64)
    y_train, y_test = y[train_idx], y[test_idx]
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test, list(raw.columns)
