# Federated Learning for Network Intrusion Detection

Reproducible, CPU-friendly research starter comparing a centralized baseline with a simulated FedAvg system on CICIDS2017 flow records. The code expects the user to obtain the public CSV files and does not claim experimental results in advance.

## What it does

- Loads and cleans CICFlowMeter CSV files, normalizes column names, removes identifiers and invalid values, and maps attack labels to binary `BENIGN` / `ATTACK` targets.
- Builds reproducible non-IID clients by allocating attack classes with a seeded Dirichlet partition (default: 5 clients).
- Trains a centralized `SGDClassifier` baseline and a synchronous FedAvg simulation using the same scaled feature space and classifier family.
- Reports accuracy, precision, recall, F1, and estimated communication bytes (client uploads plus server broadcasts).
- Saves metric tables, per-round history, and comparison plots.

This is a simulation: clients are partitions of one local dataset, and communication cost is estimated from serialized model parameter sizes. It is not a deployment-grade secure FL system.

## Dataset

Use the public **CICIDS2017** dataset from the Canadian Institute for Cybersecurity: <https://www.unb.ca/cic/datasets/ids-2017.html>. Download the machine-learning flow CSV files from that page and place the extracted CSVs in `data/raw/`. Do not commit dataset files to Git; the dataset is large and has its own terms of use. For a quick run, use a small subset of CSVs or rows.

Example layout:

```text
data/raw/Monday-WorkingHours.pcap_ISCX.csv
data/raw/Tuesday-WorkingHours.pcap_ISCX.csv
```

The script reads all `*.csv` files in that folder. CICIDS2017 labels are used as provided, with known header whitespace handled automatically.

## Setup and reproducibility

Python 3.9 or newer is supported. On macOS, check with `python3 --version`; if needed, install a current Python 3 release from python.org or Homebrew.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python3 -m pip install -r requirements.txt
python3 -m pytest -q
```

Run with defaults (5 clients, 5 rounds, seed 42):

```bash
python3 -m src.run_experiment --data-dir data/raw --output-dir outputs
```

Options:

```bash
python3 -m src.run_experiment --help
python3 -m src.run_experiment --data-dir data/raw --clients 3 --rounds 8 --alpha 0.2 --seed 17
```

`--max-rows` is available for smoke runs and samples after CSV loading. For final research, record the full dataset files used, their checksums, command, software versions, and machine details. All random operations use the supplied seed. The scaler is fitted on training data only; test data is held out before client partitioning.

## Outputs

The run writes `metrics.csv`, `round_history.csv`, and `metrics_comparison.png` under the output folder. Communication is reported in bytes and MiB. In this simulation, each round counts one model upload from every client and one broadcast to every client, with each parameter represented as float64; it excludes protocol headers, compression, optimizer state, and network retransmission.

## Results status

**Results pending final validation.** No dataset is included with this repository. A local run completed, but scikit-learn emitted numerical warnings during model computation, so its scores are not published here as final findings. Rerun after resolving numerical stability before citing or sharing performance claims.

## Project showcase

The static showcase lives in `docs/` and is published by the GitHub Pages workflow in `.github/workflows/pages.yml` whenever `main` is updated. After creating the repository, set **Settings → Pages → Build and deployment → Source** to **GitHub Actions** once. Later pushes to `main` deploy the updated showcase.

## Research summary for supervisor

This project proposes a controlled comparison of centralized machine-learning training and federated averaging for binary network intrusion detection using CICIDS2017 flow records. The data preparation removes identifying/non-numeric fields, imputes invalid numeric values, standardizes features using training data, and defines client partitions with a seeded Dirichlet label skew to model non-IID data. A centralized SGD classifier provides the reference; a synchronous FedAvg simulation trains the same classifier locally and averages model parameters. Evaluation uses accuracy, precision, recall, and F1 on a fixed holdout set, alongside estimated model communication per round and in total. At present, results are pending dataset acquisition and execution. The work is a reproducible baseline simulation and does not yet establish privacy guarantees, robustness to malicious clients, or real network deployment performance.

## Repository structure

```text
src/    data loading, client partitioning, training, metrics, plots, CLI
tests/  focused unit tests for partitioning and aggregation
data/raw/   place downloaded CSV files here (not committed)
outputs/    generated experiment artifacts (not committed)
```

## License and citation

Check CIC's dataset terms and cite the CICIDS2017 dataset paper/page in any report. Code in this repository is provided under the MIT License; see `LICENSE`.
