import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import SGDClassifier

from .data import load_csv_folder, prepare_data
from .evaluation import binary_metrics, save_plot
from .federated import fedavg, new_model
from .partition import dirichlet_partition


def run(args):
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw = load_csv_folder(args.data_dir, args.max_rows, args.seed)
    x_train, x_test, y_train, y_test, features = prepare_data(raw, seed=args.seed)
    central = new_model(args.seed)
    central.fit(x_train, y_train)
    partitions = dirichlet_partition(y_train, args.clients, args.alpha, args.seed)
    federated, history, comm = fedavg(x_train, y_train, partitions, args.rounds, args.local_epochs, args.seed)

    rows = [binary_metrics("centralized", central, x_test, y_test),
            binary_metrics("fedavg", federated, x_test, y_test)]
    metrics = pd.DataFrame(rows)
    metrics["communication_bytes_total"] = [0, comm["total_bytes"]]
    metrics["communication_mib_total"] = metrics["communication_bytes_total"] / (1024 ** 2)
    metrics.to_csv(out / "metrics.csv", index=False)
    pd.DataFrame(history).assign(communication_bytes=comm["bytes_per_round"]).to_csv(out / "round_history.csv", index=False)
    save_plot(metrics, out / "metrics_comparison.png")
    print(f"Rows: {len(raw):,}; features: {len(features)}; train: {len(y_train):,}; test: {len(y_test):,}")
    print(metrics.to_string(index=False))
    print(f"Communication: {comm['bytes_per_round']:,} bytes/round; {comm['total_bytes']:,} bytes total")


def main():
    parser = argparse.ArgumentParser(description="Centralized and FedAvg CICIDS2017 binary intrusion detection")
    parser.add_argument("--data-dir", default="data/raw")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--clients", type=int, default=5, choices=range(3, 6), metavar="{3,4,5}")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--alpha", type=float, default=0.5, help="Smaller values generally increase label skew")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-rows", type=int, default=None, help="Optional seeded sample cap for smoke runs")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
