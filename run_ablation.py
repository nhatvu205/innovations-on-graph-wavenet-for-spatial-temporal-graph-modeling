import argparse
import itertools
import json
import os
import subprocess
import sys


def build_command(args, variant, seed):
    metrics_out = os.path.join(args.metrics_dir, f"{variant}_seed{seed}.json")
    cmd = [
        sys.executable,
        "train.py",
        "--device",
        args.device,
        "--data",
        args.data,
        "--adjdata",
        args.adjdata,
        "--adjtype",
        args.adjtype,
        "--num_nodes",
        str(args.num_nodes),
        "--batch_size",
        str(args.batch_size),
        "--epochs",
        str(args.epochs),
        "--save",
        args.save_prefix,
        "--expid",
        str(seed),
        "--seed",
        str(seed),
        "--model_variant",
        variant,
        "--metrics_out",
        metrics_out,
        "--gcn_bool",
        "--addaptadj",
        "--randomadj",
    ]
    return cmd, metrics_out


def aggregate(metrics_files, output_path):
    rows = []
    for path in metrics_files:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    grouped = {}
    for row in rows:
        key = row["model_variant"]
        grouped.setdefault(key, []).append(row["average_metrics"]["mae"])

    summary = {"variants": {}}
    for key, values in grouped.items():
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((v - mean) ** 2 for v in values) / len(values)
        summary["variants"][key] = {
            "num_runs": len(values),
            "mae_mean": mean,
            "mae_std": var ** 0.5,
        }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run baseline/spatial/temporal/spatiotemporal ablations")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--data", type=str, default="data/METR-LA")
    parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
    parser.add_argument("--adjtype", type=str, default="doubletransition")
    parser.add_argument("--num_nodes", type=int, default=207)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--save_prefix", type=str, default="garage/ablation")
    parser.add_argument("--metrics_dir", type=str, default="garage/metrics/ablation")
    parser.add_argument("--summary_out", type=str, default="garage/metrics/ablation/summary.json")
    parser.add_argument("--seeds", type=str, default="42,52,62")
    parser.add_argument(
        "--variants",
        type=str,
        default="baseline,spatial,temporal,spatiotemporal",
        help="Comma-separated set from: baseline,spatial,temporal,spatiotemporal",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print commands")
    args = parser.parse_args()

    os.makedirs(args.metrics_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.summary_out), exist_ok=True)

    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    variants = [token.strip() for token in args.variants.split(",") if token.strip()]
    metrics_files = []

    for variant, seed in itertools.product(variants, seeds):
        cmd, metrics_out = build_command(args, variant, seed)
        metrics_files.append(metrics_out)
        print(" ".join(cmd))
        if not args.dry_run:
            subprocess.run(cmd, check=True)

    if not args.dry_run:
        summary = aggregate(metrics_files, args.summary_out)
        print(json.dumps(summary, indent=2))
        print(f"Saved ablation summary to: {args.summary_out}")


if __name__ == "__main__":
    main()
