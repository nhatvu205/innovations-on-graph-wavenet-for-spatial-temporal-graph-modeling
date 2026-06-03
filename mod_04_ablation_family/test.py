import argparse
import json
import os
import random
import time

import numpy as np
import torch

from shared import util
try:
    from .engine import trainer
except ImportError:
    from mod_04_ablation_family.engine import trainer


parser = argparse.ArgumentParser(description="Evaluate notebook-derived Graph WaveNet ablation family")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--data", type=str, default="data/METR-LA")
parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl")
parser.add_argument("--adjtype", type=str, default="doubletransition")
parser.add_argument("--gcn_bool", action="store_true")
parser.add_argument("--aptonly", action="store_true")
parser.add_argument("--addaptadj", action="store_true")
parser.add_argument("--randomadj", action="store_true")
parser.add_argument("--seq_length", type=int, default=12)
parser.add_argument("--nhid", type=int, default=32)
parser.add_argument("--in_dim", type=int, default=2)
parser.add_argument("--num_nodes", type=int, default=207)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--weight_decay", type=float, default=0.0001)
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--eval_horizons", type=str, default="3,6,12")
parser.add_argument("--metrics_out", type=str, default="")
parser.add_argument(
    "--model_variant",
    type=str,
    default="full",
    choices=["full", "wo_adaptive", "wo_attention", "wo_dynamic", "wo_st_attention"],
)
args = parser.parse_args()


def resolve_variant_flags(args):
    if args.model_variant == "full":
        return {
            "variant_addaptadj": True,
            "use_dynamic_adaptive_adj": True,
            "use_skip_attention": True,
            "use_skip_aggregation_attention": False,
            "use_static_adaptive_optimizations": False,
        }
    if args.model_variant == "wo_adaptive":
        return {
            "variant_addaptadj": False,
            "use_dynamic_adaptive_adj": False,
            "use_skip_attention": False,
            "use_skip_aggregation_attention": False,
            "use_static_adaptive_optimizations": False,
        }
    if args.model_variant == "wo_dynamic":
        return {
            "variant_addaptadj": True,
            "use_dynamic_adaptive_adj": False,
            "use_skip_attention": False,
            "use_skip_aggregation_attention": True,
            "use_static_adaptive_optimizations": True,
        }
    if args.model_variant in ("wo_attention", "wo_st_attention"):
        return {
            "variant_addaptadj": True,
            "use_dynamic_adaptive_adj": True,
            "use_skip_attention": False,
            "use_skip_aggregation_attention": False,
            "use_static_adaptive_optimizations": False,
        }
    raise ValueError(f"Unknown variant: {args.model_variant}")


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    variant = resolve_variant_flags(args)
    variant_addaptadj = variant["variant_addaptadj"]
    use_dynamic_adaptive_adj = variant["use_dynamic_adaptive_adj"]
    use_skip_attention = variant["use_skip_attention"]
    use_skip_aggregation_attention = variant["use_skip_aggregation_attention"]
    use_static_adaptive_optimizations = variant["use_static_adaptive_optimizations"]
    if not args.addaptadj and variant_addaptadj:
        raise ValueError("Variant requires --addaptadj, but it was not provided.")
    if args.model_variant == "wo_adaptive" and args.addaptadj:
        raise ValueError("wo_adaptive variant must run without --addaptadj.")

    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    adjinit = None if args.randomadj else supports[0]
    if args.aptonly:
        supports = None

    engine = trainer(
        scaler,
        args.in_dim,
        args.seq_length,
        args.num_nodes,
        args.nhid,
        args.dropout,
        args.learning_rate,
        args.weight_decay,
        device,
        supports,
        args.gcn_bool,
        variant_addaptadj,
        adjinit,
        use_dynamic_adaptive_adj=use_dynamic_adaptive_adj,
        use_skip_attention=use_skip_attention,
        use_skip_aggregation_attention=use_skip_aggregation_attention,
        use_static_adaptive_optimizations=use_static_adaptive_optimizations,
    )
    engine.model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=False))
    engine.model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(
        f"Resolved variant={args.model_variant} | addaptadj={variant_addaptadj} | "
        f"dynamic_adaptive_adj={use_dynamic_adaptive_adj} | local_skip_attention={use_skip_attention} | "
        f"skip_aggregation_attention={use_skip_aggregation_attention} | static_adaptive_opt={use_static_adaptive_optimizations}",
        flush=True,
    )

    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device).transpose(1, 3)[:, 0, :, :]

    t1 = time.time()
    for x, y in dataloader["test_loader"].get_iterator():
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(torch.nn.functional.pad(testx, (1, 0, 0, 0))).transpose(1, 3)
        outputs.append(preds.squeeze())
    t2 = time.time()
    print(f"Inference time: {t2 - t1:.4f} secs")

    yhat = torch.cat(outputs, dim=0)[: realy.size(0), ...]

    amae, amape, armse = [], [], []
    per_horizon = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        print(f"Horizon {i + 1:02d} | MAE: {metrics[0]:.4f}  MAPE: {metrics[1]:.4f}  RMSE: {metrics[2]:.4f}")
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        per_horizon.append({"horizon": i + 1, "mae": float(metrics[0]), "mape": float(metrics[1]), "rmse": float(metrics[2])})

    print(
        f"\nAverage over 12 horizons | MAE: {np.mean(amae):.4f}  "
        f"MAPE: {np.mean(amape):.4f}  RMSE: {np.mean(armse):.4f}"
    )

    selected_horizons = []
    for token in args.eval_horizons.split(","):
        token = token.strip()
        if token:
            idx = int(token)
            if 1 <= idx <= 12:
                selected_horizons.append(idx)
    if not selected_horizons:
        selected_horizons = [3, 6, 12]

    summary = {
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "model_variant": args.model_variant,
        "inference_seconds_total": float(t2 - t1),
        "average_metrics": {
            "mae": float(np.mean(amae)),
            "mape": float(np.mean(amape)),
            "rmse": float(np.mean(armse)),
        },
        "selected_horizons": [per_horizon[h - 1] for h in selected_horizons],
        "all_horizons": per_horizon,
    }
    if args.metrics_out:
        metrics_dir = os.path.dirname(args.metrics_out)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved evaluation summary to: {args.metrics_out}")


if __name__ == "__main__":
    main()
