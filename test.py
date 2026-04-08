import torch
import numpy as np
import argparse
import time
import random
import json
import os

from src import util
from src.engine import trainer

parser = argparse.ArgumentParser(description="Evaluate a trained Graph WaveNet checkpoint")
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
parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint file")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--eval_horizons", type=str, default="3,6,12", help="Comma-separated horizons for report")
parser.add_argument("--metrics_out", type=str, default="", help="Optional JSON path for saving evaluation summary")
parser.add_argument("--spatial_attention", action="store_true", help="Enable dynamic spatial attention support")
parser.add_argument("--temporal_attention", action="store_true", help="Enable temporal self-attention branch")
parser.add_argument(
    "--model_variant",
    type=str,
    default="",
    choices=["", "baseline", "spatial", "temporal", "spatiotemporal"],
    help="Optional shorthand to configure attention toggles for ablation",
)
parser.add_argument("--temporal_attention_heads", type=int, default=4, help="Temporal attention heads")
parser.add_argument("--temporal_attention_dropout", type=float, default=0.0, help="Temporal attention dropout")
parser.add_argument("--no_temporal_causal_mask", action="store_true", help="Disable causal mask in temporal attention")

args = parser.parse_args()


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if args.model_variant:
        args.spatial_attention = args.model_variant in ("spatial", "spatiotemporal")
        args.temporal_attention = args.model_variant in ("temporal", "spatiotemporal")

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
        args.addaptadj,
        adjinit,
        spatial_attention=args.spatial_attention,
        temporal_attention=args.temporal_attention,
        temporal_attention_heads=args.temporal_attention_heads,
        temporal_attention_dropout=args.temporal_attention_dropout,
        temporal_attention_causal=not args.no_temporal_causal_mask,
    )
    engine.model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    engine.model.eval()
    print(f"Loaded checkpoint: {args.checkpoint}")

    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device).transpose(1, 3)[:, 0, :, :]

    t1 = time.time()
    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(
                torch.nn.functional.pad(testx, (1, 0, 0, 0))
            ).transpose(1, 3)
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
        print(
            f"Horizon {i + 1:02d} | MAE: {metrics[0]:.4f}  MAPE: {metrics[1]:.4f}  RMSE: {metrics[2]:.4f}"
        )
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        per_horizon.append(
            {
                "horizon": i + 1,
                "mae": float(metrics[0]),
                "mape": float(metrics[1]),
                "rmse": float(metrics[2]),
            }
        )

    print(
        f"\nAverage over 12 horizons | "
        f"MAE: {np.mean(amae):.4f}  MAPE: {np.mean(amape):.4f}  RMSE: {np.mean(armse):.4f}"
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

    horizon_slice = [per_horizon[h - 1] for h in selected_horizons]
    summary = {
        "seed": args.seed,
        "checkpoint": args.checkpoint,
        "model_variant": (
            "spatiotemporal_attn"
            if args.spatial_attention and args.temporal_attention
            else "spatial_attn"
            if args.spatial_attention
            else "temporal_attn"
            if args.temporal_attention
            else "baseline"
        ),
        "inference_seconds_total": float(t2 - t1),
        "average_metrics": {
            "mae": float(np.mean(amae)),
            "mape": float(np.mean(amape)),
            "rmse": float(np.mean(armse)),
        },
        "selected_horizons": horizon_slice,
        "all_horizons": per_horizon,
    }
    print(
        "Selected horizons summary: "
        + ", ".join(
            [
                f"h{item['horizon']}: MAE {item['mae']:.4f}, MAPE {item['mape']:.4f}, RMSE {item['rmse']:.4f}"
                for item in horizon_slice
            ]
        )
    )
    if args.metrics_out:
        metrics_dir = os.path.dirname(args.metrics_out)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(args.metrics_out, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved evaluation summary to: {args.metrics_out}")


if __name__ == "__main__":
    main()
