import torch
import numpy as np
import argparse
import time
import random
import json
import os

from src import util
from src.engine import trainer

parser = argparse.ArgumentParser(description="Train Graph WaveNet")
parser.add_argument("--device", type=str, default="cuda:0", help="Device (e.g. cuda:0 or cpu)")
parser.add_argument("--data", type=str, default="data/METR-LA", help="Data directory")
parser.add_argument("--adjdata", type=str, default="data/sensor_graph/adj_mx.pkl", help="Adjacency pickle path")
parser.add_argument("--adjtype", type=str, default="doubletransition", help="Adjacency type")
parser.add_argument("--gcn_bool", action="store_true", help="Enable graph convolution")
parser.add_argument("--aptonly", action="store_true", help="Use only adaptive adjacency")
parser.add_argument("--addaptadj", action="store_true", help="Add adaptive adjacency matrix")
parser.add_argument("--randomadj", action="store_true", help="Randomly initialise adaptive adj embeddings")
parser.add_argument("--seq_length", type=int, default=12, help="Output sequence length")
parser.add_argument("--nhid", type=int, default=32, help="Hidden channel width")
parser.add_argument("--in_dim", type=int, default=2, help="Input feature dimension")
parser.add_argument("--num_nodes", type=int, default=207, help="Number of graph nodes")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
parser.add_argument("--weight_decay", type=float, default=0.0001, help="Adam weight decay")
parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
parser.add_argument("--print_every", type=int, default=50, help="Log interval (iterations)")
parser.add_argument("--save", type=str, default="./garage/metr", help="Checkpoint save prefix")
parser.add_argument("--expid", type=int, default=1, help="Experiment ID")
parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
parser.add_argument("--eval_horizons", type=str, default="3,6,12", help="Comma-separated horizons for report")
parser.add_argument("--metrics_out", type=str, default="", help="Optional JSON path for saving metrics summary")
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
parser.add_argument(
    "--early_stopping_patience",
    type=int,
    default=10,
    help="Stop training if val loss does not improve for this many epochs (0 = disabled)",
)
parser.add_argument(
    "--resume",
    type=str,
    default="",
    help="Path to a full training checkpoint (_best.pth) to resume from",
)

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
    sensor_ids, sensor_id_to_ind, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader["scaler"]
    supports = [torch.tensor(i).to(device) for i in adj_mx]

    print(args)

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

    # Fixed path for the resumable checkpoint (overwritten on every new best)
    best_ckpt_path = f"{args.save}_best.pth"
    save_dir = os.path.dirname(best_ckpt_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    best_val_loss = float("inf")
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
    start_epoch = 1

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        engine.model.load_state_dict(ckpt["model_state_dict"])
        engine.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt["best_val_loss"]
        best_epoch = ckpt["best_epoch"]
        patience_counter = ckpt.get("patience_counter", 0)
        his_loss = ckpt.get("his_loss", [])
        print(
            f"Resumed from epoch {ckpt['epoch']} — "
            f"best val loss so far: {best_val_loss:.4f} (epoch {best_epoch}), "
            f"patience counter: {patience_counter}/{args.early_stopping_patience}",
            flush=True,
        )

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_mape, train_rmse = [], [], []
        t1 = time.time()
        dataloader["train_loader"].shuffle()
        for iter, (x, y) in enumerate(dataloader["train_loader"].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                print(
                    f"Iter: {iter:03d}, Train Loss: {train_loss[-1]:.4f}, "
                    f"Train MAPE: {train_mape[-1]:.4f}, Train RMSE: {train_rmse[-1]:.4f}",
                    flush=True,
                )
        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss, valid_mape, valid_rmse = [], [], []
        s1 = time.time()
        for iter, (x, y) in enumerate(dataloader["val_loader"].get_iterator()):
            valx = torch.Tensor(x).to(device).transpose(1, 3)
            valy = torch.Tensor(y).to(device).transpose(1, 3)
            metrics = engine.eval(valx, valy[:, 0, :, :])
            valid_loss.append(metrics[0])
            valid_mape.append(metrics[1])
            valid_rmse.append(metrics[2])
        s2 = time.time()
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mvalid_loss = np.mean(valid_loss)
        his_loss.append(mvalid_loss)

        improved = mvalid_loss < best_val_loss
        if improved:
            best_val_loss = mvalid_loss
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": engine.model.state_dict(),
                    "optimizer_state_dict": engine.optimizer.state_dict(),
                    "best_val_loss": best_val_loss,
                    "best_epoch": best_epoch,
                    "patience_counter": patience_counter,
                    "his_loss": his_loss,
                },
                best_ckpt_path,
            )
            tag = " [NEW BEST — saved]"
        else:
            patience_counter += 1
            tag = f" (no improvement {patience_counter}/{args.early_stopping_patience})" if args.early_stopping_patience > 0 else ""

        print(
            f"Epoch: {epoch:03d}, "
            f"Train Loss: {mtrain_loss:.4f}, Train MAPE: {np.mean(train_mape):.4f}, Train RMSE: {np.mean(train_rmse):.4f}, "
            f"Valid Loss: {mvalid_loss:.4f}, Valid MAPE: {np.mean(valid_mape):.4f}, Valid RMSE: {np.mean(valid_rmse):.4f}, "
            f"Training Time: {t2 - t1:.4f}/epoch{tag}",
            flush=True,
        )

        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            print(
                f"Early stopping triggered at epoch {epoch}. "
                f"No improvement in val loss for {args.early_stopping_patience} consecutive epochs.",
                flush=True,
            )
            early_stopped = True
            break

    print(f"Average Training Time: {np.mean(train_time):.4f} secs/epoch")
    print(f"Average Inference Time: {np.mean(val_time):.4f} secs")
    if early_stopped:
        print(f"Training stopped early at epoch {epoch} of {args.epochs}.", flush=True)

    # Reload best model weights for test evaluation
    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    engine.model.load_state_dict(best_ckpt["model_state_dict"])
    print(f"Training finished. Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})")

    outputs = []
    realy = torch.Tensor(dataloader["y_test"]).to(device).transpose(1, 3)[:, 0, :, :]
    for iter, (x, y) in enumerate(dataloader["test_loader"].get_iterator()):
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(
                torch.nn.functional.pad(testx, (1, 0, 0, 0))
            ).transpose(1, 3)
        outputs.append(preds.squeeze())

    yhat = torch.cat(outputs, dim=0)[: realy.size(0), ...]

    amae, amape, armse = [], [], []
    per_horizon = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        print(
            f"Evaluate best model on test data for horizon {i + 1:02d}, "
            f"Test MAE: {metrics[0]:.4f}, Test MAPE: {metrics[1]:.4f}, Test RMSE: {metrics[2]:.4f}"
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

    avg_training_time = float(np.mean(train_time))
    avg_inference_time = float(np.mean(val_time))
    print(
        f"On average over 12 horizons, "
        f"Test MAE: {np.mean(amae):.4f}, Test MAPE: {np.mean(amape):.4f}, Test RMSE: {np.mean(armse):.4f}"
    )
    torch.save(
        engine.model.state_dict(),
        f"{args.save}_exp{args.expid}_best_{round(best_val_loss, 2):.2f}.pth",
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
        "model_variant": (
            "spatiotemporal_attn"
            if args.spatial_attention and args.temporal_attention
            else "spatial_attn"
            if args.spatial_attention
            else "temporal_attn"
            if args.temporal_attention
            else "baseline"
        ),
        "best_val_mae": float(best_val_loss),
        "average_metrics": {
            "mae": float(np.mean(amae)),
            "mape": float(np.mean(amape)),
            "rmse": float(np.mean(armse)),
        },
        "selected_horizons": horizon_slice,
        "latency": {
            "train_seconds_per_epoch": avg_training_time,
            "val_seconds_per_epoch": avg_inference_time,
        },
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
        print(f"Saved metrics summary to: {args.metrics_out}")


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Total time spent: {t2 - t1:.4f}")
