import torch
import numpy as np
import argparse
import time

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

args = parser.parse_args()


def main():
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

    print(
        f"\nAverage over 12 horizons | "
        f"MAE: {np.mean(amae):.4f}  MAPE: {np.mean(amape):.4f}  RMSE: {np.mean(armse):.4f}"
    )


if __name__ == "__main__":
    main()
