import torch
import numpy as np
import argparse
import time

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

args = parser.parse_args()


def main():
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
    )

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []

    for epoch in range(1, args.epochs + 1):
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
        print(f"Epoch: {epoch:03d}, Inference Time: {s2 - s1:.4f} secs")
        val_time.append(s2 - s1)

        mtrain_loss = np.mean(train_loss)
        mvalid_loss = np.mean(valid_loss)
        his_loss.append(mvalid_loss)
        print(
            f"Epoch: {epoch:03d}, "
            f"Train Loss: {mtrain_loss:.4f}, Train MAPE: {np.mean(train_mape):.4f}, Train RMSE: {np.mean(train_rmse):.4f}, "
            f"Valid Loss: {mvalid_loss:.4f}, Valid MAPE: {np.mean(valid_mape):.4f}, Valid RMSE: {np.mean(valid_rmse):.4f}, "
            f"Training Time: {t2 - t1:.4f}/epoch",
            flush=True,
        )
        ckpt_path = f"{args.save}_epoch_{epoch}_{round(mvalid_loss, 2):.2f}.pth"
        torch.save(engine.model.state_dict(), ckpt_path)

    print(f"Average Training Time: {np.mean(train_time):.4f} secs/epoch")
    print(f"Average Inference Time: {np.mean(val_time):.4f} secs")

    # Reload best checkpoint and test
    bestid = np.argmin(his_loss)
    best_path = f"{args.save}_epoch_{bestid + 1}_{round(his_loss[bestid], 2):.2f}.pth"
    engine.model.load_state_dict(torch.load(best_path))
    print(f"Training finished. Best validation loss: {his_loss[bestid]:.4f} (epoch {bestid + 1})")

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

    print(
        f"On average over 12 horizons, "
        f"Test MAE: {np.mean(amae):.4f}, Test MAPE: {np.mean(amape):.4f}, Test RMSE: {np.mean(armse):.4f}"
    )
    torch.save(
        engine.model.state_dict(),
        f"{args.save}_exp{args.expid}_best_{round(his_loss[bestid], 2):.2f}.pth",
    )


if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print(f"Total time spent: {t2 - t1:.4f}")
