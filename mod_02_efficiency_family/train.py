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
    from mod_02_efficiency_family.engine import trainer


parser = argparse.ArgumentParser(description='Train efficiency-oriented Graph WaveNet variants')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--data', type=str, default='data/METR-LA')
parser.add_argument('--adjdata', type=str, default='data/sensor_graph/adj_mx.pkl')
parser.add_argument('--adjtype', type=str, default='doubletransition')
parser.add_argument('--gcn_bool', action='store_true')
parser.add_argument('--aptonly', action='store_true')
parser.add_argument('--addaptadj', action='store_true')
parser.add_argument('--randomadj', action='store_true')
parser.add_argument('--seq_length', type=int, default=12)
parser.add_argument('--nhid', type=int, default=32)
parser.add_argument('--in_dim', type=int, default=2)
parser.add_argument('--num_nodes', type=int, default=207)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.3)
parser.add_argument('--weight_decay', type=float, default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--save', type=str, default='./garage/mod02')
parser.add_argument('--expid', type=int, default=1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--eval_horizons', type=str, default='3,6,12')
parser.add_argument('--metrics_out', type=str, default='')
parser.add_argument('--early_stopping_patience', type=int, default=10)
parser.add_argument('--resume', type=str, default='')
parser.add_argument('--emb_dim', type=int, default=4)
parser.add_argument('--topk', type=int, default=10)
parser.add_argument('--adj_update_freq', type=int, default=5)
parser.add_argument('--attn_lr_multiplier', type=float, default=4.0)
parser.add_argument('--adj_lr_multiplier', type=float, default=0.5)
parser.add_argument(
    '--model_variant',
    type=str,
    default='static_adj_opt',
    choices=['static_adj_opt', 'attn_skipagg_opt'],
)
args = parser.parse_args()


def resolve_variant_flags(args):
    if args.model_variant == 'static_adj_opt':
        return {
            'use_static_adaptive_optimizations': True,
            'use_causal_window_tcn': False,
            'use_skip_aggregation_attention': False,
            'use_per_module_lr': False,
        }
    return {
        'use_static_adaptive_optimizations': False,
        'use_causal_window_tcn': True,
        'use_skip_aggregation_attention': True,
        'use_per_module_lr': True,
    }


def main():
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    variant = resolve_variant_flags(args)
    if not args.addaptadj:
        raise ValueError('Both mod_02 efficiency variants require --addaptadj.')

    device = torch.device(args.device)
    _, _, adj_mx = util.load_adj(args.adjdata, args.adjtype)
    dataloader = util.load_dataset(args.data, args.batch_size, args.batch_size, args.batch_size)
    scaler = dataloader['scaler']
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
        emb_dim=args.emb_dim,
        topk=args.topk,
        adj_update_freq=args.adj_update_freq,
        attn_lr_multiplier=args.attn_lr_multiplier,
        adj_lr_multiplier=args.adj_lr_multiplier,
        **variant,
    )

    best_ckpt_path = f'{args.save}_best.pth'
    save_dir = os.path.dirname(best_ckpt_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    his_loss = []
    val_time = []
    train_time = []
    best_val_loss = float('inf')
    best_epoch = 0
    patience_counter = 0
    early_stopped = False
    start_epoch = 1

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        engine.model.load_state_dict(ckpt['model_state_dict'])
        engine.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt['best_val_loss']
        best_epoch = ckpt['best_epoch']
        patience_counter = ckpt.get('patience_counter', 0)
        his_loss = ckpt.get('his_loss', [])

    print(args)
    print(f'Resolved variant={args.model_variant} | {variant}', flush=True)
    engine.print_optimizer_info()
    print('start training...', flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_mape, train_rmse = [], [], []
        t1 = time.time()
        dataloader['train_loader'].shuffle()
        for iter, (x, y) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device).transpose(1, 3)
            trainy = torch.Tensor(y).to(device).transpose(1, 3)
            metrics = engine.train(trainx, trainy[:, 0, :, :])
            train_loss.append(metrics[0])
            train_mape.append(metrics[1])
            train_rmse.append(metrics[2])
            if iter % args.print_every == 0:
                print(
                    f'Iter: {iter:03d}, Train Loss: {train_loss[-1]:.4f}, '
                    f'Train MAPE: {train_mape[-1]:.4f}, Train RMSE: {train_rmse[-1]:.4f}',
                    flush=True,
                )
        t2 = time.time()
        train_time.append(t2 - t1)

        valid_loss, valid_mape, valid_rmse = [], [], []
        s1 = time.time()
        for x, y in dataloader['val_loader'].get_iterator():
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
                    'epoch': epoch,
                    'model_state_dict': engine.model.state_dict(),
                    'optimizer_state_dict': engine.optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'best_epoch': best_epoch,
                    'patience_counter': patience_counter,
                    'his_loss': his_loss,
                },
                best_ckpt_path,
            )
            tag = ' [NEW BEST — saved]'
        else:
            patience_counter += 1
            tag = (
                f' (no improvement {patience_counter}/{args.early_stopping_patience})'
                if args.early_stopping_patience > 0
                else ''
            )

        print(
            f'Epoch: {epoch:03d}, Train Loss: {mtrain_loss:.4f}, Train MAPE: {np.mean(train_mape):.4f}, '
            f'Train RMSE: {np.mean(train_rmse):.4f}, Valid Loss: {mvalid_loss:.4f}, '
            f'Valid MAPE: {np.mean(valid_mape):.4f}, Valid RMSE: {np.mean(valid_rmse):.4f}, '
            f'Training Time: {t2 - t1:.4f}/epoch{tag}',
            flush=True,
        )

        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            early_stopped = True
            print(
                f'Early stopping triggered at epoch {epoch}. No improvement in val loss for '
                f'{args.early_stopping_patience} consecutive epochs.',
                flush=True,
            )
            break

    print(f'Average Training Time: {np.mean(train_time):.4f} secs/epoch')
    print(f'Average Inference Time: {np.mean(val_time):.4f} secs')
    if early_stopped:
        print(f'Training stopped early at epoch {epoch} of {args.epochs}.', flush=True)

    best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
    engine.model.load_state_dict(best_ckpt['model_state_dict'])
    print(f'Training finished. Best validation loss: {best_val_loss:.4f} (epoch {best_epoch})')

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device).transpose(1, 3)[:, 0, :, :]
    for x, y in dataloader['test_loader'].get_iterator():
        testx = torch.Tensor(x).to(device).transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(torch.nn.functional.pad(testx, (1, 0, 0, 0))).transpose(1, 3)
        outputs.append(preds.squeeze())
    yhat = torch.cat(outputs, dim=0)[: realy.size(0), ...]

    amae, amape, armse = [], [], []
    per_horizon = []
    for i in range(12):
        pred = scaler.inverse_transform(yhat[:, :, i])
        real = realy[:, :, i]
        metrics = util.metric(pred, real)
        print(
            f'Evaluate best model on test data for horizon {i + 1:02d}, '
            f'Test MAE: {metrics[0]:.4f}, Test MAPE: {metrics[1]:.4f}, Test RMSE: {metrics[2]:.4f}'
        )
        amae.append(metrics[0])
        amape.append(metrics[1])
        armse.append(metrics[2])
        per_horizon.append({'horizon': i + 1, 'mae': float(metrics[0]), 'mape': float(metrics[1]), 'rmse': float(metrics[2])})

    print(f'On average over 12 horizons, Test MAE: {np.mean(amae):.4f}, Test MAPE: {np.mean(amape):.4f}, Test RMSE: {np.mean(armse):.4f}')
    torch.save(engine.model.state_dict(), f'{args.save}_exp{args.expid}_best_{round(best_val_loss, 2):.2f}.pth')

    selected_horizons = []
    for token in args.eval_horizons.split(','):
        token = token.strip()
        if token:
            idx = int(token)
            if 1 <= idx <= 12:
                selected_horizons.append(idx)
    if not selected_horizons:
        selected_horizons = [3, 6, 12]

    summary = {
        'seed': args.seed,
        'model_variant': args.model_variant,
        'best_val_mae': float(best_val_loss),
        'average_metrics': {
            'mae': float(np.mean(amae)),
            'mape': float(np.mean(amape)),
            'rmse': float(np.mean(armse)),
        },
        'selected_horizons': [per_horizon[h - 1] for h in selected_horizons],
        'latency': {
            'train_seconds_per_epoch': float(np.mean(train_time)),
            'val_seconds_per_epoch': float(np.mean(val_time)),
        },
        'all_horizons': per_horizon,
    }
    if args.metrics_out:
        metrics_dir = os.path.dirname(args.metrics_out)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        with open(args.metrics_out, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2)
        print(f'Saved metrics summary to: {args.metrics_out}')


if __name__ == '__main__':
    t1 = time.time()
    main()
    t2 = time.time()
    print(f'Total time spent: {t2 - t1:.4f}')
