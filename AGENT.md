# AGENT.md — Repository Cheat-Sheet for AI Agents

> Read this file first. It gives you everything you need to understand the codebase
> without having to parse every source file.

---

## 1. What This Repo Does

Baseline re-implementation of **Graph WaveNet** (Wu et al., IJCAI 2019) for
deep spatial-temporal graph forecasting on traffic datasets (METR-LA, PEMS-BAY).

Paper: *Graph WaveNet for Deep Spatial-Temporal Graph Modeling*
arXiv: https://arxiv.org/abs/1906.00121
Official repo: https://github.com/nnzhan/Graph-WaveNet

---

## 2. Repo Layout

```
.
├── AGENT.md                     # ← you are here
├── README.md                    # human-readable project overview
├── baseline_implementation.md   # 4-phase project plan (evaluated & corrected)
├── requirements.txt             # pip dependencies
├── .gitignore
│
├── generate_training_data.py    # STEP 1 – raw HDF5 -> train/val/test .npz
│
├── src/
│   ├── __init__.py
│   ├── model.py                 # gwnet (full model), gcn, nconv, linear
│   ├── util.py                  # DataLoader, StandardScaler, adj helpers, metrics
│   └── engine.py                # trainer class (train / eval loops)
│
├── train.py                     # STEP 2 – main training entry point
├── test.py                      # STEP 3 – load checkpoint, compute test metrics
│
├── configs/
│   ├── metr_la.yaml             # hyperparameters for METR-LA
│   └── pems_bay.yaml            # hyperparameters for PEMS-BAY
│
├── data/                        # NOT committed – created locally
│   ├── metr-la.h5               #   raw traffic readings
│   ├── pems-bay.h5
│   ├── sensor_graph/
│   │   └── adj_mx.pkl           #   pre-built adjacency matrix
│   ├── METR-LA/                 #   output of generate_training_data.py
│   │   ├── train.npz
│   │   ├── val.npz
│   │   └── test.npz
│   └── PEMS-BAY/
│
└── garage/                      # NOT committed – model checkpoints saved here
```

---

## 3. Key Model Facts

| Concept | Detail |
|---|---|
| Architecture | WaveNet-style gated TCN + Diffusion GCN, stacked 8 times |
| Blocks × Layers | 4 blocks × 2 layers = 8 ST-layers |
| Dilation sequence | (1, 2) per block → (1,2,1,2,1,2,1,2) total |
| Kernel size | 2 |
| Receptive field | 13 time steps |
| Channels | residual=32, dilation=32, skip=256, end=512 |
| Gating | `x = tanh(filter_conv) * sigmoid(gate_conv)` |
| Graph conv | Diffusion GCN, K=2, supports up to 3 matrices |
| Adaptive adj | `A_adp = SoftMax(ReLU(E1 @ E2))`, E1/E2 ∈ ℝ^{N×10} |
| Input dim | 2 (traffic speed + time-of-day fraction) |
| Output dim | 12 (simultaneous multi-step prediction) |
| Loss | Masked MAE — zero observations excluded |
| Optimiser | Adam, lr=1e-3, weight_decay=1e-4 |
| Grad clip | L2 norm ≤ 5 |
| Dropout | 0.3 |
| Batch size | 64 |
| Epochs | 100 |

---

## 4. Data Pipeline (one-time setup)

```bash
# 1. Download raw data from DCRNN Google Drive
#    https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX
#    Place metr-la.h5 and pems-bay.h5 in data/
#    Place adj_mx.pkl in data/sensor_graph/

# 2. Generate train / val / test splits
mkdir -p data/METR-LA data/PEMS-BAY

python generate_training_data.py \
    --output_dir=data/METR-LA \
    --traffic_df_filename=data/metr-la.h5

python generate_training_data.py \
    --output_dir=data/PEMS-BAY \
    --traffic_df_filename=data/pems-bay.h5
```

Split ratio: **70 / 10 / 20** (chronological, no shuffle).
Normalisation: Z-score on traffic channel only, using train-set mean/std.

---

## 5. Training

```bash
# Full Graph WaveNet (doubletransition adj + adaptive adj, random init)
python train.py \
    --device cuda:0 \
    --data data/METR-LA \
    --adjdata data/sensor_graph/adj_mx.pkl \
    --adjtype doubletransition \
    --gcn_bool \
    --addaptadj \
    --randomadj \
    --num_nodes 207 \
    --save garage/metr
```

Key flags:
- `--gcn_bool` — enable graph convolution (always set for GWN)
- `--addaptadj` — add learnable adaptive adjacency matrix
- `--randomadj` — randomly initialise E1, E2 (vs. SVD-init from pre-built adj)
- `--aptonly` — use **only** adaptive adj (ablation; requires `--addaptadj`)

---

## 6. Testing / Evaluation

```bash
python test.py \
    --device cuda:0 \
    --data data/METR-LA \
    --adjdata data/sensor_graph/adj_mx.pkl \
    --adjtype doubletransition \
    --gcn_bool \
    --addaptadj \
    --randomadj \
    --num_nodes 207 \
    --checkpoint garage/metr_exp1_best_X.XX.pth
```

Reports MAE / MAPE / RMSE at each of the 12 horizons and overall average.

---

## 7. File-by-File Summary

### `src/model.py`
- `nconv` — batched graph propagation via `einsum('ncvl,vw->ncwl')`
- `linear` — `1×1` Conv2d wrapper
- `gcn(c_in, c_out, dropout, support_len, order=2)` — K-step diffusion GCN
- `gwnet(device, num_nodes, ...)` — full Graph WaveNet; key constructor args:
  `dropout, supports, gcn_bool, addaptadj, aptinit, in_dim, out_dim,
   residual_channels, dilation_channels, skip_channels, end_channels,
   kernel_size, blocks, layers`

### `src/util.py`
- `DataLoader` — mini-batch iterator with shuffle and last-sample padding
- `StandardScaler` — z-score transform / inverse
- `sym_adj`, `asym_adj`, `calculate_normalized_laplacian`, `calculate_scaled_laplacian` — adj preprocessing
- `load_adj(pkl, adjtype)` — returns list of numpy adj matrices
- `load_dataset(dir, batch_size, ...)` — loads npz files, applies scaler, returns dict
- `masked_mae`, `masked_mape`, `masked_rmse`, `masked_mse`, `metric` — evaluation

### `src/engine.py`
- `trainer(scaler, in_dim, seq_length, num_nodes, nhid, dropout, lrate, wdecay, device, supports, gcn_bool, addaptadj, aptinit)`
  - `.train(input, real_val)` — forward + backward pass, returns (mae, mape, rmse)
  - `.eval(input, real_val)` — no-grad forward pass, returns (mae, mape, rmse)
  - Note: input is left-padded by 1 before being fed to the model

### `generate_training_data.py`
- Reads `.h5` DataFrame (index=timestamps, columns=sensor IDs)
- Appends time-of-day feature; optionally day-of-week (`--dow`)
- Saves `train.npz`, `val.npz`, `test.npz` each with keys `x`, `y`,
  `x_offsets`, `y_offsets`
- Shape: `x (N, 12, num_nodes, 2)`, `y (N, 12, num_nodes, 2)`

### `train.py`
- Parses args, loads adj + dataset, builds `trainer`, runs training loop
- Saves checkpoint after every epoch; at end reloads best-val checkpoint
- Prints per-horizon test metrics for all 12 steps

### `test.py`
- Loads a saved `.pth` checkpoint and runs inference on the test set
- Prints per-horizon and average MAE/MAPE/RMSE

---

## 8. Expected Results (from Paper, Table 1)

| Dataset | Horizon | MAE | MAPE | RMSE |
|---|---|---|---|---|
| METR-LA | 15 min (3-step) | 2.69 | 6.90% | 5.15 |
| METR-LA | 30 min (6-step) | 3.07 | 8.06% | 6.22 |
| METR-LA | 60 min (12-step) | 3.53 | 9.56% | 7.37 |
| PEMS-BAY | 15 min | 1.30 | 2.73% | 2.74 |
| PEMS-BAY | 30 min | 1.63 | 3.70% | 3.67 |
| PEMS-BAY | 60 min | 2.20 | 5.19% | 4.96 |

---

## 9. Dependencies

```
matplotlib
numpy
scipy
pandas
torch          # >= 1.0, CUDA build recommended
PyYAML         # for config loading
```

Install: `pip install -r requirements.txt`

---

## 10. Common Pitfalls

1. **Wrong input padding** — `engine.py` pads input by 1 on the left (`F.pad(input, (1,0,0,0))`). If you bypass the trainer, remember this.
2. **Scaler on traffic channel only** — `x[..., 0]` is normalised; the time-of-day channel `x[..., 1]` is left as-is.
3. **Masked loss** — zeros in the target are masked out. Using plain MSE/MAE will give different (worse) results.
4. **Chronological split** — do not shuffle the dataset before splitting.
5. **`--aptonly` requires `--addaptadj`** — setting `--aptonly` without `--addaptadj` leaves `supports=None`, which disables all graph conv.
