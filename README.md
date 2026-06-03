# Graph WaveNet — Improvement Experiments

Experiments on top of the original **Graph WaveNet** (Wu et al., IJCAI 2019).

> **Graph WaveNet for Deep Spatial-Temporal Graph Modeling**  
> Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Chengqi Zhang  
> IJCAI 2019 · [arXiv:1906.00121](https://arxiv.org/abs/1906.00121) · [Official code](https://github.com/nnzhan/Graph-WaveNet)

---

## Overview

Graph WaveNet combines a **WaveNet-style gated TCN** for temporal modelling with a **diffusion graph convolution** for spatial modelling. Its key novelty is a **self-adaptive adjacency matrix** learned end-to-end from data, requiring no pre-defined graph structure.

![Model Architecture](https://github.com/nnzhan/Graph-WaveNet/raw/master/fig/model.png)

---

## Repository Structure

```
.
├── shared/                          # utilities shared across all experiments
│   ├── util.py                      # DataLoader, StandardScaler, adj helpers, metrics
│   └── helper.py                    # build_transition_matrices
│
├── mod_01_modular_components/       # Modular component refactor
│   ├── DiffusionGraphConv.py        # K-step diffusion GCN (standalone class)
│   ├── GatedTCN.py                  # Gated TCN hierarchy (Layer → Block → Stack)
│   ├── SelfAdaptiveAdjacency.py     # Adaptive adjacency via nn.Embedding
│   ├── model.py                     # gwnet assembled from the above components
│   ├── engine.py                    # trainer (train/eval loops)
│   ├── train.py                     # training entry point
│   └── test.py                      # evaluation entry point
│
├── generate_training_data.py        # raw HDF5 → train/val/test .npz
├── configs/
│   ├── metr_la.yaml
│   └── pems_bay.yaml
└── docs/
```

Each `mod_XX_<name>/` folder is a self-contained experiment. To add a new improvement, create a new folder following the same layout and swap out `model.py`.

---

## Requirements

- Python 3.7+
- PyTorch >= 1.0 (CUDA build strongly recommended)

```bash
pip install -r requirements.txt
```

---

## Data Preparation

### Step 1 — Download raw data

Download **METR-LA** and **PEMS-BAY** from the
[DCRNN Google Drive](https://drive.google.com/open?id=10FOTa6HXPqX8Pf5WRoRwcFnW9BrNZEIX)
(provided by [DCRNN authors](https://github.com/liyaguang/DCRNN)).

Place the files as follows:

```
data/
├── metr-la.h5
├── pems-bay.h5
└── sensor_graph/
    └── adj_mx.pkl
```

### Step 2 — Generate train / val / test splits

```bash
mkdir -p data/METR-LA data/PEMS-BAY

python generate_training_data.py \
    --output_dir=data/METR-LA \
    --traffic_df_filename=data/metr-la.h5

python generate_training_data.py \
    --output_dir=data/PEMS-BAY \
    --traffic_df_filename=data/pems-bay.h5
```

Split: **70 / 10 / 20** chronological. Input/output: **12 steps** each.

---

## Running an Experiment

All experiments are run as modules from the **repo root** (required for relative imports):

### Training

```bash
python -m mod_01_modular_components.train \
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

### Evaluation

```bash
python -m mod_01_modular_components.test \
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

Key flags:

| Flag | Meaning |
|---|---|
| `--gcn_bool` | Enable graph convolution |
| `--adjtype` | `doubletransition` (forward + backward), `transition`, `symnadj`, … |
| `--addaptadj` | Add learnable self-adaptive adjacency matrix |
| `--randomadj` | Randomly initialise node embeddings (vs. SVD-seeded) |
| `--aptonly` | Use **only** adaptive adj (ablation) |

---

## Experiments

| Folder | Description |
|---|---|
| `mod_01_modular_components` | Refactor: each component (DiffusionGCN, GatedTCN, SelfAdaptiveAdj) as a standalone class |

---

## Expected Results (Paper, Table 1)

### METR-LA

| Horizon | MAE | MAPE | RMSE |
|---|---|---|---|
| 15 min | 2.69 | 6.90% | 5.15 |
| 30 min | 3.07 | 8.06% | 6.22 |
| 60 min | 3.53 | 9.56% | 7.37 |

### PEMS-BAY

| Horizon | MAE | MAPE | RMSE |
|---|---|---|---|
| 15 min | 1.30 | 2.73% | 2.74 |
| 30 min | 1.63 | 3.70% | 3.67 |
| 60 min | 2.20 | 5.19% | 4.96 |

---

## Citation

```bibtex
@inproceedings{wu2019graph,
  title     = {Graph WaveNet for Deep Spatial-Temporal Graph Modeling},
  author    = {Wu, Zonghan and Pan, Shirui and Long, Guodong and Jiang, Jing and Zhang, Chengqi},
  booktitle = {Proceedings of the 28th International Joint Conference on Artificial Intelligence (IJCAI)},
  year      = {2019}
}
```
