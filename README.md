# Graph WaveNet — Baseline Implementation

Re-implementation of the baseline from:

> **Graph WaveNet for Deep Spatial-Temporal Graph Modeling**  
> Zonghan Wu, Shirui Pan, Guodong Long, Jing Jiang, Chengqi Zhang  
> IJCAI 2019 · [arXiv:1906.00121](https://arxiv.org/abs/1906.00121)

Official code: https://github.com/nnzhan/Graph-WaveNet

---

## Overview

Graph WaveNet combines a **WaveNet-style gated TCN** for temporal modelling with a **diffusion graph convolution** for spatial modelling. Its key novelty is a **self-adaptive adjacency matrix** learned end-to-end from data, requiring no pre-defined graph structure.

![Model Architecture](https://github.com/nnzhan/Graph-WaveNet/raw/master/fig/model.png)

---

## Project Structure

```
.
├── AGENT.md                     # AI-agent cheat-sheet (read first)
├── README.md                    # this file
├── baseline_implementation.md   # 4-phase project plan
├── requirements.txt
├── .gitignore
├── generate_training_data.py    # data preprocessing
├── src/
│   ├── model.py                 # gwnet, gcn, nconv, linear
│   ├── util.py                  # data loaders, metrics, adj helpers
│   └── engine.py                # trainer (train/eval loops)
├── train.py                     # training entry point
├── test.py                      # evaluation entry point
└── configs/
    ├── metr_la.yaml
    └── pems_bay.yaml
```

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

# METR-LA
python generate_training_data.py \
    --output_dir=data/METR-LA \
    --traffic_df_filename=data/metr-la.h5

# PEMS-BAY
python generate_training_data.py \
    --output_dir=data/PEMS-BAY \
    --traffic_df_filename=data/pems-bay.h5
```

Split: **70 / 10 / 20** chronological. Input/output: **12 steps** each.

---

## Training

```bash
# Full model — doubletransition adj + adaptive adj (random init)
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

| Flag | Meaning |
|---|---|
| `--gcn_bool` | Enable graph convolution |
| `--adjtype` | `doubletransition` (forward + backward), `transition`, `symnadj`, … |
| `--addaptadj` | Add learnable self-adaptive adjacency matrix |
| `--randomadj` | Randomly initialise node embeddings (vs. SVD-seeded) |
| `--aptonly` | Use **only** adaptive adj (ablation) |

---

## Evaluation

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

## Implementation Plan

See [`baseline_implementation.md`](baseline_implementation.md) for the full
4-phase implementation timeline, technical specifications, and expert evaluation.

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
