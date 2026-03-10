> **Senior AI Expert Review & Adjusted Timeline**
>
> The original 4-phase plan is structurally sound but contains several inaccuracies relative to the paper and the official implementation. Corrections and additions are highlighted in **bold** below each table.

---

## Phase 1: Environment & Data Foundation (Days 1-3)

**Goal:** Prepare the computational environment and the METR-LA or PEMS-BAY datasets.

| Task | Assignee | Duration | Expected Outcome |
| ----- | ----- | ----- | ----- |
| **Environment Setup** | A | 1 Day | GPU-enabled environment (Python 3, PyTorch >= 1.0, scipy, pandas, matplotlib) ready for training. |
| **Data Acquisition** | B | 1 Day | Raw METR-LA (`metr-la.h5`) and PEMS-BAY (`pems-bay.h5`) downloaded from the DCRNN Google Drive link, along with their pre-built sensor-graph pickle files (`adj_mx.pkl`). |
| **Preprocessing Pipeline** | B | 2 Days | `generate_training_data.py` produces `train.npz / val.npz / test.npz` with a **70 / 10 / 20** chronological split, 12-step input windows, 12-step output windows, and a **time-of-day** auxiliary feature appended to each observation (input dim = 2). Z-score normalisation is applied to the traffic channel only, using training-set statistics. |

**Corrections vs. original plan:**
- The adjacency matrix is **not** constructed here; it is pre-built by DCRNN and loaded as a pickle file (`adj_mx.pkl`). No re-construction step is needed.
- The "5-minute windows" description refers to the raw sampling rate of METR-LA, not a windowing decision; the model uses 12 past steps -> 12 future steps (1 hour each way at 5-min resolution).
- The split must be **chronological** (no shuffling before splitting) to avoid data leakage.
- Input dimensionality is **2** (speed + time-of-day), not 1.

---

## Phase 2: Core Module Development (Days 4-7)

**Goal:** Build the spatial and temporal building blocks of Graph WaveNet.

| Task | Assignee | Duration | Expected Outcome |
| ----- | ----- | ----- | ----- |
| **Gated TCN Implementation** | D | 2 Days | Dilated causal convolution with a tanh-gate x sigmoid-gate (WaveNet-style gating). With `blocks=4` and `layers=2` per block the dilation sequence resets to 1 at each block: **(1, 2) x 4 = (1,2,1,2,1,2,1,2)**, giving a receptive field of **13** time steps (>= 12 required). Kernel size is fixed at **2**. |
| **Self-Adaptive Adjacency Matrix** | C | 2 Days | Two learnable node-embedding matrices **E1 in R^{Nx10}** and **E2 in R^{10xN}** (10-dim embeddings). Adaptive matrix: **A_adp = SoftMax(ReLU(E1 * E2))**. When `--randomadj` is set, E1 and E2 are randomly initialised; otherwise they are seeded from the SVD of the pre-built adj. |
| **Diffusion Graph Convolution** | C | 3 Days | `gcn` module supporting up to 3 transition matrices (forward D_O^{-1}A, backward D_I^{-1}A^T, adaptive A_adp). Diffusion order **K=2** (`order=2`). Output dimension equals `residual_channels` (**32**). The feature vector fed into `1x1 conv` has size `(K x support_len + 1) x C_in`. |

**Corrections vs. original plan:**
- Dilation factors are **(1,2,1,2,1,2,1,2)** -- confirmed correct -- but the plan called them "exponentially growing"; they reset at each block, so "block-periodic exponential" is more precise.
- The node embeddings are **10-dimensional**, not arbitrary-size "dictionaries".
- The GCN **concatenates** the 0th-order (identity), K=1, and K=2 diffusion features before the `1x1` conv -- this detail is critical for matching the parameter count.
- Duration for Diffusion GCN bumped from 2 -> 3 days to allow for correct K-step concatenation and multi-support handling.

---

## Phase 3: Architecture Assembly & Training (Days 8-11)

**Goal:** Integrate modules into the full Graph WaveNet framework and begin training.

| Task | Assignee | Duration | Expected Outcome |
| ----- | ----- | ----- | ----- |
| **Framework Integration** | A, C, D | 2 Days | `gwnet` class: start `1x1 Conv` (in_dim->32) -> 8 ST-layers (each: Gated-TCN -> GCN -> residual add -> BN) -> skip-sum -> ReLU -> `1x1 Conv` (256->512) -> `1x1 Conv` (512->12). `skip_channels=256`, `end_channels=512`, `residual_channels=dilation_channels=32`. |
| **Training Loop Setup** | A | 1 Day | Adam optimiser (`lr=1e-3`, `weight_decay=1e-4`), **masked MAE** loss (zeros masked), gradient clipping at norm **5**, `batch_size=64`, `epochs=100`. Input is zero-padded by **1** on the left before the start conv to keep temporal alignment. |
| **Model Training & Tuning** | A | 1 Day | Training run with `--gcn_bool --adjtype doubletransition --addaptadj --randomadj`. Evaluate per-horizon MAE/MAPE/RMSE at steps 3 (15 min), 6 (30 min), 12 (60 min). Best checkpoint selected by minimum validation MAE. |

**Corrections vs. original plan:**
- The plan omitted the **masked MAE** loss detail (zero-valued observations are excluded from the loss). This is critical to match paper results.
- `skip_channels` and `end_channels` are **256** and **512** (= nhid x 8 and nhid x 16 with nhid=32), not arbitrary.
- Gradient clipping (norm=5) and the **+1 left-pad** of the input sequence are implementation-critical and were absent from the plan.
- Evaluation horizons should be reported at steps **3, 6, and 12** (not only 15/30/60 min labels).

---

## Phase 4: Baseline Simulation & Validation (Days 12-14)

**Goal:** Compare results against standard baselines to verify methodology effectiveness.

| Task | Assignee | Duration | Expected Outcome |
| ----- | ----- | ----- | ----- |
| **Baseline Benchmarking** | B | 2 Days | Comparative metrics (MAE, RMSE, MAPE) for ARIMA, FC-LSTM, and DCRNN on the same METR-LA/PEMS-BAY splits (use published numbers from Table 1 of the paper; no re-training needed). |
| **Ablation Study (Adaptive Matrix)** | C | 1 Day | Re-run training with `--aptonly` (adaptive adj only, no pre-built graph) and without `--addaptadj` (pre-built graph only) to isolate the contribution of A_adp. |
| **Final Evaluation & Reporting** | A, B, C, D | 1 Day | Final performance table at horizons 15/30/60 min comparing GWN variants to baselines, aiming to match Table 1 in the paper (METR-LA: MAE~2.69 @ 3-step, PEMS-BAY: MAE~1.45 @ 3-step). |

**Corrections vs. original plan:**
- "Standard WaveNet" is not a canonical baseline in the paper; use **DCRNN, STGCN, ASTGCN, FC-LSTM** and **ARIMA** as per Table 1.
- Ablation should test three configurations: **(a) w/o adaptive adj**, **(b) adaptive adj only (--aptonly)**, **(c) full model**, not just two.

---

## Key Technical Specifications (Verified Against Paper & Official Code)

| Parameter | Value | Source |
| ----- | ----- | ----- |
| Input length / Output length | 12 / 12 | Paper Sec 4.1 |
| Input feature dim | 2 (speed + time-of-day) | generate_training_data.py |
| Blocks x Layers | 4 x 2 = 8 ST-layers | model.py |
| Dilation sequence (per block) | 1 -> 2 (resets each block) | model.py |
| Kernel size | 2 | model.py |
| Receptive field | 13 time steps | Derived |
| residual / dilation channels | 32 | engine.py (nhid=32) |
| Skip channels | 256 (nhid x 8) | engine.py |
| End channels | 512 (nhid x 16) | engine.py |
| Diffusion order K | 2 | model.py (order=2) |
| Node embedding dim | 10 | model.py |
| Adj type (default) | doubletransition (forward + backward) | train.py |
| Optimiser | Adam, lr=1e-3, wd=1e-4 | train.py |
| Loss | Masked MAE (null=0.0) | engine.py |
| Dropout | 0.3 | train.py |
| Batch size | 64 | train.py |
| Epochs | 100 | train.py |
| Gradient clip norm | 5 | engine.py |
| Train / Val / Test split | 70 / 10 / 20 (chronological) | generate_training_data.py |
