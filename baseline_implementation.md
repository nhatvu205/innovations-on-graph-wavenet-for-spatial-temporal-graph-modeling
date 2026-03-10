### **Phase 1: Environment & Data Foundation (Days 1–3)**

**Goal:** Prepare the computational environment and the METR-LA or PEMS-BAY datasets.

| Task | Assignee | Duration | Expected Outcome |
| ----- | ----- | ----- | ----- |
| **Environment Setup** | A | 1 Day | GPU-enabled environment (Python, PyTorch/TF) ready for training. |
| **Data Acquisition & Cleaning** | B | 2 Days | Cleaned METR-LA/PEMS-BAY data with 5-minute windows and 70/10/20 split. |
| **Preprocessing Pipeline** | B | 1 Day | Z-score normalization and adjacency matrix construction via thresholded Gaussian kernel. |

---

### **Phase 2: Core Module Development (Days 4–7)**

**Goal:** Build the spatial and temporal building blocks of Graph WaveNet.

| Task | Assignee | Duration | Expected Outcome |
| ----- | ----- | ----- | ----- |
| **Gated TCN Implementation** | D | 3 Days | Functional Gated TCN with dilated causal convolutions and exponentially growing receptive fields. |
| **Self-Adaptive Adjacency Matrix** | C | 2 Days | Module to learn hidden spatial dependencies via node embedding dictionaries $E\_1, E\_2$. |
| **Diffusion Graph Convolution** | C | 2 Days | GCN layer supporting forward, backward, and adaptive transition matrices. |

---

### **Phase 3: Architecture Assembly & Training (Days 8–11)**

**Goal:** Integrate modules into the full Graph WaveNet framework and begin training.

| Task | Assignee | Duration | Expected Outcome |
| ----- | ----- | ----- | ----- |
| **Framework Integration** | A, C, D | 2 Days | Stacked architecture of 8 spatial-temporal layers with residual and skip connections. |
| **Training Loop Setup** | A | 1 Day | Training script utilizing Adam optimizer, MAE loss, and dropout (p=0.3). |
| **Model Training & Tuning** | A | 1 Day | Model weights trained on 12-step forecasting horizons (15, 30, 60 mins). |

---

### **Phase 4: Baseline Simulation & Validation (Days 12–14)**

**Goal:** Compare results against standard baselines to verify methodology effectiveness.

| Task | Assignee | Duration | Expected Outcome |
| ----- | ----- | ----- | ----- |
| **Baseline Benchmarking** | B | 2 Days | Comparative metrics (MAE, RMSE, MAPE) for ARIMA or standard WaveNet on the same data. |
| **Ablation Study (Adaptive Matrix)** | C | 1 Day | Verification of performance with and without the self-adaptive adjacency matrix. |
| **Final Evaluation & Reporting** | A, B, C, D | 1 Day | Final performance table comparing Graph WaveNet to baselines, matching or exceeding state-of-the-art results. |

---

### **Key Technical Specifications for the Team:**

* **Temporal Logic:** Ensure the TCN uses dilation factors in the sequence (1, 2, 1, 2, 1, 2, 1, 2\) across the 8 layers to cover the input sequence length.  
* **Spatial Logic:** The GCN must utilize $K=2$ diffusion steps.  
* **Output Strategy:** The model should generate all $T$ future steps simultaneously rather than recursively to avoid training-inference inconsistency.

I can also create a **tailored report** summarizing the specific mathematical formulas for the GCN and TCN layers if you would like to hand that over to the developers, or a **slide deck** to present this plan to the team. Would you like me to generate either of those?

