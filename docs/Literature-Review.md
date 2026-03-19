# Graph WaveNet: Hướng Dẫn Nghiên Cứu Toàn Diện

> **Dành cho:** Sinh viên nghiên cứu muốn hiểu sâu về Graph WaveNet và các hướng phát triển tiếp theo  
> **Ngôn ngữ:** Tiếng Việt (thuật ngữ kỹ thuật giữ nguyên tiếng Anh)  
> **Nguồn gốc:** Wu et al. (2019), IJCAI 2019; Shleifer et al. (2019), Stanford

---

## Mục Lục

1. [Bối Cảnh và Động Lực](#1-bối-cảnh-và-động-lực)
2. [Kiến Trúc Graph WaveNet](#2-kiến-trúc-graph-wavenet)
   - 2.1 [Định nghĩa bài toán](#21-định-nghĩa-bài-toán)
   - 2.2 [Graph Convolution Layer và Self-Adaptive Adjacency Matrix](#22-graph-convolution-layer-và-self-adaptive-adjacency-matrix)
   - 2.3 [Temporal Convolution Layer (Dilated Causal Convolution)](#23-temporal-convolution-layer-dilated-causal-convolution)
   - 2.4 [Gated TCN](#24-gated-tcn)
   - 2.5 [Kiến trúc tổng thể và luồng dữ liệu](#25-kiến-trúc-tổng-thể-và-luồng-dữ-liệu)
3. [Ví Dụ Minh Họa Cụ Thể](#3-ví-dụ-minh-họa-cụ-thể)
4. [Thực Nghiệm và Kết Quả](#4-thực-nghiệm-và-kết-quả)
5. [Phân Tích Bài Báo "Incrementally Improving Graph WaveNet"](#5-phân-tích-bài-báo-incrementally-improving-graph-wavenet)
6. [Hướng Nghiên Cứu Cải Tiến và Đổi Mới](#6-hướng-nghiên-cứu-cải-tiến-và-đổi-mới)
7. [Tóm Tắt So Sánh](#7-tóm-tắt-so-sánh)

---

## 1. Bối Cảnh và Động Lực

### 1.1 Bài Toán Spatial-Temporal Graph Modeling là gì?

Hãy tưởng tượng hệ thống giao thông của thành phố Los Angeles: có 207 cảm biến tốc độ gắn trên các đường cao tốc. Mỗi cảm biến ghi lại tốc độ trung bình của xe cộ mỗi 5 phút. Câu hỏi đặt ra: **dựa vào dữ liệu tốc độ 1 giờ qua, ta có thể dự đoán tốc độ giao thông trong 1 giờ tiếp theo không?**

Đây là bài toán **spatial-temporal graph modeling** — kết hợp đồng thời:
- **Không gian (Spatial):** Các cảm biến gần nhau hoặc kết nối với nhau có tốc độ tương quan (tắc ở nút A → tắc ở nút B gần đó).
- **Thời gian (Temporal):** Tốc độ tại một cảm biến biến động theo thời gian (giờ cao điểm sáng, giờ cao điểm chiều...).

```
Đồ thị giao thông:
    [Nút A] --(15km)--> [Nút B] --(8km)--> [Nút C]
       |                                       |
    (12km)                                  (10km)
       |                                       |
    [Nút D] <--(20km)---------------------- [Nút E]

Mỗi nút có chuỗi thời gian: [v(t-11), v(t-10), ..., v(t)] → Dự đoán [v(t+1), ..., v(t+12)]
```

### 1.2 Hai Hạn Chế Cốt Lõi của Các Phương Pháp Trước

**Hạn chế 1: Cấu trúc đồ thị cố định không phản ánh đúng sự phụ thuộc thực tế**

Các mô hình trước (DCRNN, STGCN) sử dụng adjacency matrix được xây dựng từ khoảng cách địa lý giữa các nút. Tuy nhiên:
- Hai nút **gần nhau nhưng trên các tuyến một chiều ngược nhau** → không ảnh hưởng nhau thực sự.
- Hai nút **xa nhau nhưng cùng tuyến đường chính** → ảnh hưởng mạnh đến nhau nhưng bị bỏ sót.

**Hạn chế 2: Không nắm bắt được chuỗi thời gian dài**

- **RNN-based** (DCRNN): Phải xử lý tuần tự từng bước thời gian → chậm, dễ bị vanishing gradient với chuỗi dài.
- **CNN-based** (STGCN): Dùng standard 1D convolution → receptive field tăng **tuyến tính** với số layer → cần rất nhiều layer để bao phủ chuỗi dài 1 giờ (12 bước × 5 phút).

---

## 2. Kiến Trúc Graph WaveNet

### 2.1 Định Nghĩa Bài Toán

**Đầu vào:**
- Đồ thị $\mathcal{G} = (\mathcal{V}, \mathcal{E})$ với $N$ nút, $|\mathcal{V}| = N$
- Adjacency matrix $A \in \mathbb{R}^{N \times N}$
- Lịch sử $S$ bước: $X^{(t-S):t} \in \mathbb{R}^{N \times D \times S}$

**Đầu ra:** Dự đoán $T$ bước tiếp theo: $\hat{X}^{(t+1):(t+T)} \in \mathbb{R}^{N \times D \times T}$

**Hàm ánh xạ:**

$$[X^{(t-S):t},\ \mathcal{G}] \xrightarrow{f} \hat{X}^{(t+1):(t+T)}$$

Với tập dữ liệu METR-LA: $N=207$, $D=1$ (tốc độ), $S=12$ (1 giờ lịch sử), $T=12$ (1 giờ tương lai).

---

### 2.2 Graph Convolution Layer và Self-Adaptive Adjacency Matrix

#### 2.2.1 Nền Tảng: Diffusion Convolution

Graph WaveNet kế thừa **diffusion convolution** từ DCRNN (Li et al., 2018). Ý tưởng: tín hiệu lan truyền trên đồ thị như quá trình khuếch tán (diffusion) theo cả hai hướng:

$$Z = \sum_{k=0}^{K} \left( P_f^k X W_{k1} + P_b^k X W_{k2} \right)$$

Trong đó:
- $P_f = A / \text{rowsum}(A)$: **Forward transition matrix** (lan truyền xuôi chiều)
- $P_b = A^T / \text{rowsum}(A^T)$: **Backward transition matrix** (lan truyền ngược chiều)
- $K$: Số bước diffusion (độ sâu lân cận xem xét, thường $K=2$)
- $W_{k1}, W_{k2}$: Ma trận trọng số cần học

**Ví dụ với $K=2$:** Mỗi nút tổng hợp thông tin từ các nút cách nó tối đa 2 bước trên đồ thị.

#### 2.2.2 Đổi Mới Chính: Self-Adaptive Adjacency Matrix

Đây là đóng góp kỹ thuật quan trọng nhất của bài báo. Graph WaveNet **học cấu trúc đồ thị ẩn** trực tiếp từ dữ liệu:

**Bước 1:** Khởi tạo ngẫu nhiên hai embedding dictionary:
$$E_1 \in \mathbb{R}^{N \times c}, \quad E_2 \in \mathbb{R}^{N \times c}$$

$E_1$: *source node embeddings*, $E_2$: *target node embeddings* ($c=10$ chiều)

**Bước 2:** Tính ma trận phụ thuộc:
$$\tilde{A}_{adp} = \text{SoftMax}\left(\text{ReLU}\left(E_1 E_2^T\right)\right)$$

**Giải thích từng phép toán:**
- $E_1 E_2^T$: Tích vô hướng giữa source node $i$ và target node $j$ → đo "mức độ ảnh hưởng" của nút $i$ lên nút $j$
- $\text{ReLU}(\cdot)$: Loại bỏ các kết nối yếu (âm) → **thưa hóa đồ thị**
- $\text{SoftMax}(\cdot)$: Chuẩn hóa → $\tilde{A}_{adp}$ có thể coi là transition matrix của một quá trình diffusion ẩn

**Bước 3:** Tích hợp vào graph convolution:

$$Z = \sum_{k=0}^{K} \left( P_f^k X W_{k1} + P_b^k X W_{k2} + \tilde{A}_{adp}^k X W_{k3} \right)$$

Khi **không có thông tin đồ thị nào** (graph structure unavailable):
$$Z = \sum_{k=0}^{K} \tilde{A}_{adp}^k X W_k$$

**Tại sao điều này hiệu quả?**

```
Đồ thị vật lý (cố định):      Đồ thị học được (adaptive):
    A → B → C                     A → C (phụ thuộc ẩn)
    ↑       ↓                     D → B (ảnh hưởng gián tiếp)
    D ← E ← F                     ... (tự khám phá)
```

$E_1$ và $E_2$ được học **end-to-end** cùng với toàn bộ mô hình qua gradient descent — không cần bất kỳ prior knowledge nào.

---

### 2.3 Temporal Convolution Layer (Dilated Causal Convolution)

#### 2.3.1 Vấn đề với Standard 1D Convolution

Với standard 1D convolution và kernel size $k$, receptive field sau $L$ layer là $L \times (k-1) + 1$, tăng **tuyến tính**. Để bao phủ 12 bước lịch sử, cần nhiều layer → nặng và chậm.

#### 2.3.2 Giải Pháp: Dilated Causal Convolution

Kế thừa từ **WaveNet** (van den Oord et al., 2016) — ban đầu dùng cho sinh âm thanh thô. Phép tích chập dãn (dilated convolution) với dilation factor $d$:

$$x \star f(t) = \sum_{s=0}^{K-1} f(s) \cdot x(t - d \times s)$$

**Causal:** Padding zeros ở đầu → tại bước $t$, chỉ dùng thông tin từ $t$ trở về trước (không "nhìn trộm" tương lai).

**Ví dụ trực quan với kernel size 2 và dilation factors tăng dần:**

```
Input:     [x₀, x₁, x₂, x₃, x₄, x₅, x₆, x₇, x₈, x₉, x₁₀, x₁₁]
                                                              ↑ dự đoán

Dilation=1: kết nối bước t với bước t-1
  ○   ○   ○   ○   ○   ○   ○   ○
  |---|   |---|   |---|   |---|
  (mỗi nút kết nối với nút liền kề)
  → Receptive field = 2

Dilation=2: kết nối bước t với bước t-2
  ○       ○       ○       ○
  |-------|   |-------|
  → Receptive field = 5 (sau 2 layer: 1 + 2×2 = 5)

Dilation=4: kết nối bước t với bước t-4
  ○               ○
  |---------------|
  → Receptive field = 9 (sau 3 layer: 1 + 2×2 + 2×4 = 13 → lũy thừa 2)
```

**Graph WaveNet** dùng 8 layer với dilation sequence: `[1, 2, 1, 2, 1, 2, 1, 2]`.

Receptive field sau 8 layer: $1 + (1+2) \times 4 = 13 \geq 12$ → **bao phủ đúng 12 bước lịch sử** với chỉ 8 layer.

So sánh với standard convolution: cần $12 / (k-1)$ layer → nặng hơn nhiều.

---

### 2.4 Gated TCN

Lấy cảm hứng từ **Gated Linear Units** (Dauphin et al., 2017) — cơ chế gating giúp kiểm soát luồng thông tin:

$$\mathbf{h} = \tanh(\Theta_1 \star \mathcal{X} + \mathbf{b}) \odot \sigma(\Theta_2 \star \mathcal{X} + \mathbf{c})$$

Trong đó:
- $\Theta_1, \Theta_2$: Trọng số của hai nhánh TCN song song (TCN-a và TCN-b)
- $\tanh(\cdot)$: Activation function, tạo ra output candidate
- $\sigma(\cdot)$: Sigmoid function — đóng vai trò **output gate**, quyết định bao nhiêu thông tin đi qua
- $\odot$: Phép nhân element-wise

**Tại sao cần gating?** Tương tự LSTM/GRU trong RNN, gating giúp mô hình học **khi nào nên chú ý và khi nào nên bỏ qua** một bước thời gian cụ thể, quan trọng với các mẫu traffic phức tạp như sự kiện đột biến.

---

### 2.5 Kiến Trúc Tổng Thể và Luồng Dữ Liệu

#### Sơ Đồ Kiến Trúc

```
INPUT: X ∈ ℝ^{N×D×S}
  │
  ▼
┌─────────────────────┐
│   Linear Projection  │  → chiếu vào hidden dim C
└──────────┬──────────┘
           │
   ┌───────▼────────────────────────────────────┐
   │   SPATIAL-TEMPORAL LAYER (lặp K=8 lần)      │
   │                                              │
   │  ┌──────────────────────────────────┐        │
   │  │          Gated TCN               │        │
   │  │  [TCN-a (tanh)] × [TCN-b (σ)]   │        │
   │  │   dilation = {1,2,1,2,1,2,1,2}  │        │
   │  └──────────────┬───────────────────┘        │
   │                 │                            │
   │  ┌──────────────▼───────────────────┐        │
   │  │          GCN Layer               │        │
   │  │  Z = Pf·X·W₁ + Pb·X·W₂ +        │        │
   │  │      Ã_adp·X·W₃                  │        │
   │  └──────────────┬───────────────────┘        │
   │                 │                            │
   │       ┌─────────┤  Residual +                │
   │       │         │  Skip connection           │
   │       │         ▼                            │
   │  ┌────┴────┐   [output]──────────────►       │
   │  │ (input) │                         │       │
   │  └─────────┘                         │       │
   └──────────────────────────────────────┼───────┘
                                          │
                              ┌───────────▼──────────┐
                              │   Skip Aggregation    │
                              │   Σ (all layer skips) │
                              └───────────┬──────────┘
                                          │
                              ┌───────────▼──────────┐
                              │  ReLU → Linear →      │
                              │  ReLU → Linear        │
                              └───────────┬──────────┘
                                          │
OUTPUT: X̂ ∈ ℝ^{N×D×T}       ◄──────────┘
```

#### Luồng Dữ Liệu Chi Tiết trong Một Spatial-Temporal Layer

```
Đầu vào layer i: h ∈ ℝ^{N × C × L}  (N nút, C kênh, L chiều thời gian còn lại)
    │
    ├──────────────────────────────────────┐ (Residual connection)
    │                                      │
    ▼                                      │
[Gated TCN]                                │
    Nhánh a: Θ₁ ★ h  → tanh              │
    Nhánh b: Θ₂ ★ h  → σ                 │
    h_gate = tanh(...) ⊙ σ(...)           │
    Shape: ℝ^{N × C × L'}                 │
    │                                      │
    ▼                                      │
[GCN]  áp dụng lên mỗi h[:,:,t]          │
    Dùng Pf, Pb, Ã_adp                     │
    Shape: ℝ^{N × C × L'}                 │
    │                                      │
    ▼                                      │
[+]  ←────────────────────────────────────┘
    h_out = GCN_out + h (residual)
    │
    ├──→ Skip connection → Output module
    │
    ▼
Đầu ra layer i+1: h_out ∈ ℝ^{N × C × L'}
```

**Lưu ý quan trọng:** Kích thước chiều thời gian $L' < L$ do dilated convolution cắt bớt. Điều này được thiết kế cẩn thận sao cho layer cuối cùng cho ra $L'=1$, sau đó được chiếu thành $T=12$ đầu ra.

#### Hàm Loss

$$\mathcal{L}(\hat{X}^{(t+1):(t+T)}; \Theta) = \frac{1}{TND} \sum_{i=1}^{T} \sum_{j=1}^{N} \sum_{k=1}^{D} \left| \hat{X}_{jk}^{(t+i)} - X_{jk}^{(t+i)} \right|$$

Dùng **MAE** thay vì MSE vì robust hơn với outlier (tắc đường đột ngột).

**Đặc điểm quan trọng:** Graph WaveNet dự đoán **toàn bộ $T=12$ bước một lúc** (non-recursive), thay vì predict từng bước. Điều này tránh được error accumulation.

---

## 3. Ví Dụ Minh Họa Cụ Thể

### 3.1 Ví Dụ Tính Self-Adaptive Adjacency Matrix

Giả sử có $N=4$ nút, embedding size $c=2$:

```python
import numpy as np

# Khởi tạo ngẫu nhiên
E1 = np.array([[0.5, 0.3],   # Nút 0 (giao lộ chính)
               [0.1, 0.8],   # Nút 1 (đường phụ)
               [0.4, 0.4],   # Nút 2 (đường cao tốc)
               [0.9, 0.1]])  # Nút 3 (đường một chiều)

E2 = np.array([[0.6, 0.2],
               [0.3, 0.7],
               [0.5, 0.5],
               [0.2, 0.9]])

# Bước 1: Tích ma trận
raw = E1 @ E2.T
# raw[i,j] = mức độ source_i ảnh hưởng target_j

# Bước 2: ReLU (loại kết nối âm)
after_relu = np.maximum(0, raw)

# Bước 3: SoftMax theo từng hàng
def softmax_row(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

A_adp = softmax_row(after_relu)
# A_adp[i,j] = xác suất lan truyền từ nút i đến nút j
```

Sau khi học hội tụ, $A_{adp}[i,j]$ lớn → nút $i$ có ảnh hưởng mạnh đến nút $j$ **dù chúng không nhất thiết gần nhau về địa lý**.

### 3.2 Ví Dụ Dilated Causal Convolution

```python
# Input: chuỗi tốc độ của 1 nút trong 12 bước
speeds = [65, 60, 55, 50, 45, 40, 38, 42, 50, 58, 62, 65]  
#         t=0  t=1  ...                                t=11

# Layer 1: dilation=1, kernel=[w0, w1]
# Tại t=11: output = w0*speeds[11] + w1*speeds[10] = w0*65 + w1*62

# Layer 2: dilation=2, kernel=[w0, w1]  
# Tại t=11: output = w0*h1[11] + w1*h1[9]
# (h1 là output của layer 1)

# Receptive field của 2 layer:
# Layer 1 tại t: dùng speeds[t], speeds[t-1]
# Layer 2 tại t: dùng h1[t], h1[t-2] → dùng speeds[t], t-1, t-2, t-3
# → Receptive field = 4 với chỉ 2 layer!

# So sánh Standard Conv (dilation=1):
# Layer 1: RF = 2, Layer 2: RF = 3, Layer 3: RF = 4 (3 layer để đạt RF=4)
```

### 3.3 Ví Dụ Thực Tế: Dự Đoán Tắc Đường

**Kịch bản:** Cảm biến 9 (tại giao lộ nhiều tuyến đường) lúc 8:00 sáng, tốc độ đang giảm từ 65 → 40 mph.

```
Dữ liệu đầu vào (12 bước × 5 phút = 60 phút lịch sử):
Cảm biến 9:  [65, 64, 62, 58, 55, 50, 47, 45, 43, 42, 41, 40] mph
Cảm biến 14: [70, 68, 65, 63, 61, 58, 56, 54, 52, 51, 50, 49] mph (upstream)
Cảm biến 43: [60, 58, 56, 55, 54, 53, 52, 50, 48, 46, 44, 43] mph (downstream)

Đồ thị vật lý: 14 → 9 → 43 (luồng giao thông xuôi chiều)
Đồ thị học được (Ã_adp): 
  - A_adp[14,9] = 0.82 (cao, đúng với thực tế upstream ảnh hưởng downstream)
  - A_adp[9,43] = 0.75 (tắc ở 9 lan sang 43)
  - A_adp[34,9] = 0.61 (nút 34 xa nhưng cùng tuyến chính → ảnh hưởng ẩn)

Dự đoán:
GCN tổng hợp: thông tin từ cả 3 nguồn (vật lý + ẩn)
TCN phát hiện: xu hướng giảm trong 60 phút → dự đoán tiếp tục giảm
→ Đầu ra: [39, 38, 37, 36, 35, 35, 36, 37, 38, 40, 42, 44] mph
   (dự đoán tắc đường nghiêm trọng nhất vào 9:00-9:30, sau đó cải thiện)
```

---

## 4. Thực Nghiệm và Kết Quả

### 4.1 Bộ Dữ Liệu

| Tập dữ liệu | #Nút | #Cạnh | #Bước thời gian | Vùng |
|---|---|---|---|---|
| METR-LA | 207 | 1,515 | 34,272 (≈4 tháng) | Los Angeles CA |
| PEMS-BAY | 325 | 2,369 | 52,116 (≈6 tháng) | Bay Area CA |

**Tiền xử lý:**
- Chuẩn hóa Z-score trên dữ liệu training
- Adjacency matrix: Gaussian kernel trên khoảng cách đường bộ, với ngưỡng $k=0.1$
- Phân chia: 70% train / 10% validation / 20% test (theo thứ tự thời gian)

### 4.2 Kết Quả So Sánh

**Trên METR-LA:**

| Mô hình | MAE-15 | MAE-30 | MAE-60 | RMSE-60 | MAPE-60 |
|---|---|---|---|---|---|
| ARIMA | 3.99 | 5.15 | 6.90 | 13.23 | 17.40% |
| FC-LSTM | 3.44 | 3.77 | 4.37 | 8.69 | 13.20% |
| WaveNet | 2.99 | 3.59 | 4.45 | 8.93 | 13.62% |
| DCRNN | 2.77 | 3.15 | 3.60 | 7.60 | 10.50% |
| STGCN | 2.88 | 3.47 | 4.59 | 9.40 | 12.70% |
| GGRU | 2.71 | 3.12 | 3.64 | 7.65 | 10.62% |
| **Graph WaveNet** | **2.69** | **3.07** | **3.53** | **7.37** | **10.01%** |

**Điểm nổi bật:** Cải thiện rõ rệt nhất ở **60 phút** — thách thức nhất về dự đoán dài hạn.

### 4.3 Ablation Study: Hiệu Quả của Self-Adaptive Adjacency Matrix

| Cấu hình | Mean MAE (METR-LA) |
|---|---|
| Identity [I] | 3.58 |
| Forward-only [Pf] | 3.13 |
| **Adaptive-only [Ã_adp]** | **3.10** |
| Forward-backward [Pf, Pb] | 3.08 |
| **Forward-backward-adaptive [Pf, Pb, Ã_adp]** | **3.04** ✓ |

**Nhận xét quan trọng:** Adaptive-only (3.10) > Forward-only (3.13) → **không cần biết cấu trúc đồ thị vẫn đạt kết quả tốt**.

### 4.4 Chi Phí Tính Toán

| Mô hình | Training (s/epoch) | Inference (s) |
|---|---|---|
| DCRNN | 249.31 | 18.73 |
| STGCN | 19.10 | 11.37 |
| **Graph WaveNet** | **53.68** | **2.27** |

Graph WaveNet **nhanh hơn DCRNN 5×** trong training và **nhanh nhất khi inference** vì predict 12 bước cùng lúc.

---

## 5. Phân Tích Bài Báo "Incrementally Improving Graph WaveNet"

*(Shleifer, McCreery, Chitters — Stanford, 2019)*

### 5.1 Tổng Quan

Bài báo này đề xuất một chuỗi cải tiến incremental trên Graph WaveNet, giảm Mean MAE trên METR-LA từ **3.04 xuống 2.98** — cải thiện tương đương với khoảng cách từ DCRNN đến GWN gốc (0.07 MAE).

### 5.2 Cải Tiến 1: Hyperparameter Tuning (Giảm MAE ≈ 0.03)

#### a) Learning Rate Decay

**GWN gốc:** Learning rate cố định (không decay).  
**GWNV2:** Nhân LR với 0.97 sau mỗi epoch.

**Tại sao hiệu quả?** LR decay giúp mô hình "hạ nhiệt" dần — ở cuối quá trình training, các bước cập nhật nhỏ hơn giúp fine-tune quanh cực tiểu thay vì dao động qua lại.

#### b) Tăng Số Filters (32 → 40)

**Tác động:** Số tham số tăng từ 309,400 lên 477,872 (+54%), nhưng training time chỉ tăng 5%.

**Trade-off:** Biểu diễn phong phú hơn với chi phí tính toán chấp nhận được.

#### c) Gradient Clipping (L2 = 5 → L2 = 3)

Việc thêm skip connection (xem mục 5.3) làm gradients lớn hơn → cần clipping chặt hơn để tránh exploding gradients.

#### d) Xử Lý Dữ Liệu Thiếu (Missing Data)

**Vấn đề:** ~5% readings = 0 (không có xe qua sensor) → mô hình phải học ngoại lệ: "tốc độ thấp = tắc, NGOẠI TRỪ khi = 0 (không có xe)".

**Giải pháp:** Thay 0 bằng **tốc độ trung bình** của training data → mô hình không bị nhiễu bởi các giá trị 0 vô nghĩa.

**Kết quả:** Giảm thêm 0.01 MAE + hội tụ nhanh hơn.

### 5.3 Cải Tiến 2: Skip Connection Bổ Sung Around GCN (Giảm MAE ≈ 0.02)

**Vấn đề trong GWN gốc:** Gradients phải đi qua GCN để về các layer TCN đầu tiên → bị yếu đi (vanishing gradients).

**Thay đổi kiến trúc:**

```
GWN gốc:
    TCN → GCN → output_i+1

GWNV2:
    TCN → GCN → (+) → output_i+1
              ↑
        TCN_output (skip around GCN)
```

Tức là: $x_{i+1} = x_i + \text{GraphConv}(x_i)$ thay vì $x_{i+1} = \text{GraphConv}(x_i)$

**Cơ chế:** Tương tự ResNet's residual connection nhưng **trong mỗi block** thay vì giữa các block. Gradients có con đường trực tiếp về layer đầu mà không cần đi qua GCN.

**Phân tích sâu hơn:** Với 8 layer, gradients phải đi qua 8 GCN layers trong GWN gốc → mỗi GCN có Jacobian $\|J\| < 1$ → tích $\prod_{i=1}^{8} \|J_i\| \approx 0$. Skip connection phá vỡ điều này: gradient = $1 + \text{grad qua GCN}$.

### 5.4 Cải Tiến 3: Pretraining trên Shorter Prediction Horizons (Giảm MAE ≈ 0.01)

**Quan sát:** Model chuyên biệt cho dự đoán ngắn hạn (t≤6, tức 30 phút) làm tốt hơn model toàn diện ở horizons ngắn.

**Chiến lược:**
1. Train model trên task dễ hơn: chỉ predict 6 bước đầu (30 phút) → hội tụ sau 60 epochs
2. Fine-tune trên full task (12 bước) → hội tụ sau 29 epochs (nhanh hơn train from scratch 100 epochs)

**Tại sao hiệu quả?** Transfer learning giúp khởi tạo tốt hơn — model đã học các đặc trưng giao thông ngắn hạn, sau đó mở rộng ra dài hạn.

**Chi phí:** Chỉ thêm 7 phút so với train from scratch.

### 5.5 Cải Tiến 4: Range Ensemble (Giảm MAE thêm 0.01)

**Vấn đề:** Model fine-tuned trên full task "quên" (catastrophic forgetting) các đặc trưng ngắn hạn.

**Giải pháp Ensemble:**
- **Ngắn hạn (t=1→6):** Dùng model pre-trained trên t≤6
- **Dài hạn (t=7→12):** Dùng model fine-tuned trên full task

**Kết quả cuối:** METR-LA Mean MAE = **2.98** (state-of-the-art tại thời điểm 2019)

### 5.6 Những Gì Đã Thử Nhưng Không Hiệu Quả (Failed Experiments)

Phần này **cực kỳ giá trị** cho sinh viên nghiên cứu — tiết kiệm thời gian tránh lặp lại:

| Thử nghiệm | Kết quả | Lý giải |
|---|---|---|
| Thêm day-of-week | Không cải thiện | Short-term readings đã encode xu hướng tuần |
| Dùng 75 phút lịch sử (thay vì 60) | Không cải thiện | Plateu sau 5 bước |
| Transformer thay vì 1D Conv | Chậm, tệ hơn | O(n²) attention không phù hợp traffic |
| Half precision training | Tệ hơn | Model không memory-bound ở batch size mặc định |
| Transfer giữa METR-LA và PEMS-BAY | MAE tệ hơn 0.03 | Topology khác nhau |
| Batch normalization sau GCN | Không cải thiện | — |

---

## 6. Hướng Nghiên Cứu Cải Tiến và Đổi Mới

*Mỗi hướng được hỗ trợ bởi các bài báo từ hội nghị/tạp chí uy tín.*

---

### 6.1 Dynamic Graph Structure Learning

**Hạn chế hiện tại của GWN:** $\tilde{A}_{adp}$ là ma trận **tĩnh** — được học một lần và cố định suốt quá trình inference. Tuy nhiên, cấu trúc phụ thuộc không gian trong giao thông **thay đổi theo thời gian**: giờ cao điểm sáng vs chiều vs cuối tuần tạo ra các pattern khác nhau.

**Hướng cải tiến:**

**Phương án A — Input-conditioned dynamic graph:**

$$\tilde{A}_{adp}^{(t)} = f_\theta(X^{(t-S):t})$$

Đồ thị thay đổi **theo đầu vào** tại mỗi time window. Tiêu biểu:

> **AGCRN** (Bai et al., NeurIPS 2020): Adaptive Graph Convolutional Recurrent Network. Học node-specific patterns kết hợp với graph learning động.

> **DSTGCN** (Wu et al., KDD 2023): Dynamic Spatial-Temporal Cross Dependencies.

**Phương án B — Attention-based dynamic graph:**

$$A^{(t)}_{ij} = \text{Attention}(Q_i^{(t)}, K_j^{(t)})$$

Query và Key được tính từ node features tại thời điểm $t$:

> **ASTGCN** (Guo et al., AAAI 2019): Attention-Based Spatial-Temporal GCN — dùng spatial attention và temporal attention để tính dynamic weights.

> **GMAN** (Zheng et al., AAAI 2020): Graph Multi-Attention Network — sử dụng spatial và temporal attention encoder-decoder.

**Tiềm năng cải tiến:** Kết quả thực nghiệm trên AASTGNet (MDPI Electronics, 2024) cho thấy graph self-learning module động cải thiện đáng kể so với ma trận tĩnh trên dữ liệu real-time có abrupt changes.

---

### 6.2 Transformer-Based Temporal Modeling

**Hạn chế hiện tại:** Dilated convolution có receptive field cố định, không thể **chú ý linh hoạt** đến các bước thời gian quan trọng (ví dụ: tắc đường cách 3 ngày tương tự hôm nay).

**Hướng cải tiến — Thay TCN bằng Self-Attention:**

$$\text{Attention}(Q, K, V) = \text{SoftMax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Ưu điểm: Capture **long-range temporal dependencies** linh hoạt với O(1) depth thay vì O(log n) của dilated conv.

**Bài báo tiêu biểu:**

> **STAEformer** (Liu et al., CIKM 2023): Spatio-Temporal Adaptive Embedding Makes Vanilla Transformers SOTA for Traffic Forecasting — đơn giản hóa bằng positional encoding thích nghi.

> **STGformer** (2024): Efficient Spatiotemporal Graph Transformer — cân bằng GCN và Transformer, giảm chi phí tính toán.

> **PDFormer** (Jiang et al., AAAI 2023): Propagation Delay-aware Dynamic Long-range Transformer for traffic forecasting.

**Thách thức:** Transformer có độ phức tạp $O(L^2)$ theo chiều dài chuỗi — cần các kỹ thuật như **sparse attention**, **linear attention**, hoặc **local-global attention** để scale up.

**Đề xuất cụ thể cho GWN:** Thay Gated TCN bằng **Multi-Scale Temporal Attention** kết hợp local (dilated conv, fast) và global (sparse attention, long-range):

$$h_t = \alpha \cdot \text{LocalConv}(t) + (1-\alpha) \cdot \text{GlobalAttn}(t)$$

---

### 6.3 Causal và Disentangled Representation Learning

**Hạn chế:** GWN học correlation — không phân biệt **causal relationships** (A gây ra B) và **spurious correlations** (A và B cùng bị ảnh hưởng bởi C không quan sát được).

**Ý nghĩa thực tiễn:** Tắc ở nút 9 **gây ra** tắc ở nút 14 (causal) khác với nút 9 và nút 14 đều tắc vì **giờ cao điểm** (confounded). Mô hình causal sẽ generalize tốt hơn khi điều kiện thay đổi.

**Bài báo tiêu biểu:**

> **CaST** (NeurIPS 2023): Deciphering Spatio-Temporal Graph Forecasting — A Causal Lens and Treatment. Đề xuất framework causal inference cho ST forecasting, phân tách causal edges và confounded edges.

**Hướng áp dụng vào GWN:**
1. Thêm **environment disentanglement** để tách entity-specific patterns (đặc trưng của từng nút) và environment patterns (giờ cao điểm, thời tiết) ra khỏi adjacency matrix.
2. Dùng **causal discovery** để học $\tilde{A}_{adp}$ theo hướng causal thay vì correlation.

---

### 6.4 Multi-Scale Spatial-Temporal Modeling

**Hạn chế:** GWN sử dụng một cấp độ độ phân giải không gian và thời gian. Trong thực tế:
- **Thời gian:** Pattern 5 phút (micro), pattern theo giờ (meso), pattern theo ngày/tuần (macro)
- **Không gian:** Nút đơn lẻ, khu vực, toàn mạng lưới

**Bài báo tiêu biểu:**

> **STWave+** (Fang et al., IEEE TKDE 2023): Multi-Scale Efficient Spectral Graph Attention Network with Long-Term Trends for Disentangled Traffic Flow Forecasting. Phân tách trend và residual components ở nhiều scale.

> **STSGCN** (Song et al., AAAI 2020): Spatial-Temporal Synchronous Graph Convolutional Networks — xây dựng local spatial-temporal graph để capture synchronized correlations.

> **MFSTGCN** (Scientific Reports, 2025): Multi-Factor Spatial-Temporal Fusion GCN — tích hợp thông tin đa nguồn (traffic + thời tiết) vào multi-scale framework.

**Đề xuất cụ thể:**

Mở rộng GWN với **hierarchical temporal processing**:

```
Short-term (t-2:t) → Fast Conv (d=1,2)      ─┐
Medium-term (t-6:t) → Medium Conv (d=4,8)   ─┼→ Fusion → GCN → Output
Long-term (t-12:t) → Slow Conv (d=16)       ─┘
```

---

### 6.5 Self-Supervised và Contrastive Pre-training

**Hạn chế:** GWN cần large labeled dataset và khó generalize sang domains khác. Failed experiment của Shleifer et al.: transfer giữa METR-LA và PEMS-BAY giảm performance.

**Hướng cải tiến — Pre-training framework:**

Pre-train GWN encoder bằng **Masked Autoencoding (MAE)** trên raw traffic data (không cần nhãn), sau đó fine-tune trên task dự đoán:

> **GPT-ST** (NeurIPS 2023): Generative Pre-Training of Spatio-Temporal Graph Neural Networks. Đề xuất masked ST pre-training với hierarchical spatial encoder — cải thiện **mọi** baseline downstream model thử nghiệm.

> **STEP** (Shao et al., KDD 2022): Pre-training Enhanced Spatial-temporal GNN for Multivariate Time Series Forecasting. Pre-train trên long temporal context (không chỉ 1 giờ) để học broader patterns.

> **ImPreSTDG** (Scientific Reports, 2025): Improved Spatiotemporal Diffusion Graph với pre-training method — giải quyết long-term dependencies và high computational cost.

**Quy trình đề xuất:**

```
Phase 1 — Self-supervised Pre-training:
  Mask 20-50% of sensor readings → Reconstruction task
  Học: encoder E = GWN_backbone
  Không cần nhãn → có thể dùng dữ liệu từ nhiều thành phố

Phase 2 — Fine-tuning:
  Khởi tạo từ E (pre-trained)
  Fine-tune với forecasting loss (MAE)
  → Convergence nhanh hơn, generalize tốt hơn
```

---

### 6.6 Tích Hợp Thông Tin Bên Ngoài (Exogenous Features)

**Hạn chế:** GWN chỉ dùng lịch sử tốc độ. Thực tế, traffic bị ảnh hưởng bởi:
- Thời tiết (mưa, sương mù)
- Sự kiện đặc biệt (concert, thể thao)
- Ngày lễ, ngày làm việc
- Tai nạn, công trình

**Bài báo tiêu biểu:**

> **STFGCN** (Scientific Reports, 2025): Spatial-Temporal Multi-Factor Fusion GCN — tích hợp weather data và road network features vào GCN framework.

> **CLCRN** (Liu et al., 2022): Contrastive Learning-based Cross-domain Retrieval Network — dùng contrastive learning để align features từ các domain khác nhau.

**Đề xuất tích hợp vào GWN:**

```python
# Thêm exogenous embedding vào node features
X_augmented = concat([X_traffic,        # [N, 1, S]
                       X_time_of_day,    # [N, 2, S]  (sin/cos encoding)
                       X_day_of_week,    # [N, 7, S]  (one-hot)
                       X_weather],       # [N, 4, S]  (nhiệt độ, lượng mưa...)
                      dim=1)

# Linear projection xuống C chiều trước khi vào GWN
h0 = Linear(X_augmented)  # [N, C, S]
```

*Lưu ý:* Shleifer et al. đã thử day-of-week và thất bại — cần cách tích hợp tinh tế hơn (ví dụ: **time-aware attention** thay vì concatenation đơn giản).

---

### 6.7 Neural ODE / Continuous-time Modeling

**Hạn chế:** GWN giả định dữ liệu được sample đều (5 phút/bước). Trong thực tế:
- Sensor có thể bị mất dữ liệu không đều
- Cần dự đoán tại các time point tùy ý, không nhất thiết theo lưới cố định

**Bài báo tiêu biểu:**

> **STGODE** (Fang et al., KDD 2021): Spatial-Temporal Graph Ordinary Differential Equations — thay GCN discrete bằng Neural ODE để model continuous spatial propagation.

> **STG-NCDE** (Choi et al., AAAI 2022): Graph Neural Controlled Differential Equations for Traffic Forecasting — dùng CDE thay vì discrete RNN để xử lý irregular sampling.

**Ý tưởng áp dụng vào GWN:**

Thay Gated TCN (discrete) bằng **latent ODE**:
$$\frac{d\mathbf{h}}{dt} = f_\theta(\mathbf{h}(t), \mathcal{G}, t)$$

Cho phép predict tại bất kỳ thời điểm nào, robust với missing data.

---

### 6.8 Scalability và Distributed Learning

**Hạn chế:** GWN yêu cầu toàn bộ adjacency matrix trong memory → khó scale lên mạng lưới lớn (N > 1000 sensors).

**Bài báo tiêu biểu:**

> **Scalable STGNN** (2022-2023): Nhiều công trình về mini-batch training cho large-scale ST graphs, clustering-based approaches.

> **Federated STGNN** (Hu et al., IEEE Transactions on Vehicular Technology, 2023): Federated Learning framework cho traffic prediction, bảo vệ privacy của từng khu vực.

**Hướng cải tiến:**
1. **Graph clustering:** Chia N nút thành K cluster, học intra-cluster và inter-cluster dynamics riêng biệt.
2. **Node sampling:** Chỉ sample subset of neighbors cho mỗi bước GCN (GraphSAGE-style).
3. **Linear attention:** Xấp xỉ $O(N^2)$ attention bằng $O(N)$ kernel methods.

---

## 7. Tóm Tắt So Sánh

### 7.1 So Sánh Ba Thế Hệ Mô Hình

| Khía cạnh | DCRNN (2018) | Graph WaveNet (2019) | GWNV2 + Improvements |
|---|---|---|---|
| Temporal Modeling | GRU (sequential) | Dilated Causal Conv | Dilated Conv + Skip |
| Spatial Modeling | Fixed diffusion | Fixed + Adaptive | Fixed + Adaptive |
| Graph Structure | Pre-defined | Pre-defined + Learnable | Pre-defined + Learnable |
| Inference Mode | Recursive (1 step) | Non-recursive (12 steps) | Non-recursive |
| Training Time/epoch | 249s | 54s | ~55s |
| Inference Time | 18.7s | 2.3s | ~2.2s |
| METR-LA MeanMAE | 3.11 | 3.04 | **2.98** |
| PEMS-BAY MeanMAE | 1.68 | 1.58 | **1.55** |

### 7.2 Ma Trận Cải Tiến

| Hướng Nghiên Cứu | Khó khăn | Tiềm năng | Bài báo tham khảo |
|---|---|---|---|
| Dynamic Graph | Trung bình | Cao | AGCRN (NeurIPS 2020), AASTGNet (2024) |
| Transformer Temporal | Cao (O(n²)) | Rất cao | STAEformer (CIKM 2023), STGformer (2024) |
| Causal Learning | Rất cao | Rất cao | CaST (NeurIPS 2023) |
| Multi-Scale | Trung bình | Cao | STWave+ (TKDE 2023) |
| Self-Supervised Pretrain | Cao | Rất cao | GPT-ST (NeurIPS 2023), STEP (KDD 2022) |
| Exogenous Features | Thấp | Trung bình | STFGCN (2025) |
| Neural ODE | Rất cao | Cao | STGODE (KDD 2021), STG-NCDE (AAAI 2022) |
| Scalability | Cao | Cao | Federated STGNN (2023) |

---

## Tài Liệu Tham Khảo Chính

### Bài Báo Gốc
1. **Wu et al. (2019)** — Graph WaveNet for Deep Spatial-Temporal Graph Modeling. *IJCAI 2019*. arXiv:1906.00121
2. **Shleifer et al. (2019)** — Incrementally Improving Graph WaveNet Performance on Traffic Prediction. *Stanford CS224W*. arXiv:1912.07390

### Nền Tảng
3. **Li et al. (2018)** — DCRNN: Diffusion Convolutional Recurrent Neural Network. *ICLR 2018*
4. **Yu et al. (2018)** — STGCN: Spatio-Temporal Graph Convolutional Networks. *IJCAI 2018*
5. **van den Oord et al. (2016)** — WaveNet: A Generative Model for Raw Audio. arXiv:1609.03499

### Cải Tiến Đã Được Chứng Minh
6. **Bai et al. (2020)** — AGCRN: Adaptive Graph Convolutional Recurrent Network. *NeurIPS 2020*
7. **Guo et al. (2019)** — ASTGCN: Attention Based Spatial-Temporal GCN. *AAAI 2019*
8. **Zheng et al. (2020)** — GMAN: Graph Multi-Attention Network. *AAAI 2020*

### Hướng Nghiên Cứu Mới
9. **Liu et al. (2023)** — STAEformer: Vanilla Transformers SOTA for Traffic Forecasting. *CIKM 2023*
10. **Chen et al. (2023)** — GPT-ST: Generative Pre-Training of Spatio-Temporal GNNs. *NeurIPS 2023*
11. **Shao et al. (2022)** — STEP: Pre-training Enhanced ST-GNN. *KDD 2022*
12. **NeurIPS 2023** — CaST: Causal Spatio-Temporal Forecasting
13. **Fang et al. (2021)** — STGODE: Spatial-Temporal Graph ODE. *KDD 2021*
14. **Choi et al. (2022)** — STG-NCDE: Graph Neural CDE for Traffic. *AAAI 2022*
15. **Kong et al. (2024)** — STG4Traffic: Survey and Benchmark of ST-GNNs. arXiv:2307.00495

---

*Tài liệu này được tổng hợp dựa trên hai bài báo gốc và tổng quan tài liệu học thuật đến năm 2024. Các con số kết quả được lấy trực tiếp từ bài báo gốc để đảm bảo độ chính xác.*
