# Hướng dẫn chạy cải tiến Spatial-Temporal Attention trên Google Colab

> Yêu cầu: đã chạy baseline thành công trước đó (có dữ liệu `data/` trên Google Drive).

---

## 1. Chuẩn bị môi trường

### 1.1 Bật GPU Runtime

Vào **Runtime > Change runtime type > Hardware accelerator > T4 GPU** trước khi bắt đầu.

### 1.2 Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

### 1.3 Clone repo và cài dependencies

```bash
!git clone https://github.com/nhatvu205/innovations-on-graph-wavenet-for-spatial-temporal-graph-modeling.git
%cd innovations-on-graph-wavenet-for-spatial-temporal-graph-modeling
!pip install -r requirements.txt -q
```

### 1.4 Kết nối thư mục data và garage từ Drive

Thay `YOUR_DRIVE_PATH` bằng đường dẫn thực tế bạn đã lưu dữ liệu trên Drive (ví dụ: `MyDrive/graph-wavenet`):

```bash
DRIVE_ROOT="/content/drive/MyDrive/graph-wavenet"

# Tạo symlink để train.py tìm đúng đường dẫn mặc định
!ln -sfn "$DRIVE_ROOT/data" data
!ln -sfn "$DRIVE_ROOT/garage" garage

# Tạo thư mục garage/metrics nếu chưa có
!mkdir -p "$DRIVE_ROOT/garage/metrics"
```

Kiểm tra dữ liệu tồn tại:

```bash
!ls data/METR-LA
!ls data/sensor_graph
```

Output mong đợi:
```
train.npz  val.npz  test.npz
adj_mx.pkl  adj_mx_bay.pkl
```

---

## 2. Tổng quan các biến thể cải tiến

| `--model_variant` | Mô tả |
|---|---|
| `spatial` | Thêm Spatial Attention động — adjacency thay đổi theo input |
| `temporal` | Thêm Temporal Self-Attention song song với dilated conv |
| `spatiotemporal` | Kết hợp cả hai (joint block) |

Tất cả biến thể đều tương thích ngược với cấu hình baseline (cùng `--gcn_bool`, `--addaptadj`, `--randomadj`).

---

## 3. Chạy từng biến thể

### 3.1 Spatial Attention — METR-LA

```bash
!python train.py \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool --addaptadj --randomadj \
  --num_nodes 207 \
  --model_variant spatial \
  --seed 42 \
  --epochs 100 \
  --save garage/metr_spatial \
  --expid 1 \
  --metrics_out garage/metrics/metr_spatial_seed42.json
```

### 3.2 Temporal Attention — METR-LA

```bash
!python train.py \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool --addaptadj --randomadj \
  --num_nodes 207 \
  --model_variant temporal \
  --temporal_attention_heads 4 \
  --seed 42 \
  --epochs 100 \
  --save garage/metr_temporal \
  --expid 1 \
  --metrics_out garage/metrics/metr_temporal_seed42.json
```

### 3.3 Spatiotemporal Attention (joint) — METR-LA

```bash
!python train.py \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool --addaptadj --randomadj \
  --num_nodes 207 \
  --model_variant spatiotemporal \
  --temporal_attention_heads 4 \
  --seed 42 \
  --epochs 100 \
  --save garage/metr_st \
  --expid 1 \
  --metrics_out garage/metrics/metr_st_seed42.json
```

---

## 4. Chạy trên PEMS-BAY (tùy chọn)

Thay các tham số sau cho dataset PEMS-BAY:

```bash
--data data/PEMS-BAY \
--adjdata data/sensor_graph/adj_mx_bay.pkl \
--num_nodes 325 \
--save garage/bay_<variant> \
--metrics_out garage/metrics/bay_<variant>_seed42.json
```

Ví dụ — Spatiotemporal trên PEMS-BAY:

```bash
!python train.py \
  --device cuda:0 \
  --data data/PEMS-BAY \
  --adjdata data/sensor_graph/adj_mx_bay.pkl \
  --adjtype doubletransition \
  --gcn_bool --addaptadj --randomadj \
  --num_nodes 325 \
  --model_variant spatiotemporal \
  --temporal_attention_heads 4 \
  --seed 42 \
  --epochs 100 \
  --save garage/bay_st \
  --expid 1 \
  --metrics_out garage/metrics/bay_st_seed42.json
```

---

## 5. Kiểm tra một checkpoint đã train

Dùng `test.py` khi muốn đánh giá lại một checkpoint cụ thể mà không cần train lại:

```bash
!python test.py \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool --addaptadj --randomadj \
  --num_nodes 207 \
  --model_variant spatiotemporal \
  --temporal_attention_heads 4 \
  --seed 42 \
  --checkpoint garage/metr_st_exp1_best_X.XX.pth \
  --metrics_out garage/metrics/metr_st_test_seed42.json
```

> Thay `X.XX` bằng giá trị val MAE thực tế trong tên file checkpoint được lưu sau khi train xong.

---

## 6. Đọc kết quả metrics JSON

```python
import json

for path in [
    "garage/metrics/metr_spatial_seed42.json",
    "garage/metrics/metr_temporal_seed42.json",
    "garage/metrics/metr_st_seed42.json",
]:
    with open(path) as f:
        r = json.load(f)
    avg = r["average_metrics"]
    horizons = {f"h{h['horizon']}": h["mae"] for h in r["selected_horizons"]}
    print(f"[{r['model_variant']}] Avg MAE: {avg['mae']:.4f} | {horizons}")
```

Output mẫu:
```
[spatial_attn]       Avg MAE: 3.02  | {'h3': 2.72, 'h6': 3.10, 'h12': 3.49}
[temporal_attn]      Avg MAE: 3.01  | {'h3': 2.71, 'h6': 3.09, 'h12': 3.48}
[spatiotemporal_attn] Avg MAE: 2.99 | {'h3': 2.69, 'h6': 3.06, 'h12': 3.45}
```

---

## 7. Lưu ý khi dùng Colab

| Vấn đề | Giải pháp |
|---|---|
| Session bị ngắt giữa chừng | Checkpoint được lưu sau mỗi epoch vào Drive qua symlink — chỉ cần re-mount và tiếp tục |
| Session reset hoàn toàn | Clone lại repo, re-mount Drive, symlink lại `data/` và `garage/` |
| Colab T4 hết bộ nhớ GPU | Giảm `--batch_size` xuống 32 |
| Muốn chạy nhanh thử | Thêm `--epochs 10` để kiểm tra pipeline trước |
| Muốn disable causal mask trong temporal attention | Thêm cờ `--no_temporal_causal_mask` |

---

## 8. Tham chiếu nhanh — tất cả cờ cải tiến

| Cờ | Mặc định | Mô tả |
|---|---|---|
| `--model_variant` | _(không set)_ | Shorthand: `spatial` / `temporal` / `spatiotemporal` |
| `--spatial_attention` | False | Bật spatial attention riêng lẻ (thay thế cho `--model_variant spatial`) |
| `--temporal_attention` | False | Bật temporal attention riêng lẻ |
| `--temporal_attention_heads` | 4 | Số attention heads |
| `--temporal_attention_dropout` | 0.0 | Dropout trong temporal attention |
| `--no_temporal_causal_mask` | False | Tắt causal mask (mặc định: bật) |
| `--seed` | 42 | Random seed để tái lập kết quả |
| `--eval_horizons` | `3,6,12` | Các horizon hiển thị trong summary |
| `--metrics_out` | _(không lưu)_ | Đường dẫn file JSON lưu kết quả |
