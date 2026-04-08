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

Thêm `repo` ở cuối lệnh clone để Colab đặt tên thư mục là `repo` thay vì tên mặc định dài:

```bash
!git clone https://github.com/nhatvu205/innovations-on-graph-wavenet-for-spatial-temporal-graph-modeling.git repo
%cd repo
!pip install -r requirements.txt -q
```

### 1.4 Kết nối thư mục data từ Drive và tạo garage cục bộ

`data/` được symlink từ Drive (dùng lại dữ liệu đã tải). `garage/` được tạo **local bên trong thư mục `repo`** để checkpoint của thí nghiệm này tách biệt hoàn toàn với các thí nghiệm khác.

Thay `YOUR_DRIVE_PATH` bằng đường dẫn thực tế trên Drive của bạn (ví dụ: `MyDrive/graph-wavenet`):

```bash
DRIVE_ROOT="/content/drive/MyDrive/graph-wavenet"

# Symlink data từ Drive (chỉ đọc, không ghi)
!ln -sfn "$DRIVE_ROOT/data" data

# Tạo garage cục bộ trong repo — checkpoint lưu tại đây
!mkdir -p garage/metrics
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

## 2. Lệnh `train.py` đã bao gồm test chưa?

**Có.** Sau khi toàn bộ vòng lặp epoch kết thúc (hoặc dừng sớm do early stopping), `train.py` tự động:

1. Load lại checkpoint có val loss thấp nhất.
2. Chạy inference trên tập test.
3. In MAE/MAPE/RMSE cho từng horizon 1–12 và trung bình 12 horizon.
4. Lưu file JSON tổng hợp kết quả (nếu có `--metrics_out`).

**Không cần chạy thêm `test.py`** trừ khi muốn đánh giá lại một checkpoint cụ thể sau đó.

---

## 3. Early Stopping và lưu checkpoint

Từ phiên bản cải tiến này, `train.py` thay đổi hai hành vi quan trọng so với baseline:

| Hành vi | Baseline cũ | Phiên bản mới |
|---|---|---|
| Lưu checkpoint | Sau **mỗi** epoch | Chỉ khi có **val loss mới thấp hơn** |
| Dừng huấn luyện | Luôn chạy hết `--epochs` | Dừng sớm nếu không cải thiện `--early_stopping_patience` epoch liên tiếp |

**Tham số điều chỉnh:**

- `--early_stopping_patience 10` _(mặc định)_ — dừng sau 10 epoch không cải thiện.
- `--early_stopping_patience 0` — tắt early stopping, chạy hết `--epochs`.

**Log mẫu khi chạy:**
```
Epoch: 042, ..., Valid Loss: 3.0421, ...  [NEW BEST]
Epoch: 043, ..., Valid Loss: 3.0498, ...  (no improvement 1/10)
...
Epoch: 052, ..., Valid Loss: 3.0711, ...  (no improvement 10/10)
Early stopping triggered at epoch 52. No improvement in val loss for 10 consecutive epochs.
Training finished. Best validation loss: 3.0421 (epoch 42)
```

---

## 4. Tổng quan các biến thể cải tiến

| `--model_variant` | Mô tả |
|---|---|
| `spatial` | Thêm Spatial Attention động — adjacency thay đổi theo input |
| `temporal` | Thêm Temporal Self-Attention song song với dilated conv |
| `spatiotemporal` | Kết hợp cả hai (joint block) |

Tất cả biến thể đều tương thích ngược với cấu hình baseline (cùng `--gcn_bool`, `--addaptadj`, `--randomadj`).

---

## 5. Chạy từng biến thể

Mỗi lệnh dưới đây tự động:
- Lưu checkpoint chỉ khi val loss cải thiện.
- Dừng sớm sau 10 epoch không cải thiện (`--early_stopping_patience 10`).
- Chạy test trên tập test và lưu kết quả JSON.

### 5.1 Spatial Attention — METR-LA

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
  --early_stopping_patience 10 \
  --save garage/metr_spatial \
  --expid 1 \
  --metrics_out garage/metrics/metr_spatial_seed42.json
```

### 5.2 Temporal Attention — METR-LA

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
  --early_stopping_patience 10 \
  --save garage/metr_temporal \
  --expid 1 \
  --metrics_out garage/metrics/metr_temporal_seed42.json
```

### 5.3 Spatiotemporal Attention (joint) — METR-LA

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
  --early_stopping_patience 10 \
  --save garage/metr_st \
  --expid 1 \
  --metrics_out garage/metrics/metr_st_seed42.json
```

---

## 6. Chạy trên PEMS-BAY (tùy chọn)

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
  --early_stopping_patience 10 \
  --save garage/bay_st \
  --expid 1 \
  --metrics_out garage/metrics/bay_st_seed42.json
```

---

## 7. Kiểm tra một checkpoint đã train (tùy chọn)

Vì `train.py` đã chạy test tự động, `test.py` chỉ cần dùng khi muốn đánh giá lại một checkpoint cụ thể mà không train lại:

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

## 8. Đọc kết quả metrics JSON

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

## 9. Lưu ý khi dùng Colab

| Vấn đề | Giải pháp |
|---|---|
| Session bị ngắt giữa chừng | Checkpoint nằm **local** trong `repo/garage/` — bị mất khi session reset. Copy checkpoint quan trọng lên Drive ngay sau khi train xong (xem bên dưới) |
| Session reset hoàn toàn | Clone lại repo thành `repo`, re-mount Drive, symlink lại `data/`, tạo lại `garage/metrics/`. Checkpoint cũ khôi phục từ Drive nếu đã copy |
| Colab T4 hết bộ nhớ GPU | Giảm `--batch_size` xuống 32 |
| Muốn chạy nhanh thử | Thêm `--epochs 10` để kiểm tra pipeline trước |
| Muốn disable causal mask trong temporal attention | Thêm cờ `--no_temporal_causal_mask` |

**Sao lưu checkpoint và metrics lên Drive sau khi train:**

```bash
DRIVE_ROOT="/content/drive/MyDrive/graph-wavenet"

# Tạo thư mục backup trên Drive nếu chưa có
!mkdir -p "$DRIVE_ROOT/garage-st-attention/metrics"

# Copy toàn bộ checkpoint và metrics JSON
!cp garage/*.pth "$DRIVE_ROOT/garage-st-attention/" 2>/dev/null || true
!cp garage/metrics/*.json "$DRIVE_ROOT/garage-st-attention/metrics/" 2>/dev/null || true
echo "Backup done."
```

---

## 10. Tham chiếu nhanh — tất cả cờ cải tiến

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
| `--early_stopping_patience` | 10 | Dừng sớm nếu val loss không cải thiện sau N epoch (0 = tắt) |
