# Hướng dẫn chạy cải tiến Spatial-Temporal Attention trên Colab & Kaggle

> Yêu cầu: đã chuẩn bị dữ liệu `data/` (METR-LA, PEMS-BAY, sensor_graph).

---

## 1. Chuẩn bị môi trường

### 1.1 Bật GPU
- **Colab:** Runtime > Change runtime type > Hardware accelerator > T4 GPU.
- **Kaggle:** Settings > Accelerator > GPU P100 hoặc GPU T4 x2.

### 1.2 Clone repo và cài dependencies

```bash
# Clone repo vào thư mục 'repo'
!git clone https://github.com/nhatvu205/innovations-on-graph-wavenet-for-spatial-temporal-graph-modeling.git repo
%cd repo
!pip install -r requirements.txt -q
```

### 1.3 Kết nối dữ liệu (Data Connection)

#### A. Trên Google Colab (Dùng Drive)
```python
from google.colab import drive
drive.mount('/content/drive')

# Thay đường dẫn tới folder chứa 'data' trên Drive của bạn
DRIVE_ROOT = "/content/drive/MyDrive/graph-wavenet"
!ln -sfn "$DRIVE_ROOT/data" data
!mkdir -p garage/metrics
```

#### B. Trên Kaggle (Dùng Kaggle Dataset)
Giả sử bạn đã upload thư mục `data` lên Kaggle Dataset tên là `gwnet-data`:
```python
# Kaggle dataset nằm trong /kaggle/input/
KAG_DATA = "/kaggle/input/gwnet-data/data"
!ln -sfn "$KAG_DATA" data
!mkdir -p garage/metrics
```

---

## 2. Lệnh `train.py` đã bao gồm test chưa?

**Có.** Sau khi kết thúc huấn luyện (hoặc dừng sớm), `train.py` tự động:
1. Load lại checkpoint tốt nhất (`_best.pth`).
2. Chạy inference trên tập test.
3. In MAE/MAPE/RMSE cho các horizon (mặc định 3, 6, 12).
4. Lưu file JSON tổng hợp kết quả vào `--metrics_out`.

---

## 3. Early Stopping, lưu checkpoint và resume

### Hành vi lưu checkpoint
- **`garage/{save}_best.pth`**: File đầy đủ (weights + optimizer + epoch) dùng để **Resume**. Ghi đè mỗi khi có Best mới.
- **`garage/{save}_exp{id}_best_{val}.pth`**: File model-only dùng để **Test**.

### Resume từ checkpoint
Nếu session bị ngắt, dùng `--resume` để tiếp tục từ epoch đã dừng:
```bash
!python train.py \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool --addaptadj --randomadj \
  --num_nodes 207 \
  --model_variant spatiotemporal \
  --epochs 100 \
  --early_stopping_patience 10 \
  --save garage/metr_st \
  --resume garage/metr_st_best.pth
```

---

## 4. Tổng quan các biến thể cải tiến

| `--model_variant` | Mô tả |
|---|---|
| `spatial` | Thêm Spatial Attention động (batch-specific adjacency) |
| `temporal` | Thêm Temporal Self-Attention (long-range dependencies) |
| `spatiotemporal` | Kết hợp cả hai (Joint ST-Attention Block) |

---

## 5. Chạy thực nghiệm (Ví dụ METR-LA)

### 5.1 Spatial Attention
```bash
!python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --randomadj --num_nodes 207 --model_variant spatial --save garage/metr_spatial --metrics_out garage/metrics/metr_spatial.json
```

### 5.2 Temporal Attention
```bash
!python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --randomadj --num_nodes 207 --model_variant temporal --save garage/metr_temporal --metrics_out garage/metrics/metr_temporal.json
```

### 5.3 Spatiotemporal Attention
```bash
!python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --randomadj --num_nodes 207 --model_variant spatiotemporal --save garage/metr_st --metrics_out garage/metrics/metr_st.json
```

---

## 6. Lưu ý quan trọng cho từng nền tảng

### Google Colab
- **Lưu trữ:** Thư mục `garage/` nằm local trong session. Hãy chạy cell backup sau khi train:
```bash
# Backup lên Drive
!mkdir -p "$DRIVE_ROOT/garage-backup"
!cp -r garage/* "$DRIVE_ROOT/garage-backup/"
```

### Kaggle
- **Lưu trữ:** Mọi file trong `/kaggle/working` sẽ được lưu khi bạn nhấn **"Save Version" > "Save & Run All"**.
- **Dataset:** Nếu upload data qua Kaggle Dataset, hãy đảm bảo cấu trúc thư mục đúng để symlink trỏ tới được.
- **GPU:** P100 (16GB) rất mạnh, có thể tăng `--batch_size 128` nếu cần.

---

## 7. Đọc kết quả metrics JSON

```python
import json
import glob

for path in glob.glob("garage/metrics/*.json"):
    with open(path) as f:
        r = json.load(f)
    print(f"Variant: {r['model_variant']} | Avg MAE: {r['average_metrics']['mae']:.4f}")
```
