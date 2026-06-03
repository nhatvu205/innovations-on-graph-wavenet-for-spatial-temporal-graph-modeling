# Notebooks

Thư mục chứa các Jupyter Notebook phân tích và thực nghiệm bổ sung cho dự án.

---

## `pems_bay_error_analysis.ipynb`

Notebook phân tích sai số của mô hình **Graph WaveNet + Dynamic Adaptive Adjacency** (`mod_03_dynamic_adj`) sau khi huấn luyện trên tập dữ liệu **PEMS-BAY** (325 cảm biến giao thông, vịnh San Francisco).

### Luồng thực thi

**1. Setup & Patching model**
- Clone repo từ nhánh `feat/STAttention` và cài dependencies.
- Định nghĩa hai class mở rộng:
  - `DynamicAdaptiveAdj`: tính ma trận kề động theo từng batch bằng cách chiếu đặc trưng ẩn qua Conv1x1 rồi nhân với node embeddings.
  - `gcn_patched`: GCN layer hỗ trợ cả adjacency tĩnh (2D) lẫn động (3D batch-wise).
- Monkey-patch `model.gwnet.__init__`, `model.gwnet.forward`, và `engine.trainer.__init__` để inject dynamic adjacency vào kiến trúc gốc mà không sửa source code.

**2. Load dữ liệu & Checkpoint**
- Tải adjacency matrix PEMS-BAY (`adj_mx_bay.pkl`, `doubletransition`).
- Load checkpoint đã train sẵn (`gwnet_dynamic_epoch_86_1.60.pth`) với `strict=False` để bỏ qua các key không khớp.

**3. Đánh giá trên tập Test**
- Chạy inference toàn bộ test set, inverse transform predictions về đơn vị tốc độ gốc (mph).
- Tính **MAE, MAPE, RMSE** cho từng horizon (bước 1–12, tương ứng 5–60 phút).
- Tích lũy sai số tuyệt đối theo từng **node** và từng **horizon** để phân tích sau.
- Xuất hai file CSV:
  - `mae_by_node.csv` — MAE trung bình của 325 cảm biến, sắp xếp giảm dần.
  - `mae_by_horizon.csv` — MAE theo từng bước dự báo.

**4. Phân tích lỗi theo Node (Spatial EDA)**
- Tính **node degree** từ adjacency matrix (số kết nối không gian mỗi cảm biến).
- Load dữ liệu tốc độ gốc `pems-bay.h5` để tính:
  - `traffic_mean`: tốc độ trung bình của cảm biến.
  - `traffic_std`: độ biến động tốc độ (proxy cho mức độ phức tạp giao thông).
  - `zero_ratio`: tỷ lệ tín hiệu bị thiếu / bằng 0.
- Vẽ: phân bố MAE tổng thể, scatter plot `traffic_std vs MAE`, scatter plot `mean speed vs MAE` tô màu theo node degree.
- So sánh đặc trưng nhóm **10% cảm biến dự báo kém nhất** vs **10% tốt nhất**.

**5. Phân tích lỗi theo Horizon (Temporal EDA)**
- Tính: `mae_diff` (tăng tuyệt đối mỗi bước), `error_growth_rate_pct` (tốc độ tăng %), `cumulative_degradation_pct` (tổng suy giảm so với bước 1).
- Vẽ: đường cong tích lũy lỗi theo thời gian, bar chart tăng trưởng lỗi mỗi 5 phút.
- Phân đoạn: short-term (5–15 min), mid-term (20–40 min), long-term (45–60 min).

### Kết quả chính

| Horizon | MAE    | MAPE   | RMSE   |
|---------|--------|--------|--------|
| 5 min   | 0.8523 | 1.65%  | 1.5059 |
| 30 min  | 1.6402 | 3.74%  | 3.3842 |
| 60 min  | 1.9413 | 4.61%  | 4.0158 |
| **Avg** | **1.5766** | **3.57%** | **3.2056** |

Top 5 cảm biến dự báo kém nhất (theo MAE): node 134, 146, 10, 133, 45 — thường là các nút giao thông phức tạp hoặc có tín hiệu nhiễu cao.

### Yêu cầu môi trường

Notebook được thiết kế chạy trên **Kaggle** với dữ liệu từ các Kaggle Datasets:
- `nhl875/pems-bay` — dữ liệu tốc độ PEMS-BAY.
- `izmildqnh/metr-la` — chứa `adj_mx_bay.pkl`.
- `nhihhunhth/graohwavenet` — checkpoint model đã huấn luyện.
