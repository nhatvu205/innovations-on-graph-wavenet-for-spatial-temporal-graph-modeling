# Kiến trúc Spatial-Temporal Attention trong Graph WaveNet

Tài liệu này tóm tắt chi tiết về cơ chế Spatial-Temporal Attention vừa được tích hợp vào mã nguồn, cách thức hoạt động, tác động đến luồng dữ liệu (code flow), kỳ vọng cải thiện (expected outcomes), và các vấn đề liên quan.

---

## 1. Cấu trúc và Cơ chế hoạt động

Cải tiến chia thành 2 nhánh độc lập và có thể bật/tắt linh hoạt thông qua cờ `--model_variant` (hoặc `--spatial_attention`, `--temporal_attention`).

### 1.1 Temporal Attention (Chú ý Không gian Thời gian)
- **Vị trí trong block:** Hoạt động **song song** với nhánh Local Gated TCN (Dilated Convolution).
- **Cơ chế:**
  - Dùng `nn.MultiheadAttention` áp dụng dọc theo **trục thời gian (T)** cho từng node riêng biệt.
  - Sinh ra một biểu diễn `temporal_out` có khả năng nhìn thấy toàn bộ chuỗi lịch sử đầu vào (tránh giới hạn receptive field của TCN cục bộ).
  - Sử dụng **Causal Mask** (tùy chọn bật/tắt qua `--no_temporal_causal_mask`) để đảm bảo mô hình tại bước $t$ không "nhìn trộm" thông tin từ $t+1$ trong cửa sổ đầu vào.
- **Fusion (Hợp nhất):** 
  - Mô hình dùng một tham số học được `temporal_fusion_logits` để tự động quyết định tỷ lệ kết hợp giữa nhánh cục bộ (TCN) và nhánh toàn cục (Temporal Attention):
    $$x_{fused} = \alpha \cdot x_{TCN} + (1 - \alpha) \cdot x_{Temporal}$$
    (với $\alpha = \text{sigmoid}(logits)$).

### 1.2 Dynamic Spatial Attention (Chú ý Không gian Động)
- **Vị trí trong block:** Tính toán trước khi đi qua nhánh Graph Convolution (GCN).
- **Cơ chế:**
  - Graph WaveNet gốc sử dụng ma trận kề tĩnh (Static Adjacency) và ma trận học được (Adaptive Adjacency) nhưng chúng **cố định cho mọi batch**.
  - **Cải tiến:** Tính toán ma trận kề **động (Dynamic Adjacency)** phụ thuộc trực tiếp vào dữ liệu hiện tại (`x` của lô hiện tại).
  - Đầu tiên trung bình hóa các đặc trưng trên trục thời gian để có `node_states`.
  - Đưa qua biến đổi tuyến tính (`spatial_q`, `spatial_k`) để sinh ra các vector truy vấn (Query) và khóa (Key) cho mỗi node.
  - Ma trận Attention động được tính bằng: `A_attn = Softmax(Q * K^T / sqrt(d))`.
- **Trộn với đồ thị tĩnh (Mixing):**
  - Hệ số $\beta = \text{sigmoid}(\text{spatial\_beta\_logit})$ được dùng để trộn ma trận tĩnh (trung bình của các `supports`) với ma trận động:
    $$A_{mix} = \beta \cdot A_{static} + (1 - \beta) \cdot A_{dynamic}$$
  - $A_{mix}$ sau đó được nối thêm vào danh sách `supports` để đưa vào GCN.

---

## 2. Tác động lên Code hiện tại

Sự thay đổi tập trung ở tầng **Model** và tầng **Pipeline Huấn luyện**:

### 2.1 File `src/model.py`
- Lớp `TemporalAttentionLayer`: Được thêm mới hoàn toàn để thực hiện Self-Attention trên chuỗi thời gian.
- Lớp `gwnet`: 
  - Khởi tạo thêm các Module `temporal_attn_layers`, tham số fusion `temporal_fusion_logits`.
  - Khởi tạo `spatial_q`, `spatial_k`, `spatial_beta_logit` để tính toán Attention động.
  - Sửa đổi hàm `forward()`: 
    - Tính `temporal_out` và trộn với `x` (kết quả của Gated TCN).
    - Tính `A_attn` và trộn để sinh ra đồ thị mới cho bước GCN.

### 2.2 File `src/engine.py` & `train.py` / `test.py`
- `engine.py` nhận thêm các tham số mới (`spatial_attention`, `temporal_attention`, số lượng heads, v.v.) và truyền cho `gwnet`.
- `train.py` / `test.py`: 
  - Thêm các Argument (cờ CLI) như `--model_variant`, `--temporal_attention_heads`,...
  - Thêm **chế độ cố định seed ngẫu nhiên** (`random.seed`, `torch.manual_seed`) để đảm bảo tái lập thí nghiệm.
  - Lưu và phân tích các chỉ số đánh giá theo từng horizon (đặc biệt là 3, 6, 12).
  - Tự động đo lường và báo cáo **Latency (thời gian training/inference)**.
  - Early stopping và ghi đè Checkpoint (`_best.pth`) để hỗ trợ khôi phục (Resume).

---

## 3. Kỳ vọng (Expected Outcomes)

1. **Với Temporal Attention:** 
   - **Cải thiện Long-horizon (Bước 6, 12):** TCN bị giới hạn bởi độ sâu của mạng (dilation). Temporal attention kết nối trực tiếp khoảng cách xa với $O(1)$, giúp mô hình nắm bắt xu hướng vĩ mô tốt hơn, qua đó giảm MAE/RMSE ở các bước dự đoán xa.
2. **Với Spatial Attention:**
   - **Tương thích với các sự kiện bất thường (Abnormal traffic):** Đồ thị giao thông thay đổi liên tục (giờ cao điểm, tắc nghẽn, tai nạn). Khả năng cấu trúc lại trọng số kết nối $A_{dynamic}$ theo từng batch giúp mô hình điều hướng tín hiệu GCN dựa trên bối cảnh thời gian thực thay vì trọng số cứng.
3. **Đánh đổi (Trade-offs):**
   - Sự đánh đổi dự kiến là tăng nhẹ thời gian huấn luyện và suy luận (thêm độ phức tạp tính toán $O(T^2)$ và $O(N^2)$ ở mỗi ST-Layer).

---

## 4. Các vấn đề liên quan và Lưu ý khi thực nghiệm

1. **Nguy cơ Overfitting:**
   - Thêm Attention đồng nghĩa với việc thêm nhiều tham số (Parameters). Đối với dataset nhỏ, nó có thể dẫn đến hiện tượng Overfitting nhanh hơn so với baseline.
   - *Cách giải quyết trong code:* Đã thêm cơ chế **Early Stopping** (`--early_stopping_patience 10`) và lưu `_best.pth` dựa vào Val Loss để chọn ra điểm hội tụ tốt nhất.

2. **Causal Mask trong Temporal Attention:**
   - Khi thực hiện Attention dọc theo thời gian trên input history (kích thước T=12), mặc định Causal Mask được bật.
   - Nếu tắt Causal Mask (`--no_temporal_causal_mask`), bước thời gian $t$ trong cửa sổ đầu vào sẽ được quyền "nhìn thấy" thông tin của bước $t+1$ (vẫn nằm trong quá khứ). Một số bài báo cho rằng trên input history, việc tắt Causal Mask cho performance tốt hơn vì toàn bộ input đã là quá khứ. Đã để sẵn cờ này để chạy Ablation so sánh.

3. **Tính toán Ma trận Không gian tĩnh (Base Adjacency):**
   - Trong code, `spatial_base_adj` được tính bằng cách trung bình các ma trận `supports` ban đầu. Hệ số học được $\beta$ sẽ khởi tạo ở mức gần cân bằng (~0.5 qua sigmoid của 0) và tự động nghiêng về Static Graph hay Dynamic Graph tùy theo hàm loss.

4. **Kế hoạch Ablation:**
   - Thí nghiệm cần chạy 4 biến thể với cùng chung một seed (`baseline`, `spatial`, `temporal`, `spatiotemporal`). Code đã có sẵn `run_ablation.py` hỗ trợ việc này để dễ dàng lập bảng so sánh theo MAE của horizon 3/6/12 và Inference Time.