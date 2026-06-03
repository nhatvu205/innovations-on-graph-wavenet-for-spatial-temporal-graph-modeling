### **1\. Lớp Tích chập Đồ thị (Graph Convolution Layer \- GCN)**

Đây là module quan trọng nhất để trích xuất các đặc trưng không gian của các nút dựa trên cấu trúc đồ thị.

* **Ma trận Kề Tự thích nghi (Self-adaptive Adjacency Matrix):** Đây là đóng góp cốt lõi của bài báo. Thay vì chỉ dựa vào một ma trận kề cố định có sẵn, Graph WaveNet tự học một ma trận kề $\\tilde{A}\_{adp}$ từ dữ liệu thông qua các phép nhúng nút (node embeddings). Điều này cho phép mô hình khám phá các mối quan hệ ẩn mà cấu trúc đồ thị ban đầu có thể bỏ sót.  
* **Tích chập Khuếch tán (Diffusion Convolution):** Mô hình mô phỏng quá trình khuếch tán thông tin trên đồ thị theo cả hai hướng thuận và nghịch để nắm bắt các phụ thuộc không gian phức tạp.  
* **Tổng hợp:** Kết quả cuối cùng là sự kết hợp giữa thông tin từ cấu trúc đồ thị định sẵn (nếu có) và ma trận kề tự học được.

### **2\. Lớp Tích chập Thời gian (Temporal Convolution Layer \- TCN)**

Module này chịu trách nhiệm bắt kịp các xu hướng thay đổi theo thời gian của từng nút.

* **Tích chập Nhân quả Giãn nở (Dilated Causal Convolution):** Khác với các mô hình RNN truyền thống, TCN sử dụng các lớp tích chập với các hệ số giãn nở (dilation factors) tăng dần. Điều này giúp mở rộng vùng tiếp nhận (receptive field) theo cấp số nhân, cho phép mô hình xử lý các chuỗi thời gian rất dài mà không bị bùng nổ hay biến mất gradient.  
* **Cơ chế Gated TCN:** Mô hình áp dụng một cơ chế cổng tương tự như trong LSTM để kiểm soát luồng thông tin. Nó bao gồm hai nhánh tích chập song song: một nhánh đi qua hàm kích hoạt tanh (để lọc thông tin chính) và một nhánh đi qua hàm sigmoid (để đóng vai trò là cổng lọc).

### **3\. Cấu trúc Khung (Framework) và Các Kết nối**

Toàn bộ mô hình được xây dựng bằng cách xếp chồng nhiều lớp không-thời gian.

* **Lớp Không-Thời gian (Spatial-Temporal Layer):** Mỗi lớp bao gồm một module Gated TCN để xử lý thời gian, theo sau là một module GCN để xử lý không gian. Việc xếp chồng giúp mô hình học được các phụ thuộc ở các mức độ chi tiết thời gian khác nhau (ví dụ: lớp dưới học thông tin ngắn hạn, lớp trên học thông tin dài hạn).  
* **Kết nối Dư (Residual Connections) và Kết nối Nhảy (Skip Connections):**  
  * Mỗi lớp không-thời gian đều có các kết nối dư để tránh hiện tượng suy giảm hiệu năng khi mô hình sâu hơn.  
  * Tất cả các đầu ra từ các lớp trung gian đều được chuyển tiếp đến lớp đầu ra thông qua kết nối nhảy để tổng hợp thông tin đa cấp.

### **4\. Lớp Đầu ra (Output Layer)**

Lớp này bao gồm hai lớp tuyến tính (Linear layers) kết hợp với các hàm kích hoạt ReLU. Thay vì dự báo từng bước một cách đệ quy (vốn dễ gây sai số tích lũy), Graph WaveNet tạo ra toàn bộ chuỗi dự báo cho nhiều bước thời gian tiếp theo trong một lần chạy duy nhất, giúp tăng tốc độ tính toán và độ chính xác.

### **Ý nghĩa tổng thể**

Sự kết hợp giữa **TCN giãn nở** và **GCN thích nghi** cho phép Graph WaveNet không chỉ hiểu được dòng chảy thời gian mà còn tự động "vẽ" lại bản đồ quan hệ giữa các nút trong hệ thống, từ đó đạt được hiệu quả vượt trội trong các nhiệm vụ như dự báo tốc độ giao thông hay phân tích chuỗi thời gian trên đồ thị.

