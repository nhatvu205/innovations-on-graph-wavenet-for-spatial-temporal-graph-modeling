# mod_02_efficiency_family

Nhóm biến thể tập trung vào **giảm thời gian training / inference** mà không làm tăng lỗi quá nhiều.

## Nguồn gốc
Module này được dựng lại từ 2 notebook tham chiếu:
- `ref-artifacts/mod_02/metr-la-graph-wavenet-optimized.ipynb`
- `ref-artifacts/mod_02/pems-bay-graph-wavenet-optimized.ipynb`

Hai notebook không phải cùng một kỹ thuật duy nhất; chúng là **2 hướng tối ưu efficiency khác nhau**, nên module này gom chúng thành một family với 2 variant.

## Variant 1 — `static_adj_opt`
Tương ứng `ref-artifacts/mod_02/metr-la-graph-wavenet-optimized.ipynb`

Kỹ thuật chính:
- **Low-rank adaptive embedding**: giảm chiều embedding từ 10 xuống `emb_dim`
- **Top-k sparsification**: chỉ giữ `k` cạnh mạnh nhất mỗi node
- **Adjacency caching**: không tính lại adaptive adjacency ở mọi forward

Mục tiêu:
- giảm chi phí tính adaptive adjacency
- giảm overhead khi train/eval
- giữ gần baseline nhất về kiến trúc

## Variant 2 — `attn_skipagg_opt`
Tương ứng `ref-artifacts/mod_02/pems-bay-graph-wavenet-optimized.ipynb`

Kỹ thuật chính:
- **CausalWindowAttnTCN** thay temporal conv chuẩn
- **SkipAggregationAttn** để aggregate skip features giữa các layer
- **Per-module learning rate** cho optimizer

Mục tiêu:
- tối ưu biểu diễn temporal/skip path
- giữ mô hình gọn hơn full-transformer style
- tinh chỉnh tốc độ hội tụ bằng param groups riêng

## Cấu trúc
- `model.py`: family model + helper optimizer
- `engine.py`: training/eval wrapper
- `train.py`: train entry point
- `test.py`: test entry point

## Train
### Static adaptive optimization
```bash
python -m mod_02_efficiency_family.train \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --save ./garage/mod02_static_adj_opt \
  --model_variant static_adj_opt
```

### Attention + skip aggregation optimization
```bash
python -m mod_02_efficiency_family.train \
  --device cuda:0 \
  --data data/PEMS-BAY \
  --adjdata data/sensor_graph/adj_mx_bay.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 325 \
  --save ./garage/mod02_attn_skipagg_opt \
  --model_variant attn_skipagg_opt
```

## Test
```bash
python -m mod_02_efficiency_family.test \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --checkpoint ./garage/mod02_static_adj_opt_best.pth \
  --model_variant static_adj_opt
```

## Lưu ý
- Cả 2 variant đều giả định có `--addaptadj`.
- `static_adj_opt` gần baseline hơn về kiến trúc.
- `attn_skipagg_opt` thay đổi temporal path và skip aggregation rõ rệt hơn.
