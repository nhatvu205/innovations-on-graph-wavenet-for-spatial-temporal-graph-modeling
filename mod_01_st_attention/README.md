# mod_01_st_attention

Biến thể Graph WaveNet theo hướng thêm attention vào mô hình baseline.

## Mục tiêu
Module này dùng để thử nghiệm attention trên 2 nhánh:
- **Spatial attention**: sinh thêm support động theo attention giữa các node
- **Temporal attention**: self-attention theo trục thời gian cho từng node

## Thành phần chính
- `model.py`: model Graph WaveNet variant
- `engine.py`: training/eval wrapper
- `train.py`: train entry point
- `test.py`: test entry point
- `GatedTCN.py`: tách riêng phần gated temporal conv
- `DiffusionGraphConv.py`: diffusion graph convolution
- `SelfAdaptiveAdjacency.py`: adaptive adjacency matrix

## Khác gì so với baseline
Ngoài Graph WaveNet chuẩn (GCN + adaptive adjacency), module này hỗ trợ:
- `--spatial_attention`
- `--temporal_attention`
- `--model_variant baseline|spatial|temporal|spatiotemporal`

Trong `model.py` hiện tại, các attention branch đã được tích hợp trực tiếp vào `gwnet`.

## Cách train
Ví dụ baseline trong module này:

```bash
python -m mod_01_st_attention.train \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --save ./garage/mod01_baseline \
  --model_variant baseline
```

Ví dụ bật cả spatial + temporal attention:

```bash
python -m mod_01_st_attention.train \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --save ./garage/mod01_spatiotemporal \
  --model_variant spatiotemporal
```

## Cách test
```bash
python -m mod_01_st_attention.test \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --checkpoint ./garage/mod01_spatiotemporal_best.pth \
  --model_variant spatiotemporal
```

## Ghi chú
- Module này là một hướng cải tiến riêng, không phải notebook ablation family mới.
- Các file component rời hiện chủ yếu mang tính tách logic; `model.py` vẫn là nơi assemble chính.
