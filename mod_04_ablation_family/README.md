# mod_04_ablation_family

Nhóm ablation được dựng lại từ 3 notebook:
- `ref-artifacts/mod_04/mod_04_full_model.ipynb`
- `ref-artifacts/mod_04/mod_04_wo_adaptive.ipynb`
- `ref-artifacts/mod_04/mod_04_wo_attention.ipynb`
- `ref-artifacts/mod_04/mod_04_wo_dynamic.ipynb`
- `ref-artifacts/mod_04/mod_04_wo_st_attention.ipynb`

## Mục tiêu
Module này gom một family kiến trúc chung để so sánh ảnh hưởng của từng thành phần cải tiến so với full model.

## Thành phần chính của family này
- **CausalWindowAttnTCN**: thay temporal conv chuẩn bằng causal window attention TCN
- **DynamicAdaptiveAdj**: adaptive adjacency động theo batch
- **LocalCausalSkipAttention**: local causal skip attention trên skip connection
- **SkipAggregationAttn**: attention để aggregate skip features giữa nhiều layer

> Lưu ý: trong nhóm notebook này có **hai khái niệm attention khác nhau**:
> 1. `LocalCausalSkipAttention`
> 2. `SkipAggregationAttn` / spatial-temporal attention ở bước gom skip connections

## Model variants
Dùng cờ `--model_variant`:

### `full`
Tương ứng `ref-artifacts/mod_04/mod_04_full_model.ipynb`
- có dynamic adaptive adjacency
- có local causal skip attention
- có causal-window attention TCN

### `wo_adaptive`
Tương ứng `ref-artifacts/mod_04/mod_04_wo_adaptive.ipynb`
- bỏ adaptive adjacency
- bỏ skip attention, dùng simple sum skip
- giữ causal-window attention TCN

### `wo_attention`
Tương ứng `ref-artifacts/mod_04/mod_04_wo_attention.ipynb`
- giữ adaptive adjacency
- bỏ local causal skip attention, dùng simple sum skip
- giữ causal-window attention TCN

### `wo_dynamic`
Tương ứng `ref-artifacts/mod_04/mod_04_wo_dynamic.ipynb`
- bỏ dynamic adaptive adjacency
- dùng static adaptive adjacency với low-rank + top-k + cache
- dùng SkipAggregationAttn để gom skip connections
- giữ causal-window attention TCN

### `wo_st_attention`
Tương ứng `ref-artifacts/mod_04/mod_04_wo_st_attention.ipynb`
- giữ dynamic adaptive adjacency
- bỏ spatial-temporal attention kiểu skip aggregation
- dùng simple sum skip
- giữ causal-window attention TCN

## Cấu trúc
- `model.py`: define toàn bộ family model
- `engine.py`: training/eval wrapper
- `train.py`: train entry point
- `test.py`: test entry point

## Train
### Full model
```bash
python -m mod_04_ablation_family.train \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --save ./garage/mod04_full \
  --model_variant full
```

### W/o adaptive
```bash
python -m mod_04_ablation_family.train \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --num_nodes 207 \
  --save ./garage/mod04_wo_adaptive \
  --model_variant wo_adaptive
```

### W/o local causal skip attention
```bash
python -m mod_04_ablation_family.train \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --save ./garage/mod04_wo_attention \
  --model_variant wo_attention
```

### W/o dynamic
```bash
python -m mod_04_ablation_family.train \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --save ./garage/mod04_wo_dynamic \
  --model_variant wo_dynamic
```

### W/o spatial-temporal attention
```bash
python -m mod_04_ablation_family.train \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --save ./garage/mod04_wo_st_attention \
  --model_variant wo_st_attention
```

## Test
```bash
python -m mod_04_ablation_family.test \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --checkpoint ./garage/mod04_full_best.pth \
  --model_variant full
```

## Lưu ý
- `wo_adaptive` trong notebook gốc không chỉ bỏ adaptive adjacency mà còn bỏ luôn skip attention. README này phản ánh đúng logic code hiện tại.
- Nếu dùng variant cần adaptive adjacency (`full`, `wo_attention`, `wo_dynamic`, `wo_st_attention`) thì phải truyền `--addaptadj`.
- `wo_attention` là bỏ **local causal skip attention**.
- `wo_st_attention` là bỏ **skip aggregation / spatial-temporal attention** theo notebook `model-without-attention.ipynb`.
