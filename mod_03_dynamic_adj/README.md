# mod_03_dynamic_adj

Biến thể Graph WaveNet theo hướng dùng **dynamic adaptive adjacency matrix**.

## Nguồn gốc
Module này được dựng lại từ notebook tham chiếu:
- `ref-artifacts/mod_03/mod_03_dynamic_adj_metr_la.ipynb`

Notebook gốc patch trực tiếp Graph WaveNet baseline để thay adaptive adjacency tĩnh bằng adaptive adjacency động theo batch.

## Ý tưởng chính
So với adaptive adjacency chuẩn:

- **Baseline adaptive adjacency**: `SoftMax(ReLU(E1 @ E2))`
- **Dynamic adaptive adjacency**: thêm tín hiệu phụ thuộc input hiện tại qua tensor đặc trưng trung bình theo thời gian

Trong code này, thành phần đó là `DynamicAdaptiveAdj`:
- lấy đặc trưng từ hidden state hiện tại
- chiếu về embedding space
- cộng vào `nodevec1/nodevec2`
- sinh ma trận adjacency động theo từng batch

## Model variants
Dùng `--model_variant`:

### `dynamic_adj`
Tương ứng notebook gốc.
- giữ Graph WaveNet chuẩn
- thay adaptive adjacency tĩnh bằng dynamic adaptive adjacency

### `baseline`
Thêm để đối chiếu nhanh trong cùng package.
- dùng adaptive adjacency tĩnh chuẩn Graph WaveNet
- không bật dynamic adjacency

## Cấu trúc
- `model.py`: Graph WaveNet + dynamic adjacency
- `engine.py`: training/eval wrapper
- `train.py`: train entry point
- `test.py`: test entry point

## Train
```bash
python -m mod_03_dynamic_adj.train \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --save ./garage/mod03_dynamic \
  --model_variant dynamic_adj
```

## Test
```bash
python -m mod_03_dynamic_adj.test \
  --device cuda:0 \
  --data data/METR-LA \
  --adjdata data/sensor_graph/adj_mx.pkl \
  --adjtype doubletransition \
  --gcn_bool \
  --addaptadj \
  --num_nodes 207 \
  --checkpoint ./garage/mod03_dynamic_best.pth \
  --model_variant dynamic_adj
```

## Lưu ý
- `dynamic_adj` cần `--addaptadj`.
- Notebook gốc đang hard-code METR-LA; module này được đưa về dạng reusable entry point như các `mod_*` khác.
