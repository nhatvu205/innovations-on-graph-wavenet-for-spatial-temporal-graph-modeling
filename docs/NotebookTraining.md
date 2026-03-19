# Training on Colab or Kaggle

## Colab / Kaggle Steps

1. Open a new notebook with GPU enabled.
2. Clone the repo:

```bash
!git clone https://github.com/nhatvu205/innovations-on-graph-wavenet-for-spatial-temporal-graph-modeling.git
%cd innovations-on-graph-wavenet-for-spatial-temporal-graph-modeling
```

3. Install dependencies:

```bash
!pip install -r requirements.txt
```

4. Place data in this layout. Click into this [link](https://drive.google.com/drive/folders/1bmXiXxCcGWqO16URDWoEHKYlb29-9od1?usp=sharing) to download the data folder:


```
data/
  PEMS-BAY/
    train.npz
    val.npz
    test.npz
  METR-LA/
    train.npz
    val.npz
    test.npz
  sensor_graph/
    adj_mx.pkl
    adj_mx_bay.pkl
```

5. Train:

a) METR-LA

```bash
!python train.py --device cuda:0 --data data/METR-LA --adjdata data/sensor_graph/adj_mx.pkl --adjtype doubletransition --gcn_bool --addaptadj --randomadj --num_nodes 207 --save garage/metr
```
b) 
```bash
!python train.py --device cuda:0 --data data/PEMS-BAY --adjdata data/sensor_graph/adj_mx_bay.pkl --adjtype doubletransition --gcn_bool --addaptadj --randomadj --num_nodes 325 --save garage/bay
```