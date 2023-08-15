# GSSP: Graph-based Structural Sketch Parsing

### Installation

1. Create a new environment and install the libraries:
```
conda create -n gssp python=3.7
conda activate gssp
```
2. install `pytorch`, `torch_geometric` :
```
pip install torch==1.12.0+cu102 torchvision==0.13.0+cu102 torchaudio==0.12.0 --extra-index-url https://download.pytorch.org/whl/cu102
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.12.0+cu102.html
```
3. Install diffvg:
```
git clone https://github.com/BachiLi/diffvg
cd diffvg
git submodule update --init --recursive
python setup.py install
```
4. Reference `requirements.txt` to install other dependent libraries


### Train

1. link to dataset toolkit [DatasetToolkit](https://github.com/YufengLii/SUUDataset)
download ready for train [dataset](https://drive.google.com/file/d/1WhvipuYH20O7Pk-rDHyLorVFm6zh31j3/view?usp=share_link)

2. node trainning
```
python train_node.py --config_file=train_junc.yaml --gpu=1 --last_epoch=-1 
```
change the `Data_Root` in train_junc.yaml according your path

3. LP training

```
python train_graph.py --config_file=train_graph.yaml --gpu=1 --last_epoch=19
```
