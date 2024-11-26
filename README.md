# DJAA

Pytorch implementation of paper ["Anti-Forgetting Adaptation for Unsupervised Person Re-identification
"](https://arxiv.org/abs/2411.14695).

## Installation

```shell
conda create -n djaa python=3.7
source activate djaa 
pip install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install h5py six Pillow scipy scikit-learn metric-learn tqdm faiss-gpu
python setup.py develop
```

## Prepare Datasets

```shell
cd examples && mkdir data
```
Download the raw datasets [Market-1501](http://www.liangzheng.org/Project/project_reid.html), 
[Cuhk-Sysu](http://www.ee.cuhk.edu.hk/~xgwang/PS/dataset.html), 
[MSMT17(Google Drive)](https://drive.google.com/drive/folders/11I7p0Dr-TCC9TnvY8rWp0B47gCB3K0T4), 
[VIPeR](http://users.soe.ucsc.edu/~manduchi/VIPeR.v1.0.zip), 
[PRID2011](https://www.tugraz.at/institute/icg/research/team-bischof/lrs/downloads/PRID11/), 
[GRID](http://personal.ie.cuhk.edu.hk/~ccloy/files/datasets/underground_reid.zip), 
[iLIDS](http://www.eecs.qmul.ac.uk/~jason/data/i-LIDS_Pedestrian.tgz), 
[CUHK01](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html), 
[CUHK02](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html), 
[SenseReID](https://drive.google.com/file/d/0B56OfSrVI8hubVJLTzkwV2VaOWM/view), 
[CUHK03](http://www.ee.cuhk.edu.hk/~xgwang/CUHK_identification.html) and
[3DPeS](https://aimagelab.ing.unimore.it/imagelab/page.asp?IdPage=16), 
and then unzip them under the directory like
```
DJAA/examples/data
├── market1501
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   └── query/
├── cuhk-sysu
│   └── CUHK-SYSU
│       ├── Image/
│       └── annotation/
├── msmt17
│   └── MSMT17_V2
│       └── market_style/
│           ├── bounding_box_train/
│           ├── bounding_box_test/
│           └── query/
├── personx
│   ├── bounding_box_train/
│   ├── bounding_box_test/
│   └── query/
├── viper
│   └── VIPeR
├── prid2011
│   └── prid_2011
├── grid
│   └── underground_reid
├── ilids
│   └── i-LIDS_Pedestrian
├── cuhk01
│   └── campus
├── cuhk02
│   └── Dataset
├── sensereid
│   └── SenseReID
├── cuhk03
│   └── cuhk03_release
└── 3dpes
    └── 3DPeS
```


## Train:
Train DJAA on default order (Market to Cuhk-Sysu to MSMT17). 
The results reported in the paper were obtained with **4 GPUs**.
#### Unsupervised lifelong training
```shell
sh fully_unsupervised.sh
```

## Test:
```shell
python examples/test.py --init examples/logs/step3.pth.tar
```

## Citation
If you find this project useful, please kindly star our project and cite our paper.
```bibtex
@ARTICLE{10742299,
  author={Chen, Hao and Bremond, Francois and Sebe, Nicu and Zhang, Shiliang},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Anti-Forgetting Adaptation for Unsupervised Person Re-Identification}, 
  year={2024},
  volume={},
  number={},
  pages={1-16},
  keywords={Adaptation models;Data models;Feature extraction;Contrastive learning;Prototypes;Training;Incremental learning;Cameras;Annotations;Training data;Backward compatible representation;contrastive learning;domain generalization;incremental learning;re-identification},
  doi={10.1109/TPAMI.2024.3490777}}