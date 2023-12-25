# UltraLiDAR

Unofficial implementation of [UltraLiDAR](https://openaccess.thecvf.com/content/CVPR2023/papers/Xiong_Learning_Compact_Representations_for_LiDAR_Completion_and_Generation_CVPR_2023_paper.pdf). We add the code to support training and generation on the [nuScenes](https://nuscenes.org) and [Waymo](https://waymo.com/open/) datasets.

## Installation instructions

Following https://mmdetection3d.readthedocs.io/en/v1.0.0rc1/getting_started.html

**a. Create a conda virtual environment and activate it.**
```shell
conda create -n ultralidar python=3.8 -y
conda activate ultralidar
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Recommended torch==1.10
```

**c. Install mmcv-full, mmdet and mmseg.**
```shell
pip3 install openmim
mim install mmcv-full==1.5.0
mim install mmdet==2.28.2
mim install mmsegmentation==0.30.0
```


**d. Install mmdet3d from source code and other packages.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git
cd mmdetection3d
git checkout v1.0.0rc1 # Other versions may not be compatible.
pip install -v -e .
```
**e. Clone UltraLIDAR.**
```
git clone git@github.com:myc634/UltraLiDAR_nusc_waymo.git
cd UltraLiDAR_nusc_waymo
pip install -r requirements.txt
```


## Preparing Dataset

Please follow the official MMDetection3D dataset preparation [guidance](https://mmdetection3d.readthedocs.io/en/v0.17.1/data_preparation.html)

The final data structure should be:
```
UltraLiDAR_nusc_waymo
├── datasets
│   ├── kitti-360
│   │── nuScenes
│   │── waymo
```

For Kitti360 dataset, please run
```
python tools/data_converter/kitti360_converter.py
```

For nuScenes dataset, please run
```
python tools/data_converter/nuscenes_converter.py --data-root datasets/nuScenes/
```
to generate the `pkl` file which fits the MMDetection3D framework.

For Waymo Open Dataset, please run: `cd mmdetection3d` and follow the official instruction of [MMDetection3D Data Preparation pipeline](https://mmdetection3d.readthedocs.io/en/v1.0.0rc0/data_preparation.html) to convert waymo format into kitti format for training


## Training

For Stage 1 training:
```sh 
./tools/dist_train.sh configs/ultralidar_kitti360.py 8
```

For Stage 2 training, please select the best result in Stage 1, copy the dir and paste it on `configs/ultralidar_kitti360_gene.py: 238` and run:
```sh 
./tools/dist_train.sh configs/ultralidar_kitti360_gene.py 8
```

## Eval

### Step0: 
```sh
./tools/dist_test.sh configs/ultralidar_kitti360_static_blank_code.py ${PATH_TO_WEIGHTS} --eval "mIoU"
```
Run this command to calculate the blank code, this will generate a ```blank_code.pkl``` file which contains 20 blank codes.

### Step1: 
```sh
./tools/dist_test.sh configs/ultralidar_kitti360_gene.py ${PATH_TO_WEIGHTS} --eval "mIoU"
```
After running this command, there will be a file named `ultralidar_samples` containing 2000 generated samples with `.ply` format.

### Step2: 

First:
```
export KITTI360_DATASET=datasets/kitti-360
```

For evaluation MMD, please run:
```
python mmd.py
```

For evaluation JSD, please run:
```
python jsd.py
```

Here, we provide a set of [2k](https://drive.google.com/file/d/1N-rzs6jSOOlKLAGRj-w4WtYEBQyfJuEm/view?usp=sharing) and [10k](https://drive.google.com/file/d/1G1AlXWmAV9wWpucnAl_M3u6Eu7q8QIE9/view?usp=sharing) generated LiIDAR point cloud samples, utilizing the [weightes](https://drive.google.com/file/d/1jDk1h9Xh3i7pCwqB_C1DvqjReL6Ca60D/view?usp=sharing) from our implementated UltraLiDAR model.

|   | MMD    | JSD     |
| -------- | -------- | -------- |
| KITTI360 | 2.25e-4 | 0.0989 |

