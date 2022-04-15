# 2022数字中国算法赛题 【卫星应用赛题：海上船舶智能检测】赛题方案

## 队伍名称：default(鹰眼) 

testA: 0.9707  testB: 0.9712

## 1. 解题方案整体思路：

+ Detector: 
    - Backbone: Cascade + Convnvext or Swin Transformer
    - RCNN Head:
        - box head:
            - 4Conv+1FC
            - GIou Loss
    - Post-process: nms + min-max-score filter
+ Data augmentation:
    - custom mosaic （close in final phase）
    - multi-scale training and testing
    - all other methods such as flip/rotation are not work
+ Model ensemble:
    - wbf (提升很小，单模已经97+)

## 2. 运行环境
+ 系统 ubuntu 18
+ python3.7 或 3.8
+ torch 1.8 或者 1.7.1
+ cuda 11.1
+ cudnn 8
+ GPU： 2080Ti x 4 
+ 特殊依赖：
    - mmcv-full=1.4.0

## 3. 文件目录说明
进行训练测试时需要的具体的文件如下：
训练好的模型文件可以从[百度网盘](https://pan.baidu.com/s/1h4sP15batarT0p47sCZ1gw?pwd=4cy3)下载
```
|-- user_data
    |-- work_dirs/ 包含3个训练好的模型的模型文件
    |-- annotations 包含推理或者训练过程中转换的coco格式文件，其中文件代码自动生成
    |-- pretrained 包含模型训练需要的公开的coco预训练模型，由于模型体积大，这里只给出了开源github的下载链接(来自convnext和swin官方)，需要下载后放在此目录下
    |-- 剩余文件夹均为训练或者预测过程中的中间文件
|-- prediction_result
    |-- result.json   按照比赛提交格式生成的提交文件
|-- code
    |-- 代码，包括训练和测试全部的代码
    |-- 其中 run.sh 为一键推理命令， train.sh 为一键训练命令
|-- raw_data
    |-- 比赛的数据集文件（官网数据下载解压后的格式, 需要将官方数据放置在这里）：
    |-- 训练数据目录：/data/raw_data/training_dataset/A/
    |-- 测试数据目录：/data/raw_data/test_dataset/测试集/
```
## 4. 推理得到B榜提交结果的运行说明
+ 按照上述过程下载3个模型文件并放置于/data/user_data/work_dirs
+ 运行 run.sh 一键推理命令
+ 推理过程中会使用 /data/user_data/work_dirs 目录下的训练好的模型文件


## 5. 模型训练的运行说明
+ 首先根据data/user_data/pretrained 目录下给出的下载地址文件，下载两个开源预训练模型
+ 下载上述3个预训练模型后，放置在data/user_data/pretrained目录下，最终 data/user_data/pretrained 目录下应该有2个.pth后缀的预训练模型文件：
    - cascade_mask_rcnn_convnext_base_22k_3x.pth
    - cascade_mask_rcnn_swin_small_patch4_window7.pth
+ 运行 train.sh 一键训练命令


## 6. Author

rill: 18813124313@163.com
