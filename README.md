## Final Project: MixupCrop
Representation: [video](https://drive.google.com/file/d/1BMUldS8roIFo2q5fXChp_Qa58jGVcapa/view?usp=sharing)


Slides: [CurriculumSSL](https://docs.google.com/presentation/d/1D9RNMQuaFLVkM1PEpTDSVH0_GD8FeibwTX2aOg4j5FI/edit?usp=sharing)

### Project Description
Contrastive learning has been one of the most successful approaches for
self-supervised learning and has gained more attention ever since. Recent works showed the
importance of crafting good positive pairs, especially for contrastive methods that only leverage
positive pairs. Most previous works implement the standard data augmentation with random
cropping and did not take into account the congruence within positive pairs. Some works have
tried to tackle this problem at the cost of computational overhead. In this work, we propose
CurriculumSSL, an efficient yet effective method that produces better positive
image pairs without hindering the training speed. Specifically, we use MixUp as a stronger data
augmentation method and leverage curriculum learning to overcome the difficulty of additional
perturbations on the input image. Empirical results show that our method achieved consistent
improvements on multiple contrastive learning frameworks on CIFAR datasets.


### Dependencies

If you don't have python 3 environment:
```
conda create -n simsiam python=3.8
conda activate simsiam
```
Then install the required packages:
```
pip install -r requirements.txt
```

### Run SimSiam

```
CUDA_VISIBLE_DEVICES=0 python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress
```
The data folder `../Data/` should look like this:
```
➜  ~ tree ../Data/
├── cifar-10-batches-py
│   ├── batches.meta
│   ├── data_batch_1
│   ├── ...
└── stl10_binary
    ├── ...
```
```
Training: 100%|#################################################################| 800/800 [11:46:06<00:00, 52.96s/it, epoch=799, accuracy=90.3]
Model saved to /root/.cache/simsiam-cifar10-experiment-resnet18_cifar_variant1.pth
Evaluating: 100%|##########################################################################################################| 100/100 [08:29<00:00,  5.10s/it]
Accuracy = 90.83
Log file has been saved to ../logs/completed-simsiam-cifar10-experiment-resnet18_cifar_variant1(2)
```
To evaluate separately:
```
CUDA_VISIBLE_DEVICES=4 python linear_eval.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simsiam_cifar_eval.yaml --ckpt_dir ~/.cache/ --hide_progress --eval_from ~/simsiam-cifar10-experiment-resnet18_cifar_variant1.pth

creating file ../logs/in-progress_0111061045_simsiam-cifar10-experiment-resnet18_cifar_variant1
Evaluating: 100%|##########################################################################################################| 200/200 [16:52<00:00,  5.06s/it]
Accuracy = 90.87
```
![simsiam-cifar10-800e](simsiam-800e90.83acc.svg)

>`export DATA="/path/to/your/datasets/"` and `export LOG="/path/to/your/log/"` will save you the trouble of entering the folder name every single time!

### Run SimCLR

```
CUDA_VISIBLE_DEVICES=1 python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/simclr_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress
```

### Run BYOL
```
CUDA_VISIBLE_DEVICES=2 python main.py --data_dir ../Data/ --log_dir ../logs/ -c configs/byol_cifar.yaml --ckpt_dir ~/.cache/ --hide_progress
```

### TODO

- convert from data-parallel (DP) to distributed data-parallel (DDP)
- create PyPI package `pip install simsiam-pytorch`


If you find this repo helpful, please consider star so that I have the motivation to improve it.



