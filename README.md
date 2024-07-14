# Dynamic Neural Radiance Field From Defocused Monocular Video

[Xianrui Luo](https://scholar.google.com/citations?hl=en&user=tUeWQ5AAAAAJ)<sup>1</sup>,
[Huiqiang Sun](https://scholar.google.com/citations?user=CafUdpEAAAAJ&hl=en)<sup>1</sup>,
[Juewen Peng](https://scholar.google.com/citations?hl=en&user=fYC6lCUAAAAJ)<sup>2</sup>,
[Zhiguo Cao](http://english.aia.hust.edu.cn/info/1085/1528.htm)<sup>1</sup>

<sup>1</sup>Huazhong University of Science and Technology, <sup>2</sup>Nanyang Technological University

<p align="center">
<img src=https://github.com/notorious-eric/D2RF/blob/main/quantitative_github.png/>
</p>

## [Data](https://drive.google.com/drive/folders/1nUNWrFLKmK2g-ClJ4Nd9OGuxhYeu6Sv7?usp=sharing) | [Paper](https://arxiv.org/abs/2407.05586) 

This repository is the official PyTorch implementation of the ECCV 2024 paper "Dynamic Neural Radiance Field From Defocused Monocular Video".




## Usage 
### Dependency
- numpy
- scikit-image
- imageio
- configargparse
- opencv-python
- kornia
- torch
- torchvision
### Training
```
python run_nerf.py --config configs/xxx.txt
```

We use the left view for training and the right view for evaluation.
```
python run_nerf.py --config configs/xxx.txt --render_test
```

The checkpoints are in [Google Drive](https://drive.google.com/drive/folders/1_2FjsZnYlXcfPQfSC39t4BLXh34Tse_X).

There are a total of 8 scenes used in the paper. You can download all the data in [here](https://drive.google.com/drive/folders/1nUNWrFLKmK2g-ClJ4Nd9OGuxhYeu6Sv7?usp=sharing).
## Acknowledge
This source code is derived from [Deblur-NeRF](https://github.com/limacv/Deblur-NeRF) and [NSFF](https://github.com/zhengqili/Neural-Scene-Flow-Fields). We appreciate the effort of the contributors to these repositories.
