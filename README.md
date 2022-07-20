# Social ODE

This is Latent ODE training. More codes will be released soon.

## Requirements
```Shell
conda create -n SocialODE python=3.8
conda activate SocialODE
conda install pytorch==1.7.0 torchvision==0.8.0 torchaudio==0.7.0 cudatoolkit=11.0 -c pytorch
conda install matplotlib
pip install torchdiffeq
```

## Dataset preparation
Download [inD](https://www.ind-dataset.com/), [highD](https://www.highd-dataset.com/) and [rounD](https://www.round-dataset.com/) dataset. Unzip and put the datasets within the directory `./data`.

## Training
``` Shell
python train.py -train_data_dir /data/inD/data/&
```

## Acknowledgement

Part of the codes are modified from [torchdiffeq](https://github.com/pytorch/vision/blob/master/torchvision/models) and [cvpr_dNRI](https://github.com/cgraber/cvpr_dNRI). 
We sincerely thank for their great work.
