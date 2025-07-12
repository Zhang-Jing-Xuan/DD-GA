# Gradient Amplification for Gradient Matching Based Dataset Distillation

## Highlights :sparkles:
- Label cycle shifting strategy produces informative gradient information. 
- The early exit mechanism alleviates matching the useless gradients.
- Ensembling distilled datasets makes the training process more stable. 
- Gradient matching and distribution matching mutually enhance each other. 

## Getting Started
Prepare the CIFAR10, CIFAR100, ImageNet10, and TinyImageNet datasets.

## Experiment Commands

### Pretrain Early-Stage Models on Real Data

Pretrain early-stage models on real data, simply run the following codes  (In our experiment, seed is set to 2023):

```
python pretrain.py -d [dataset] --nclass [nclass] -n [network] --pt_from [epoch] --seed [seed] --lr [lr] --aug_type [aug]
```

- ```-d```: training dataset.
- ```--nclass```: the number of classes in training dataset.
- ```-n```: the network of early-stage models. ```-n convnet``` for CIFAR-10 and CIFAR-100, ```-n resnetap``` for ImageNet-10.
- ```--pt_from```: the epochs of pretraining. ```--pt_from 2``` for CIFAR-10 and CIFAR-100, ```--pt_from 10``` for ImageNet-10.
- ```--seed```: the random seed of model. 
- ```--aug_type```: the data augmentation of training and ```--aug_type``` can be selected from color, crop, cutout, flip, scale and rotate and joined with '-', like ```--aug_type color_crop_cutout_flip_scale_rotate```.


### Optimize Condensed Data

First change "Your_Pretrained_Model_Path" to your pretrained model path in condense.py.

Synthesize condensed data, run the following codes:

```
python condense.py --reproduce -d [dataset] -f [factor] --ipc [ipc] -n [network] --model_path [path] --niter [niter] --tag [tag]
```

- ```-d```: training dataset.
- ```-f```: factor. ```-f 2``` for CIFAR-10, CIFAR-100, and TinyImageNet ```-f 3``` for ImageNet-10.
- ```--ipc```: number of image per class in distilled dataset.
- ```-n```: the network. ```-n convnet``` for CIFAR-10 and CIFAR-100, ```-n resnetap``` for ImageNet-10.
- ```--model_path```: the path of checkpoints of pretrained early-stage models.
- ```--niter```: the number of outer loop iterations.
- ```--tag```: the name of your current experiment.
- You can also modify the parameters for other datasets.


## Acknowledgement
This project is mainly developed based on the following works:
- [IDC](https://github.com/snu-mllab/efficient-dataset-condensation)
- [AccDD](https://github.com/ncsu-dk-lab/Acc-DD)
- [DREAM](https://github.com/vimar-gu/DREAM)

## Citation
If you find this work helpful, please cite:
```
@article{zhang2025gradient,
  title={Gradient amplification for gradient matching based dataset distillation},
  author={Zhang, Jingxuan and Chen, Zhihua and Dai, Lei and Li, Ping and Sheng, Bin},
  journal={Neural Networks},
  pages={107819},
  year={2025},
}
```
