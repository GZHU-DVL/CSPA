# CSPA
** Cross-shaped Adversarial Patch Attack **     
*Yu Ran, Weijia Wang, Mingjie Li, Lin-Cheng, Li Yuan-Gen Wang and Li Jin*         
IEEE Transactions on Circuits and Systems for Video Technology (TCSVT)

## Setup
### Requirements  
* PyTorch 1.7.1 or above

### Datasets   
We evaluate the proposed method on the MNIST, CIFAR10, and ImageNet datasets.    
In main.py, set the following variable:  

* `IMAGENET_PATH`: path to the ImageNet validation set.
* `TINYIMAGENET_PATH`: path to the TinyImageNet validation set.


##  Using models and attacks from the paper
The following provides the arguments to run the attacks described in the paper. 

### Untargeted attack
Untargeted attack performance against ResNet50 model on ImageNet dataset.
```bash
python main.py --seed 1 --dataset ImageNet --model resnet50 --loss margin --bs 100 --n_queries 2000 --miu_init .5 --interval 7 --width 1 --length 200
```
Untargeted attack performance against VGG16_bn model on TinyImageNet dataset.
```bash
python main.py --seed 1 --dataset TinyImageNet --model vgg16_bn --loss margin --bs 100 --n_queries 2000 --miu_init .5 --interval 7 --width 1 --length 32
```
Untargeted attack performance against DesNet121 model on CIFAR100 dataset.
```bash
python main.py --seed 1 --dataset CIFAR100 --model desnet121 --loss margin --bs 100 --n_queries 2000 --miu_init .5 --interval 7 --width 1 --length 25
```
Untargeted attack performance against CNN model on CIFAR10 dataset.
```bash
python main.py --seed 1 --dataset CIFAR10 --model cnn --loss margin --bs 100 --n_queries 2000 --miu_init .5 --interval 7 --width 1 --length 25
```

### Targeted attack
Targeted attack performance against ResNet50 model on ImageNet dataset.
```bash
python main.py --seed 1 --dataset ImageNet --model resnet50 --loss ce --bs 100 --n_queries 2000 --miu_init .4 --interval 10 --width 1 --length 200 --targeted
```
Targeted attack performance against VGG16_bn model on TinyImageNet dataset.
```bash
python main.py --seed 1 --dataset TinyImageNet --model vgg16_bn --loss ce --bs 100 --n_queries 2000 --miu_init .4 --interval 10 --width 1 --length 32 --targeted
```
Targeted attack performance against DesNet121 model on CIFAR100 dataset.
```bash
python main.py --seed 1 --dataset CIFAR100 --model desnet121 --loss ce --bs 100 --n_queries 2000 --miu_init .4 --interval 10 --width 1 --length 25 --targeted
```
Targeted attack performance against CNN model on CIFAR10 dataset.
```bash
python main.py --seed 1 --dataset CIFAR10 --model cnn --loss ce --bs 100 --n_queries 2000 --miu_init .4 --interval 10 --width 1 --length 25 --targeted
```

## License
This source code is made available for research purposes only.

## Acknowledgment
Our code is built upon [**Patch-RS**](https://github.com/fra31/sparse-rs).

