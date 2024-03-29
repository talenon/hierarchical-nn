# Leveraging the class hierarchy in deep learning
## Under review at ICML2023

*🔨This is preliminary anonymous directory for respecting double-blind constraints at ICML. The code will be moved to a public directory after the reviewing process. A notebook will be also published for running Linear Discriminant Analysis on representations🔨*

![](assets/lda.PNG)


## Abstract 

> When data availability is limited, the implementation of deep learning complex models with many parameters requires the exploitation of a priori knowledge. We propose here a classification method aiming at integrating such knowledge of the classes, expressed in the form of a hierarchy, in which the classes are organized according to a tree structure expressing the inclusion of classes in superclasses. We present a loss function that incorporates this hierarchy, by stipulating that each example belongs not only to a class but also to all superclasses that contain it, associated with an appropriate regularization of the coefficients of the last softmax layer of the network. These two components of the learning algorithm can be utilized in any feedforward architecture with a softmax output layer. We conduct an experimental study performed on three state-of-the-art networks (InceptionV3, ResNet50, and ViT-B16) and three benchmarks (ImageNet-1K, CIFAR100, and FGVC-Aircraft), varying the size of the training sets from small to medium and show an almost systematic improvement in performance. We finally apply our approach to the real problem that originally motivated this study, namely the classification of femur fractures; despite the large class imbalance, our method likewise achieves notable accuracy improvements. 

## Dependencies
A Conda environment is available in the ```env``` folder. To import it type:
```
conda env create -f hierarchical-nn-env.yml
```

## Experiment
To reproduce the experiment, ImageNet-1K, CIFAR100 and FGVC-Aircraft should be download from the official release and re-organized into train and test.  
1. [ImageNet-1K](https://www.image-net.org/challenges/LSVRC/index.php)  
2. [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar.html)  
3. [FGVC-Aircraft](https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/)    

The tree hierarchies for the tree dataset are available in the folder ```trees```.  

Run ```train.py``` to train a network and ```test.py``` to test it. The required arguments are:

```-hl```: Whether to use loss hierarchical or not  
```-m```: Choose between ```inception```, ```resnet``` or ```vit```  
```-d```: Choose between ```imagenet```, ```fgvc``` or ```cifar``` 
```-r```: Reduction factor of the training dataset 
```-tp```: Path to the tree file  
```-dp```: Path to the dataset folder containing train and test folders  
```-op```: Path to folder where to save models  
```-lp```: Path to folder where to save logs with Tensorboard 

While the non-required argument are:  
```-b```: Batch size, default=64  
```-e```: Number of epochs, default=30  
```-lr```: Learning rate, default=0.001  
```-wd```: Weight decay, default=0.1  
```-pm```: Wether to plot or not top-down and down-top matrices, default=False  

