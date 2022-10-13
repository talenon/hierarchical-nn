# hierarchical-nn

## Official code for Leveraging the class hierarchy in deep learning, under review at AISTATS2023

# This is a anonymous directory. The code will be moved to a public directory under acceptance.
# A Notebook will be also published for running Linear Discriminant Analysis on representations

To reproduce the experiment, ImageNet-1k, CIFAR100 and FGVC-Aricraft should be download from the official release and re-organized into train and test

Run train.py to train a network and test.py to test it. The required arguments are:

"-hl": Using loss hierarchical or not
"-m": Inception, ResNet or ViT"
"-d": imagenet, fgvc, cifar, bones
"-r": Reduction factor
"-tp": Path to the tree file
"-dp": Path to the dataset folder containing train and test folders
"-op": Path to folder where to save models
"-lp": Path to folder where to save logs

While the non-required argument are:
"-b": Batch size, default=64
"-e": Number of epochs, default=30
"-lr": Learning rate, default=0.001
"-wd": Weight decay, default=0.1
"-pm": Weather to plot or not matrices, default=False

