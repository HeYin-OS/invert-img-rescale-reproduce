

# invert-img-rescale-reproduce

# Introduction



For U-TOKYO Visual Media Assignment, the GitHub link is https://github.com/HeYin-OS/invert-img-rescale-reproduce

This is a reproduction of a paper named Invertible Image Rescaling (https://arxiv.org/abs/2005.05650) using pytorch, numpy, yaml, and other relevant public libraries. The code was built on Windows but not MacOS or Linux.

As the copyright is limited, I DO NOT PROVIDE the dataset. Please prepare the ground-truth SR images yourselves and modify the only YAML file.

The whole training will take 4 days to 2 weeks based on the performance of the GPU. Please be ready before you start the program.

# Value

In this paper, they propose a novel method to largely mitigate the ill-posed problem of image upscaling. High-frequency contents will be lost during downscaling. but using some distributions to rebuild the connections between this form and HF contents is possible. They developed an invertible model called Invertible Rescaling Net (IRN) which captures some knowledge on the lost information in the form of distribution and embeds it into the model’s parameters to mitigate the ill-posedness. Converting case-specific high-frequency content into an auxiliary case-agnostic latent variable whose marginal distribution obeys a fixed pre-specified distribution (e.g., isotropic Gaussian) makes the downscaling invertible.

Powered by the deliberately designed invertibility, IRN can largely mitigate the ill-posed nature of image upscaling reconstruction from the downscaled LR image.

The proposed IRN can significantly boost the performance of upscaling reconstruction from downscaled LR images compared with state-of-the-art downscaling-SR and encoder-decoder methods. Moreover, the amount of parameters of IRN is significantly reduced, indicating the new IRN model's lightweight and high efficiency.

# Implemented contents

## dataset.py

It defines two PyTorch dataset classes, LQDataset and LQGTDataset, for handling low-quality (LQ) and ground-truth (GT) image datasets. Both classes initialize with configuration options and set up paths and environments for accessing image data, including support for LMDB databases. The __getitem__ method in each class retrieves and processes images, including reading from paths, handling color channel conversions, and data augmentation. LQGTDataset additionally handles paired LQ and GT images, supporting operations such as resizing, cropping, and augmenting images for training and evaluation purposes.

## discriminator.py

It defines two PyTorch neural network modules, Discriminator and Extractor. The Discriminator class is designed for distinguishing real from generated images, featuring multiple convolutional layers with LeakyReLU activations and batch normalization, followed by fully connected layers for the final classification. The Extractor class, based on a pre-trained VGG19 network, extracts feature representations from images, optionally normalizing inputs before feeding them into a truncated version of the VGG network. The features layer's parameters are frozen to prevent updates during training.

## INV.py

It defines several PyTorch modules for image processing and transformation tasks. The InvBlock class implements an invertible block for image transformations, utilizing subnetworks for forward and inverse transformations with Jacobian calculation. The ConvGrey class performs color space transformations between RGB/BGR and grayscale, with learnable parameters. The GreyNet class integrates ConvGrey and multiple InvBlock instances for complex transformations. The HaarDS class implements the Haar wavelet transform for downsampling and its inverse, with pre-defined weights. The ConvDS class handles convolution-based downsampling and its inverse, adjusting the input shape accordingly. Finally, the Rescale class orchestrates a sequence of downsampling operations and InvBlock instances, supporting various configurations for flexible image processing pipelines.

## IRN.py

It implements various components for training and optimizing a neural network, specifically designed for image restoration tasks using PyTorch. It includes custom implementations of a quantization function (Quant), learning rate schedulers (MRestart and CRestart), and different loss functions (ReLoss, GANLoss, GPL). The IRN class integrates these components, managing the training process, including feeding data, optimizing the model, handling learning rate adjustments, and performing upscaling and downscaling operations. It also supports saving and loading model states, handling distributed and data-parallel training, and logging the training progress. The IRN class encapsulates the entire training workflow, making it robust and adaptable for various training configurations and requirements.

## main.py

It implements a comprehensive training framework for a neural network using PyTorch, specifically designed for image restoration tasks. It includes functions for parsing configuration options from a YAML file, setting up directories, resuming training from checkpoints, and preparing datasets for training and validation. The main function initializes the training environment, seeds for reproducibility, and loads datasets and model configurations. It then orchestrates the training procedure, including feeding data, optimizing the model, validating performance, logging progress, and saving checkpoints. The framework supports distributed training, learning rate scheduling, and various model utilities to ensure robust and efficient training of image restoration models.

## networks.py

It defines several functions (makeG, makeR, makeGrey, makeD, and makeF) to create different neural network components based on given configuration options. These components include a generator network (makeG) for image rescaling, a residual network (makeR), a grayscale network (makeGrey), a discriminator network (makeD) for adversarial training, and a feature extractor network (makeF). Each function configures its respective network according to specified initialization methods, network structures, and parameters such as input and output channels, number of blocks, and whether to use batch normalization. This setup facilitates the construction and initialization of various model architectures for tasks such as image restoration and enhancement.

## residual.py

It defines three PyTorch neural network modules: ResidualDB, DenseBlock, and BlockNet, which are used for constructing a residual dense block-based network. The DenseBlock class creates a block with multiple convolutional layers connected densely, where each layer takes input from all previous layers. The ResidualDB class stacks three such dense blocks, adding the input to the output of the last block to create a residual connection. The BlockNet class defines the overall network architecture, starting with an initial convolution, followed by a series of residual dense blocks, another convolution, and a final convolution to produce the output, effectively combining the features learned from multiple dense blocks to enhance the network's performance in image processing tasks.

## sampler.py

It defines an IterSampler class, which is a custom PyTorch sampler designed for distributed training. The IterSampler class ensures that each process (or replica) in a distributed setting gets a unique subset of the dataset for each epoch. It calculates the number of samples per replica based on the total number of replicas and the dataset size, shuffles the indices of the dataset each epoch, and then partitions the indices so that each replica processes a distinct subset. This setup facilitates balanced and randomized sampling across multiple GPUs or nodes, enhancing the training efficiency and effectiveness in distributed machine learning environments.

## subnet.py

It defines a DenseBlock class and a subnet function to construct dense blocks for neural networks. The DenseBlock class implements a densely connected convolutional block with five convolutional layers, each followed by a LeakyReLU activation, and initializes weights using either Xavier or a default method. The subnet function returns a constructor for creating DenseBlock instances based on the specified network structure (DBNet) and initialization method. This setup allows for flexible creation of dense blocks with customizable initialization and growth rate for use in neural network architectures.

## utils.py

It implements a comprehensive utility module for image processing and neural network operations using PyTorch. It includes several functionalities such as data augmentation (aug), image saving (save_img), color space conversion (bgr2ycbcr), YAML support with ordered dictionaries (OrderedYamlSupport), and model creation (create_model). The code also contains functions for reading images from disk or LMDB databases (_read_img_lmdb, read_img), image resizing with anti-aliasing (resizeImg), and calculating image quality metrics like PSNR (calculate_psnr). Utilities for weight initialization (initialize_weights, initialize_weights_xavier), making neural network layers (make_layer), and image grid creation (make_grid) are provided. Additionally, the code includes specialized classes and functions for neural network components, such as ResidualBlock_noBN for residual blocks without batch normalization, and flow_warp for warping images based on optical flow. Helper functions for getting image paths from directories or LMDB (_get_paths_from_images, _get_paths_from_lmdb, getImgPath) and converting image channels (CvtChannel) are also included, along with support for cubic interpolation (cubic) and modular cropping (modcrop).
