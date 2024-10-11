2# FSRCNN (Fast Super Resolution Convolutional Neural Networks)

## Objective
The aim of the project is to implement the **FSRCNN Model** which deals with conversion of Low Resolution images to High Resolution images with the help of Convolutional Neural Networks

## Approach
1. Firstly we started with learning basic concepts of Neural Netwokrs and Machine Learning.
2. With this knowledge we started the implementation of the handwritten digit recognition model (MNIST dataset) by a simple 2 layer ANN , using numpy from scratch.
3. Further we learned the basic concepts of Optimizers, Hyperparameter tuning, Convolutional Neural Networks and studied its various architectures.
4. Then we implemented the MNIST model using the PyTorch framework (Firstly using a 2 layer ANN and then by using a Deep CNN architecture).
5. Furthermore we also implemented an object detection model based on the CIFAR-10 dataset using a Deep CNN architecture along with batch normalization and dropout regularization.
6. Then we implemented a custom dataloader to extract the raw High Res and Low Res images from the BSD-100 dataser which would be further used as train and test datasets for the implementation of SRCNN and FSRCNN.
7. We implemented the SRCNN architecture in PyTorch for the BSD-100 dataset, taking reference for the architecture from the following research paper :-
https://doi.org/10.48550/arXiv.1501.00092.pdf
8. Finally we implemented the FSRCNN architecture for the same dataset. Reference from the following research paper :-
https://arxiv.org/pdf/1608.00367v1.pdf

