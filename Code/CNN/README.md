# Convolutional Neural Networks

![](http://ForTheBadge.com/images/badges/made-with-python.svg) 

## Dependencies

![](https://img.shields.io/badge/TensorFlow-2.1.0-FF6F00?style=for-the-badge&logo=TensorFlow)
![](https://img.shields.io/badge/numpy-1.19.2-013243?style=for-the-badge&logo=NumPy)
![](https://img.shields.io/badge/matplotlib-3.3.4-224099?style=for-the-badge)
![](https://img.shields.io/badge/utils-1.0.1-123456?style=for-the-badge)
![](https://img.shields.io/badge/pil-1.1.6-654321?style=for-the-badge)

## Introduction

In deep learning, a convolutional neural network (CNN, or ConvNet) is a class of deep neural networks, most commonly applied to analyzing visual imagery.

CNNs are regularized versions of multilayer perceptrons. Multilayer perceptrons usually mean fully connected networks, that is, each neuron in one layer is connected to all neurons in the next layer. The "full connectivity" of these networks make them prone to overfitting data. Typical ways of regularization, or preventing overfitting, include: penalizing parameters during training (such as weight decay) or trimming connectivity (skipped connections, dropout, etc.) CNNs take a different approach towards regularization: they take advantage of the hierarchical pattern in data and assemble patterns of increasing complexity using smaller and simpler patterns embossed in their filters. Therefore, on a scale of connectivity and complexity, CNNs are on the lower extreme.

CNNs use relatively little pre-processing compared to other image classification algorithms. This means that the network learns to optimize the filters (or kernels) through automated learning, whereas in traditional algorithms these filters are hand-engineered. This independence from prior knowledge and human intervention in feature extraction is a major advantage.

Head [Here](Notebook.ipynb) to see detailed explanation and code.

## Thanks for Reading :)