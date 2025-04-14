# CNN and Rate-Based SNN on CIFAR-10

This repository contains a Python script that implements a conventional Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset and then demonstrates a simple rate-based conversion to a Spiking Neural Network (SNN). The script also includes evaluation routines that display a confusion matrix and compute overall accuracy.

## Overview

The script performs the following key tasks:

1. **CNN Model Definition and Training:**
   - **CNN Architecture:**  
     The CNN model consists of two convolutional layers with ReLU activations and max pooling, followed by two fully connected layers. The network processes CIFAR-10 images (3 channels, 32×32 pixels) and outputs class logits for 10 classes.
   - **Training Process:**  
     The CNN is trained using the Adam optimizer and Cross-Entropy loss function. Training progress is logged by printing the average loss at the end of each epoch.
   
2. **Evaluation and Confusion Matrix:**
   - After training, the script evaluates the CNN on the test set. It computes the total accuracy by comparing predictions with true labels.
   - A confusion matrix is then generated using scikit-learn and visualized with seaborn's heatmap. This matrix displays the number of correct and incorrect predictions per class.

3. **Conversion to a Rate-Based SNN:**
   - **SNN Conversion:**  
     A simple conversion is performed by reusing the CNN’s layers to build an SNN model. The SNN model uses the same weights and structure but interprets the activations as firing rates (by applying ReLU activations) to approximate spiking behavior.
   - **SNN Evaluation:**  
     The SNN is evaluated on the test data similarly to the CNN, and its accuracy is printed.

## Installation

To run the script, ensure you have Python 3.7+ installed along with these dependencies:

- [PyTorch](https://pytorch.org/)
- torchvision
- matplotlib
- numpy
- scikit-learn
- seaborn

You can install the required packages using pip:

```bash
pip install torch torchvision matplotlib numpy scikit-learn seaborn
