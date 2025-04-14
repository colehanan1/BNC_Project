# Biologically Inspired Spiking Neural Network for MNIST

This repository contains a Python script that implements a hybrid biologically inspired spiking neural network (SNN) for image classification on the MNIST dataset. The script combines unsupervised feature extraction using a layer of Leaky Integrate-and-Fire (LIF) neurons with Spike-Timing-Dependent Plasticity (STDP) and a supervised linear classifier that is trained on the extracted spike count features.

## Overview

The script is divided into three key phases:

1. **Unsupervised Training (Feature Extraction):**
   - **Retina-Inspired Encoding:** Each MNIST image is pre-processed using a Difference-of-Gaussians (DoG) filter to mimic early visual processing in the retina.
   - **Poisson Spike Encoding:** The filtered image is converted into a spike train based on pixel intensity, simulating the probabilistic nature of neuronal firing.
   - **LIF Neuron Layer with STDP:** A layer of spiking neurons processes the spike trains using Leaky Integrate-and-Fire dynamics, while synaptic weights are updated using a basic STDP rule.
   - **Batch Processing:** The network processes images in batches, leveraging the GPU (MPS backend on Apple M2 or CUDA if available) for acceleration.

2. **Feature Extraction:**
   - After unsupervised training, features for each image are extracted by counting the number of spikes from each neuron over the simulation period.

3. **Supervised Training (Linear Readout):**
   - A simple linear classifier is trained using the extracted features to map the spike counts to one of the 10 MNIST digit classes.
   - Training logs display loss and accuracy per epoch.

4. **Evaluation and Visualization:**
   - The model is evaluated on the test set, and a confusion matrix is generated to analyze class-wise performance.
   - A raster plot visualizes the temporal spiking activity of the unsupervised layer for a sample image.

## Installation

To run this script, ensure you have the following dependencies installed:

- Python 3.7+
- [PyTorch](https://pytorch.org/) (with MPS support if using an Apple M2 or CUDA support for NVIDIA GPUs)
- torchvision
- matplotlib
- numpy
- scipy
- scikit-learn
- seaborn

You can install these with pip:

```bash
pip install torch torchvision matplotlib numpy scipy scikit-learn seaborn
