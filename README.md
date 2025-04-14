# BNC_Project

## Overview
This project explores biologically plausible spiking neural networks (SNNs) for image classification tasks, focusing on the CIFAR-10 dataset. The goal is to implement and compare different levels of biological plausibility in SNN architectures, ranging from highly biologically plausible to less plausible but computationally efficient models. The project uses frameworks such as `Brian2` and `PyTorch` to simulate spiking neurons and train models.

## Scripts Overview

### `Most_biologically_plausable.py`
This script implements a highly biologically plausible spiking neural network using the `Brian2` library. It incorporates mechanisms such as:

- **Poisson Encoding**: Converts grayscale images into spike trains based on pixel intensity.
- **STDP (Spike-Timing-Dependent Plasticity)**: A biologically inspired learning rule for synaptic weight updates.
- **R-STDP (Reward-Modulated STDP)**: Adds a reward signal to guide learning in the output layer.
- **Lateral Inhibition**: Introduces competition between neurons to enhance diversity in firing patterns.

The network consists of three layers:
1. **Input Layer**: Encodes image data into spikes.
2. **Hidden Layers**: Two hidden layers with STDP-based learning.
3. **Output Layer**: Uses R-STDP to classify images.

The script trains the network on CIFAR-10 and evaluates its performance, visualizing results such as weight matrices, confusion matrices, and neuron firing rates.

---

### `Medium_Biological_plausablity.py`
This script balances biological plausibility and computational efficiency. It uses a combination of `PyTorch` and biologically inspired mechanisms:

- **Difference-of-Gaussians (DoG) Filtering**: Simulates retina-like preprocessing of images.
- **Poisson Encoding**: Converts filtered images into spike trains.
- **LIF (Leaky Integrate-and-Fire) Neurons**: Simulates spiking neurons with membrane potential dynamics.
- **STDP**: Implements unsupervised learning in the hidden layer.

The workflow includes:
1. **Unsupervised Training**: Trains an SNN layer using STDP to extract features from images.
2. **Feature Extraction**: Converts images into spike-based features.
3. **Supervised Training**: Trains a linear classifier on the extracted features for image classification.

The script also provides tools for visualizing spiking activity (raster plots) and evaluating classification performance.

---

### `Less_biologically_plausable.py`
This script focuses on computational efficiency while retaining some biological inspiration. It uses `PyTorch` to implement:

- **CNN to SNN Conversion**: A pre-trained convolutional neural network (CNN) is converted into a rate-based spiking neural network.
- **Rate-Based Approximation**: Simulates spiking activity using ReLU activations as firing rates.

The workflow includes:
1. **CNN Training**: Trains a standard CNN on CIFAR-10.
2. **SNN Conversion**: Converts the trained CNN into an SNN.
3. **Evaluation**: Compares the performance of the CNN and the SNN on the test dataset.

This approach sacrifices biological plausibility for faster training and inference while maintaining reasonable accuracy.

---

## Key Features
- Comparison of biologically plausible SNNs with varying levels of realism.
- Use of advanced learning rules like STDP and R-STDP.
- Visualization of neuron activity, weight matrices, and classification performance.
- Application of biologically inspired preprocessing techniques like DoG filtering.

## Dependencies
- `Brian2`
- `PyTorch`
- `TensorFlow/Keras`
- `scikit-image`
- `matplotlib`
- `seaborn`
- `scikit-learn`

## Conclusion
This project demonstrates the trade-offs between biological plausibility and computational efficiency in SNNs. The scripts provide a foundation for further exploration of biologically inspired machine learning models.
