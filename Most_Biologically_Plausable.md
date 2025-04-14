# Brian2-Based Spiking Neural Network for CIFAR-10

This project implements a biologically inspired spiking neural network (SNN) using the [Brian2](https://brian2.readthedocs.io/) simulator. The model is trained and tested on a subset of the CIFAR-10 dataset and features both unsupervised learning (via spike-timing-dependent plasticity, STDP, and reward-modulated STDP) and several diagnostic visualizations.

## Overview

The script performs the following key tasks:

1. **Data Preparation and Encoding:**
   - **Dataset Loading:**  
     CIFAR-10 images are loaded using TensorFlow Keras. A subset is selected for robust yet manageable training/testing (10,000 images for training, 2,000 for testing).
   - **Preprocessing & Poisson Encoding:**  
     Each color image is first converted to grayscale using `rgb2gray`. The image is intensity-normalized using its 2nd and 98th percentiles. Then, a rate map (proportional to pixel intensity) is created. A Poisson process is simulated over a fixed duration (default 100 ms) to generate spike trains.  
     
2. **Network Architecture:**
   - **Neuron Groups:**  
     - **G_input:** A `SpikeGeneratorGroup` representing 32×32 input pixels (flattened to 1024 neurons).
     - **G_hidden1:** 500 leaky integrate-and-fire (LIF) neurons receiving input from G_input with STDP-based synapses.
     - **G_hidden2:** 200 LIF neurons receiving input from G_hidden1; lateral inhibition is applied within this group to promote competitive firing.
     - **G_output:** 10 LIF neurons (one for each CIFAR-10 class) receiving inputs from G_hidden2 with reward-modulated STDP.
   - **Synapses:**  
     - **Input → Hidden1:** Synapses implement a basic STDP rule (using parameters such as `A_pre`, `A_post`, `tau_pre`, and `tau_post`).
     - **Hidden1 → Hidden2:** Fixed-weight connections.
     - **Hidden2 → Output:** Synapses use a reward-modulated version of STDP (R-STDP) that adjusts weights based on a reward signal (1 for a correct prediction, -1 otherwise).
     - **Lateral Inhibition:** Applied within hidden layer 2 and output layer to enforce competition.

3. **Simulation & Training:**
   - For each training image:
     - The image is encoded into a spike train via Poisson encoding.
     - The input spikes are set on the G_input group and the network is run for 200 ms.
     - The output neurons’ spike counts are computed; the predicted class is taken as the neuron with the highest count.
     - A reward signal is applied (positive if the prediction is correct; negative otherwise) to modulate synaptic plasticity.
   - After training, synaptic weights (from G_input to G_hidden1) are normalized.

4. **Evaluation & Diagnostics:**
   - **Test Evaluation:**  
     The network is evaluated on a selected set of test images. Predictions are compared to true labels to compute accuracy.
   - **Confusion Matrix:**  
     A confusion matrix is generated from the predicted and true labels and visualized using seaborn’s heatmap. This matrix reveals class-by-class performance.
   - **Visual Diagnostics:**  
     - A **bar plot** shows the firing rates (spike counts) of neurons in the G_hidden2 group.
     - A **heatmap** visualizes a 100×100 block of the synaptic weight matrix (from input to hidden layer 1), providing insight into the learned connectivity.

## Dependencies

The script relies on:

- [Brian2](https://brian2.readthedocs.io/)
- TensorFlow Keras (for CIFAR-10 data)
- scikit-image (for `rgb2gray`)
- NumPy
- Matplotlib
- Seaborn
- scikit-learn

Install the required packages via pip:

```bash
pip install brian2 tensorflow scikit-image numpy matplotlib seaborn scikit-learn
