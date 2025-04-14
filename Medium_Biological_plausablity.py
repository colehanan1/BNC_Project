import numpy as np
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# -------------------------------
# Parameters
# -------------------------------
T = 100  # Total simulation time (ms)
dt = 1.0  # Time step (ms)
num_steps = int(T / dt)
max_rate = 100.0  # Maximum firing rate (Hz)


# -------------------------------
# Retina-Inspired Poisson Encoder
# -------------------------------
def poisson_encode(image, T, max_rate):
    """
    Converts a normalized image (values in [0, 1]) into a spike train using Poisson encoding.

    Args:
        image (np.array): 2D array (H x W) of pixel intensities.
        T (int): Total simulation time in ms.
        max_rate (float): Maximum firing rate (Hz).

    Returns:
        spike_train (np.array): Binary array of shape (num_pixels, T) where each element is 0 or 1.
    """
    flat = image.flatten()  # Flatten image to a 1D vector of size H*W.
    # For each pixel, each ms a spike is generated with probability proportional to its intensity.
    spike_train = np.random.rand(flat.size, T) < (flat * max_rate * dt / 1000.0)
    return spike_train.astype(int)


# -------------------------------
# LIF Neuron Layer with STDP
# -------------------------------
class LIFNeuronLayer:
    def __init__(self, n_neurons, input_size, dt=1.0, tau_m=20.0, V_thresh=1.0, V_reset=0.0):
        """
        Initializes a layer of LIF neurons with a weight matrix connecting the input.

        Args:
            n_neurons (int): Number of output neurons.
            input_size (int): Number of input neurons (e.g., 28*28 for MNIST).
        """
        self.n_neurons = n_neurons
        self.input_size = input_size
        self.dt = dt
        self.tau_m = tau_m
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.V = np.zeros(n_neurons)  # Membrane potentials
        # Initialize weights randomly (shape: [n_neurons, input_size])
        self.weights = np.random.rand(n_neurons, input_size) * 0.1
        # Store last spike times (for STDP timing differences)
        self.last_post_spike = -np.ones(n_neurons) * 1e8
        self.last_pre_spike = -np.ones(input_size) * 1e8
        # STDP parameters
        self.A_plus = 0.01  # Potentiation factor
        self.A_minus = -0.012  # Depression factor
        self.tau_plus = 20.0  # Time constant (ms) for potentiation
        self.tau_minus = 20.0  # Time constant (ms) for depression

    def reset(self):
        """ Resets the membrane potentials to baseline. """
        self.V = np.zeros(self.n_neurons)

    def forward(self, input_spikes, t):
        """
        Simulates one time step: integrates input, updates potentials, and generates output spikes.

        Args:
            input_spikes (np.array): Binary array of shape (input_size,) for the current time step.
            t (int): Current simulation time step.

        Returns:
            spikes (np.array): Binary array of shape (n_neurons,) indicating which neurons spiked.
        """
        I = np.dot(self.weights, input_spikes)  # Compute weighted input current
        # Update membrane potentials (Euler integration of LIF dynamics)
        self.V = self.V + self.dt * (-self.V / self.tau_m + I)
        # Determine which neurons fire
        spikes = (self.V >= self.V_thresh).astype(int)
        for j in range(self.n_neurons):
            if spikes[j]:
                self.V[j] = self.V_reset  # Reset potential after spike
                self.last_post_spike[j] = t
        return spikes

    def update_stdp(self, input_spikes, spikes, t):
        """
        Updates the synaptic weights using a simple STDP rule.

        Args:
            input_spikes (np.array): Binary input spike vector (input_size,).
            spikes (np.array): Binary output spike vector (n_neurons,).
            t (int): Current simulation time.
        """
        # Update weights for each connection when the postsynaptic neuron fires
        for j in range(self.n_neurons):
            if spikes[j]:
                for i in range(self.input_size):
                    if input_spikes[i]:
                        dt_diff = t - self.last_pre_spike[i]
                        if dt_diff >= 0:
                            dw = self.A_plus * np.exp(-dt_diff / self.tau_plus)
                        else:
                            dw = self.A_minus * np.exp(dt_diff / self.tau_minus)
                        self.weights[j, i] += dw
                        # Prevent weights from becoming negative
                        self.weights[j, i] = max(self.weights[j, i], 0)
        # Update last presynaptic spike times
        for i in range(self.input_size):
            if input_spikes[i]:
                self.last_pre_spike[i] = t


# -------------------------------
# Training and Testing Functions
# -------------------------------
def train_snn(train_dataset, num_neurons=10, num_epochs=1, T=100, dt=1.0, max_rate=100.0):
    """
    Trains the SNN using unsupervised STDP on the MNIST dataset.

    For each training image, the network is simulated for T ms. A simple association matrix
    is updated so that output neurons (via winner-take-all) build an association with image labels.

    Returns:
        layer: The trained LIFNeuronLayer.
        associations: An array of shape (num_neurons, 10) counting associations with each label.
    """
    input_size = 28 * 28  # MNIST images are 28x28
    num_steps = int(T / dt)
    layer = LIFNeuronLayer(n_neurons=num_neurons, input_size=input_size, dt=dt)
    # Association matrix: for each output neuron, count how many times it fired for each true label.
    associations = np.zeros((num_neurons, 10))

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for idx, (img, label) in enumerate(train_dataset):
            # img is a tensor with shape [1, 28, 28]; convert it to a numpy array of shape (28,28)
            img_np = img.squeeze().numpy()
            # Encode the image into a spike train using Poisson encoding
            spike_train = poisson_encode(img_np, T, max_rate)
            layer.reset()
            output_spike_count = np.zeros(num_neurons)
            # Simulate over the duration T
            for t in range(num_steps):
                input_spikes = spike_train[:, t]
                spikes = layer.forward(input_spikes, t)
                layer.update_stdp(input_spikes, spikes, t)
                output_spike_count += spikes
            # Determine the "winning" neuron (i.e. the one that fired most) for the image.
            winner = np.argmax(output_spike_count)
            associations[winner, label] += 1
            if idx % 1000 == 0:
                print(f"Processed {idx} images")
    return layer, associations


def test_snn(test_dataset, layer, associations, T=100, dt=1.0, max_rate=100.0):
    """
    Tests the SNN on the MNIST test dataset.

    Uses the association matrix (built during training) to map neurons to labels.

    Returns:
        accuracy: Classification accuracy (percentage).
    """
    input_size = 28 * 28
    num_steps = int(T / dt)
    num_neurons = layer.n_neurons
    # Determine the label associated with each neuron (winner-take-all)
    neuron_labels = np.argmax(associations, axis=1)
    correct = 0
    total = 0
    for idx, (img, label) in enumerate(test_dataset):
        img_np = img.squeeze().numpy()
        spike_train = poisson_encode(img_np, T, max_rate)
        layer.reset()
        output_spike_count = np.zeros(num_neurons)
        for t in range(num_steps):
            input_spikes = spike_train[:, t]
            spikes = layer.forward(input_spikes, t)
            output_spike_count += spikes
        winner = np.argmax(output_spike_count)
        predicted_label = neuron_labels[winner]
        if predicted_label == label:
            correct += 1
        total += 1
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def visualize_raster(layer, img, T=100, dt=1.0, max_rate=100.0):
    """
    Visualizes the spiking activity (raster plot) of the network for a given test image.
    """
    img_np = img.squeeze().numpy()
    spike_train = poisson_encode(img_np, T, max_rate)
    num_steps = int(T / dt)
    layer.reset()
    spikes_over_time = np.zeros((layer.n_neurons, num_steps))
    for t in range(num_steps):
        input_spikes = spike_train[:, t]
        spikes = layer.forward(input_spikes, t)
        spikes_over_time[:, t] = spikes
    plt.figure(figsize=(10, 5))
    # Create a vertical line at each time a neuron fired.
    for neuron in range(layer.n_neurons):
        times = np.where(spikes_over_time[neuron] > 0)[0]
        plt.vlines(times, neuron + 0.5, neuron + 1.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.title("Raster Plot of Output Neuron Activity")
    plt.show()


# -------------------------------
# Main Script
# -------------------------------
if __name__ == "__main__":
    # Load MNIST dataset using torchvision (images are converted to tensors in [0, 1])
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Train the SNN (unsupervised STDP training)
    num_epochs = 1  # Increase epochs for improved learning (set small for demonstration)
    snn_layer, associations = train_snn(train_dataset, num_neurons=10, num_epochs=num_epochs, T=T, dt=dt,
                                        max_rate=max_rate)

    # Evaluate the SNN on the test dataset
    test_accuracy = test_snn(test_dataset, snn_layer, associations, T=T, dt=dt, max_rate=max_rate)

    # Visualize spiking activity (raster plot) for a sample test image
    sample_img, sample_label = test_dataset[0]
    visualize_raster(snn_layer, sample_img, T=T, dt=dt, max_rate=max_rate)
