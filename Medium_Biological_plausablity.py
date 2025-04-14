import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import time

# -------------------------------
# Device selection: Use MPS on Apple M2 if available, otherwise CUDA or CPU.
# -------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple GPU (MPS)")
elif torch.cuda.is_available():
    device = torch.device("cuda")
    print("Using CUDA GPU")
else:
    device = torch.device("cpu")
    print("Using CPU")

# -------------------------------
# Global simulation parameters
# -------------------------------
T = 100  # Total simulation time in ms
dt = 1.0  # Time step in ms
num_steps = int(T / dt)
max_rate = 100.0  # Maximum firing rate (Hz)


# -------------------------------
# Retina-Inspired Poisson Encoder
# -------------------------------
def poisson_encode(image, T, max_rate, dt, device):
    """
    Convert a normalized image tensor (values in [0,1]) into a spike train.

    Args:
        image (torch.Tensor): Tensor of shape (H, W).
        T (int): Total simulation time in ms.
        max_rate (float): Maximum firing rate (Hz).
        dt (float): Simulation time step in ms.
        device (torch.device): Device to run on.

    Returns:
        torch.Tensor: Spike train of shape (num_pixels, T) with 0s and 1s.
    """
    flat = image.view(-1)  # Flatten to 1D vector (num_pixels,)
    # Each pixel fires with probability proportional to its intensity.
    probability = flat * max_rate * dt / 1000.0  # dt in ms, rate in Hz
    rand_vals = torch.rand(flat.size(0), T, device=device)
    spike_train = (rand_vals < probability.unsqueeze(1)).float()
    return spike_train


# -------------------------------
# LIF Neuron Layer with STDP (using PyTorch)
# -------------------------------
class LIFNeuronLayer:
    def __init__(self, n_neurons, input_size, dt=1.0, tau_m=20.0, V_thresh=1.0, V_reset=0.0, device=device):
        """
        Initializes a layer of LIF neurons.

        Args:
            n_neurons (int): Number of output neurons.
            input_size (int): Number of input neurons.
            dt (float): Time step.
            tau_m (float): Membrane time constant (ms).
            V_thresh (float): Firing threshold.
            V_reset (float): Reset potential after a spike.
            device (torch.device): Device to run on.
        """
        self.device = device
        self.n_neurons = n_neurons
        self.input_size = input_size
        self.dt = dt
        self.tau_m = tau_m
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        self.V = torch.zeros(n_neurons, device=device)
        self.weights = torch.rand(n_neurons, input_size, device=device) * 0.1
        self.last_post_spike = torch.full((n_neurons,), -1e8, device=device)
        self.last_pre_spike = torch.full((input_size,), -1e8, device=device)
        # STDP parameters
        self.A_plus = 0.01
        self.A_minus = -0.012
        self.tau_plus = 20.0
        self.tau_minus = 20.0

    def reset(self):
        """Reset membrane potentials."""
        self.V = torch.zeros(self.n_neurons, device=self.device)

    def forward(self, input_spikes, t):
        """
        Update the neuron layer for one time step.

        Args:
            input_spikes (torch.Tensor): Binary tensor of shape (input_size,) for the current time step.
            t (int): Current simulation time step.

        Returns:
            torch.Tensor: Output spike vector (n_neurons,).
        """
        I = torch.matmul(self.weights, input_spikes)  # Input current, shape (n_neurons,)
        self.V = self.V + self.dt * ((-self.V / self.tau_m) + I)
        # Determine which neurons fire
        spikes = (self.V >= self.V_thresh).float()
        fired_idx = (spikes == 1).nonzero(as_tuple=True)[0]
        if fired_idx.numel() > 0:
            self.V[fired_idx] = self.V_reset  # Reset potentials after spike
            self.last_post_spike[fired_idx] = t  # Update last spike time for postsynaptic neurons
        return spikes

    def update_stdp(self, input_spikes, spikes, t):
        """
        Update synaptic weights using a simple STDP rule.

        Args:
            input_spikes (torch.Tensor): Binary input spike vector (input_size,).
            spikes (torch.Tensor): Binary output spike vector (n_neurons,).
            t (int): Current simulation time.
        """
        for j in range(self.n_neurons):
            if spikes[j] == 1:
                for i in range(self.input_size):
                    if input_spikes[i] == 1:
                        dt_diff = t - self.last_pre_spike[i]
                        if dt_diff >= 0:
                            dw = self.A_plus * torch.exp(-dt_diff / self.tau_plus)
                        else:
                            dw = self.A_minus * torch.exp(dt_diff / self.tau_minus)
                        new_weight = self.weights[j, i].item() + dw.item()
                        # Ensure weight remains non-negative
                        new_weight = max(new_weight, 0)
                        self.weights[j, i] = torch.tensor(new_weight, device=self.device)
        # Update last presynaptic spike times.
        for i in range(self.input_size):
            if input_spikes[i] == 1:
                self.last_pre_spike[i] = t


# -------------------------------
# Training and Evaluation Functions
# -------------------------------
def train_snn(train_dataset, num_neurons=10, num_epochs=1, T=T, dt=dt, max_rate=max_rate, device=device):
    """
    Train the SNN with unsupervised STDP on the MNIST dataset.

    Args:
        train_dataset: PyTorch MNIST training dataset.
        num_neurons (int): Number of neurons in the layer.
        num_epochs (int): Number of epochs to train.
        T (int): Total simulation time per image (ms).
        dt (float): Time step (ms).
        max_rate (float): Maximum firing rate (Hz).
        device (torch.device): Device to run on.

    Returns:
        layer (LIFNeuronLayer): Trained neuron layer.
        associations (torch.Tensor): Association matrix (num_neurons x 10).
    """
    input_size = 28 * 28  # MNIST images size
    num_steps = int(T / dt)
    layer = LIFNeuronLayer(n_neurons=num_neurons, input_size=input_size, dt=dt, device=device)
    associations = torch.zeros(num_neurons, 10, device=device)  # Count associations between neuron and label

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        for idx, (img, label) in enumerate(train_dataset):
            # Ensure image is on the correct device.
            img = img.to(device)
            spike_train = poisson_encode(img.squeeze(), T, max_rate, dt, device)  # shape: (input_size, T)
            layer.reset()
            output_spike_count = torch.zeros(num_neurons, device=device)
            for t in range(num_steps):
                input_spikes = spike_train[:, t]  # Current time step (shape: (input_size,))
                spikes = layer.forward(input_spikes, t)
                layer.update_stdp(input_spikes, spikes, t)
                output_spike_count += spikes
            # Determine the "winning" neuron (the one that fired most) for this image.
            winner = torch.argmax(output_spike_count)
            associations[winner, label] += 1
            if idx % 1000 == 0:
                print(f"Processed {idx} images")
    return layer, associations.cpu()


def evaluate_snn(test_dataset, layer, associations, T=T, dt=dt, max_rate=max_rate, device=device):
    """
    Evaluate the trained SNN on the MNIST test dataset.

    Args:
        test_dataset: PyTorch MNIST test dataset.
        layer (LIFNeuronLayer): Trained neuron layer.
        associations (torch.Tensor): Association matrix (num_neurons x 10).
        T (int): Total simulation time per image (ms).
        dt (float): Time step (ms).
        max_rate (float): Maximum firing rate (Hz).
        device (torch.device): Device to run on.

    Returns:
        float: Classification accuracy percentage.
    """
    input_size = 28 * 28
    num_steps = int(T / dt)
    num_neurons = layer.n_neurons
    # Map each output neuron to the class with maximum association.
    neuron_labels = torch.argmax(associations, dim=1)
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (img, label) in enumerate(test_dataset):
            img = img.to(device)
            spike_train = poisson_encode(img.squeeze(), T, max_rate, dt, device)
            layer.reset()
            output_spike_count = torch.zeros(num_neurons, device=device)
            for t in range(num_steps):
                input_spikes = spike_train[:, t]
                spikes = layer.forward(input_spikes, t)
                output_spike_count += spikes
            winner = torch.argmax(output_spike_count)
            predicted_label = neuron_labels[winner].item()
            if predicted_label == label:
                correct += 1
            total += 1
    accuracy = 100 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def visualize_raster(layer, img, T=T, dt=dt, max_rate=max_rate, device=device):
    """
    Visualizes network spiking activity (raster plot) for a given test image.
    """
    img = img.to(device)
    spike_train = poisson_encode(img.squeeze(), T, max_rate, dt, device)
    num_steps = int(T / dt)
    layer.reset()
    spikes_over_time = torch.zeros(layer.n_neurons, num_steps, device=device)
    for t in range(num_steps):
        input_spikes = spike_train[:, t]
        spikes = layer.forward(input_spikes, t)
        spikes_over_time[:, t] = spikes
    spikes_over_time = spikes_over_time.cpu().numpy()

    plt.figure(figsize=(10, 5))
    for neuron in range(layer.n_neurons):
        times = [t for t in range(num_steps) if spikes_over_time[neuron, t] > 0]
        if times:
            plt.vlines(times, neuron + 0.5, neuron + 1.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.title("Raster Plot of Output Neuron Activity")
    plt.show()


# -------------------------------
# Main Script Execution
# -------------------------------
if __name__ == "__main__":
    # Define transformation: MNIST images are already in [0,1]
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # Train the SNN (use more epochs for better learning)
    num_epochs = 1  # Increase if desired
    start_time = time.time()
    snn_layer, associations = train_snn(train_dataset, num_neurons=10, num_epochs=num_epochs, T=T, dt=dt,
                                        max_rate=max_rate, device=device)
    print(f"Training completed in {(time.time() - start_time):.2f} seconds.")

    # Evaluate the SNN on the test dataset
    accuracy = evaluate_snn(test_dataset, snn_layer, associations, T=T, dt=dt, max_rate=max_rate, device=device)

    # Visualize spiking activity for a sample test image
    sample_img, sample_label = test_dataset[0]
    visualize_raster(snn_layer, sample_img, T=T, dt=dt, max_rate=max_rate, device=device)
