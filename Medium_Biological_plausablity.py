import torch
from torchvision import datasets, transforms
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.metrics import confusion_matrix
import seaborn as sns
# -------------------------------
# Device Selection: Use MPS on Apple M2 if available, otherwise CUDA or CPU.
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
# Global Simulation and Training Parameters
# -------------------------------
T = 100  # Total simulation time (ms) per image
dt = 1.0  # Time step (ms)
num_steps = int(T / dt)
max_rate = 100.0  # Maximum firing rate (Hz)

# DoG filter parameters for retina-inspired encoding
sigma1 = 1.0
sigma2 = 2.0

global_true_labels = []
global_predicted_labels = []

# -------------------------------
# Utility Functions
# -------------------------------

def dog_filter(image, sigma1=1.0, sigma2=2.0):
    """
    Apply Difference-of-Gaussians (DoG) filtering.

    Args:
        image (np.ndarray): 2D image array.
        sigma1 (float): Standard deviation for the first Gaussian.
        sigma2 (float): Standard deviation for the second Gaussian.

    Returns:
        np.ndarray: Filtered image.
    """
    smooth1 = gaussian_filter(image, sigma=sigma1)
    smooth2 = gaussian_filter(image, sigma=sigma2)
    filtered = smooth1 - smooth2
    # Normalize filtered image to be in [0, 1]
    filtered = filtered - filtered.min()
    if filtered.max() > 0:
        filtered = filtered / filtered.max()
    return filtered


def poisson_encode(image, T, max_rate, dt, device):
    """
    Convert a normalized image (values in [0, 1]) into a spike train using Poisson coding.

    Args:
        image (torch.Tensor): Tensor of shape (H, W) with values in [0,1].
        T (int): Total simulation time (ms).
        max_rate (float): Maximum firing rate (Hz).
        dt (float): Time step (ms).
        device (torch.device): Device to use.

    Returns:
        torch.Tensor: Spike train of shape (num_pixels, T)
    """
    # Convert tensor to numpy for DoG filtering, then back to tensor.
    img_np = image.cpu().numpy()
    filtered = dog_filter(img_np, sigma1, sigma2)
    filtered_tensor = torch.tensor(filtered, dtype=torch.float32, device=device)
    flat = filtered_tensor.view(-1)  # shape: (num_pixels,)
    probability = flat * max_rate * dt / 1000.0  # dt in ms, rate in Hz
    rand_vals = torch.rand(flat.size(0), T, device=device)
    spike_train = (rand_vals < probability.unsqueeze(1)).float()
    return spike_train


# -------------------------------
# LIF Neuron Layer with STDP (Unsupervised Layer)
# -------------------------------
class LIFNeuronLayer:
    def __init__(self, n_neurons, input_size, dt=1.0, tau_m=20.0, V_thresh=1.0, V_reset=0.0, device=device):
        """
        Initialize a layer of LIF neurons.

        Args:
            n_neurons (int): Number of output neurons.
            input_size (int): Number of input neurons.
            dt (float): Time step in ms.
            tau_m (float): Membrane time constant (ms).
            V_thresh (float): Threshold for spiking.
            V_reset (float): Reset potential after spike.
            device (torch.device): Device to use.
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
        Process one simulation time step.

        Args:
            input_spikes (torch.Tensor): Binary tensor (input_size,) for current time step.
            t (int): Current simulation time step.

        Returns:
            torch.Tensor: Output spike vector (n_neurons,)
        """
        I = torch.matmul(self.weights, input_spikes)  # Weighted input current
        self.V = self.V + self.dt * ((-self.V / self.tau_m) + I)
        spikes = (self.V >= self.V_thresh).float()
        fired_idx = (spikes == 1).nonzero(as_tuple=True)[0]
        if fired_idx.numel() > 0:
            self.V[fired_idx] = self.V_reset
            self.last_post_spike[fired_idx] = t
        return spikes

    def update_stdp(self, input_spikes, spikes, t):
        """
        Update weights with a basic STDP rule.

        Args:
            input_spikes (torch.Tensor): Binary input spike vector (input_size,).
            spikes (torch.Tensor): Binary output spike vector (n_neurons,).
            t (int): Current time.
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
                        new_weight = max(new_weight, 0)  # Keep non-negative
                        self.weights[j, i] = torch.tensor(new_weight, device=self.device)
        for i in range(self.input_size):
            if input_spikes[i] == 1:
                self.last_pre_spike[i] = t


# -------------------------------
# Unsupervised Training Phase (STDP)
# -------------------------------
def unsupervised_training(train_dataset, n_neurons=100, num_epochs=1, T=T, dt=dt, max_rate=max_rate, device=device):
    """
    Train the unsupervised SNN layer with STDP on MNIST (with data augmentation).

    Args:
        train_dataset: PyTorch MNIST training dataset.
        n_neurons (int): Number of neurons (features) in the unsupervised layer.
        num_epochs (int): Number of training epochs.
        T (int): Simulation time per image.
        dt (float): Time step.
        max_rate (float): Maximum firing rate.
        device (torch.device): Device to use.

    Returns:
        layer: Trained LIFNeuronLayer.
    """
    input_size = 28 * 28  # MNIST images are 28x28
    num_steps = int(T / dt)
    layer = LIFNeuronLayer(n_neurons=n_neurons, input_size=input_size, dt=dt, device=device)

    previous_weight_norm = None
    previous_avg_spikes = None

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_spike_sum = 0.0
        num_images = 0

        print(f"\n--- Unsupervised Training Epoch {epoch + 1}/{num_epochs} ---")
        for idx, (img, label) in enumerate(train_dataset):
            # img shape: (1, 28, 28)
            # Remove channel dimension and send to device.
            img = img.squeeze().to(device)
            # Apply Poisson encoding after DoG filtering.
            spike_train = poisson_encode(img, T, max_rate, dt, device)  # shape: (784, T)
            layer.reset()
            output_spike_count = torch.zeros(n_neurons, device=device)
            for t in range(num_steps):
                input_spikes = spike_train[:, t]
                spikes = layer.forward(input_spikes, t)
                layer.update_stdp(input_spikes, spikes, t)
                output_spike_count += spikes
            epoch_spike_sum += output_spike_count.sum().item()
            num_images += 1
            if idx % 1000 == 0 and idx > 0:
                print(f"  Processed {idx} images in current epoch...")

        avg_spikes = epoch_spike_sum / num_images
        weight_norm = torch.norm(layer.weights).item() / (n_neurons * input_size)
        epoch_duration = time.time() - epoch_start_time

        print(f"\nEpoch {epoch + 1} complete in {epoch_duration:.2f} sec.")
        print(f"   Average total spikes per image: {avg_spikes:.2f}")
        print(f"   Average weight norm: {weight_norm:.6f}")
        if epoch > 0:
            diff_weight = weight_norm - previous_weight_norm
            diff_spikes = avg_spikes - previous_avg_spikes
            print("   Changes from previous epoch:")
            print(f"       Weight norm change: {diff_weight:+.6f}")
            print(f"       Average spikes change: {diff_spikes:+.2f}")
        previous_weight_norm = weight_norm
        previous_avg_spikes = avg_spikes

    return layer


# -------------------------------
# Feature Extraction from Unsupervised Layer
# -------------------------------
def extract_features(dataset, layer, T=T, dt=dt, max_rate=max_rate, device=device):
    """
    Extract spike count features from the unsupervised SNN for every image in a dataset.

    Returns:
        features: Tensor of shape (num_images, n_neurons)
        labels: Tensor of shape (num_images,)
    """
    layer.eval = True
    feature_list = []
    label_list = []
    num_steps = int(T / dt)
    for img, label in dataset:
        img = img.squeeze().to(device)
        spike_train = poisson_encode(img, T, max_rate, dt, device)
        layer.reset()
        output_spike_count = torch.zeros(layer.n_neurons, device=device)
        for t in range(num_steps):
            input_spikes = spike_train[:, t]
            spikes = layer.forward(input_spikes, t)
            output_spike_count += spikes
        feature_list.append(output_spike_count.cpu())
        label_list.append(label)
    features = torch.stack(feature_list)
    labels = torch.tensor(label_list)
    return features, labels


# -------------------------------
# Supervised Training Phase (Linear Readout)
# -------------------------------
class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def supervised_training(features, labels, num_epochs=20, lr=0.01):
    """
    Train a simple linear classifier on the extracted features.

    Args:
        features: Tensor of shape (num_images, feature_dim).
        labels: Tensor of shape (num_images,).
        num_epochs (int): Number of epochs.
        lr (float): Learning rate.

    Returns:
        model: Trained linear classifier.
    """
    input_dim = features.shape[1]
    model = LinearClassifier(input_dim)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(features.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        # Compute training accuracy
        _, predicted = torch.max(outputs, 1)
        correct = (predicted.cpu() == labels).sum().item()
        accuracy = 100.0 * correct / labels.size(0)
        print(f"[Supervised] Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    return model


def evaluate_supervised(model, features, labels):
    """
    Evaluate the supervised classifier on the test set,
    store the true labels and predicted labels globally, and display a confusion matrix.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(features.to(device))
        _, predicted = torch.max(outputs, 1)

        # Store predictions and labels as globals (converted to Python lists)
        global global_true_labels, global_predicted_labels
        global_true_labels = labels.cpu().numpy().tolist()
        global_predicted_labels = predicted.cpu().numpy().tolist()

        # Compute confusion matrix using scikit-learn's confusion_matrix function:
        cm = confusion_matrix(global_true_labels, global_predicted_labels)

        # Plot the confusion matrix.
        plt.figure(figsize=(10, 8))
        # MNIST classes: '0' through '9'
        classes = [str(i) for i in range(10)]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix for Classifier")
        plt.show()

        # Optionally, compute and print accuracy:
        correct = (predicted.cpu() == labels).sum().item()
        accuracy = 100.0 * correct / labels.size(0)
        print(f"[Supervised] Evaluation Accuracy: {accuracy:.2f}%")

    return accuracy


# -------------------------------
# Raster Plot Visualization for a Sample Image
# -------------------------------
def visualize_raster(layer, img, T=T, dt=dt, max_rate=max_rate, device=device):
    """
    Visualize spiking activity (raster plot) for a given image.
    """
    img = img.squeeze().to(device)
    spike_train = poisson_encode(img, T, max_rate, dt, device)
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
    # Define data augmentation + transformation for MNIST.
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # --- Phase 1: Unsupervised Training (STDP) ---
    n_unsupervised_neurons = 100  # Increase number of neurons for a richer feature set.
    unsup_epochs = 10  # Increase as needed for your experiment.
    start_time = time.time()
    unsup_layer = unsupervised_training(train_dataset, n_neurons=n_unsupervised_neurons, num_epochs=unsup_epochs, T=T,
                                        dt=dt, max_rate=max_rate, device=device)
    print(f"\nUnsupervised training completed in {(time.time() - start_time):.2f} seconds.")

    # --- Phase 2: Feature Extraction ---
    print("\nExtracting features from unsupervised layer (training set)...")
    train_features, train_labels = extract_features(train_dataset, unsup_layer, T=T, dt=dt, max_rate=max_rate,
                                                    device=device)
    print(f"Extracted feature shape: {train_features.shape}")

    print("\nExtracting features from unsupervised layer (test set)...")
    test_features, test_labels = extract_features(test_dataset, unsup_layer, T=T, dt=dt, max_rate=max_rate,
                                                  device=device)
    print(f"Extracted feature shape: {test_features.shape}")

    # --- Phase 3: Supervised Training of the Linear Classifier ---
    sup_epochs = 20
    lr = 0.01
    print("\nTraining supervised read-out (linear classifier) on extracted features...")
    classifier = supervised_training(train_features, train_labels, num_epochs=sup_epochs, lr=lr)

    # Evaluate the classifier on test features.
    evaluate_supervised(classifier, test_features, test_labels)

    # --- Visualization: Raster Plot for a Sample Image ---
    sample_img, sample_label = test_dataset[0]
    print(f"\nVisualizing spiking activity for sample image with true label: {sample_label}")
    visualize_raster(unsup_layer, sample_img, T=T, dt=dt, max_rate=max_rate, device=device)