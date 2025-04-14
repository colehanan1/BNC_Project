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
from torch.utils.data import DataLoader

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
dt = 2.0  # Time step (ms)
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


def poisson_encode_batch(images, T, max_rate, dt, device):
    """
    Batch version of Poisson encoding with DoG filtering.

    Args:
        images (torch.Tensor): Tensor of shape (B, 1, H, W).
        T (int): Total simulation time (ms).
        max_rate (float): Maximum firing rate (Hz).
        dt (float): Time step (ms).
        device (torch.device): Device to run on.

    Returns:
        torch.Tensor: Spike train tensor of shape (B, num_pixels, T).
    """
    batch_size = images.shape[0]
    # Remove the channel dimension and convert each image using DoG
    images = images.squeeze(1)  # shape: (B, H, W)
    spike_trains = []
    for b in range(batch_size):
        # Apply DoG filtering on CPU (or vectorize this with torch if you prefer)
        img_np = images[b].cpu().numpy()
        filtered = dog_filter(img_np, sigma1, sigma2)
        filtered_tensor = torch.tensor(filtered, dtype=torch.float32, device=device)
        flat = filtered_tensor.view(-1)  # shape: (num_pixels,)
        probability = flat * max_rate * dt / 1000.0  # Convert to probability
        rand_vals = torch.rand(flat.size(0), T, device=device)
        spike_train = (rand_vals < probability.unsqueeze(1)).float()  # shape: (num_pixels, T)
        spike_trains.append(spike_train)
    return torch.stack(spike_trains)  # shape: (B, num_pixels, T)


# -------------------------------
# LIF Neuron Layer with STDP (Unsupervised Layer)
# -------------------------------
class LIFNeuronLayer:
    def __init__(self, n_neurons, input_size, dt=2.0, tau_m=20.0, V_thresh=1.0, V_reset=0.0, device=device):
        self.device = device
        self.n_neurons = n_neurons
        self.input_size = input_size
        self.dt = dt
        self.tau_m = tau_m
        self.V_thresh = V_thresh
        self.V_reset = V_reset
        # We don't initialize V here since it will be created per batch.
        self.weights = torch.rand(n_neurons, input_size, device=device) * 0.1
        # For batched simulation, we maintain last spike times for each sample separately.
        # To keep it simple, we can update these per batch during the simulation.
        self.last_post_spike = None  # will be set during batch simulation
        self.last_pre_spike = None
        self.A_plus = 0.01
        self.A_minus = -0.012
        self.tau_plus = 20.0
        self.tau_minus = 20.0

    def reset_batch(self, batch_size):
        """Reset membrane potentials and last spike times for a batch."""
        self.V = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.last_post_spike = torch.full((batch_size, self.n_neurons), -1e8, device=self.device)
        self.last_pre_spike = torch.full((batch_size, self.input_size), -1e8, device=self.device)

    def forward_batch(self, input_spikes, t):
        """
        Vectorized forward pass for a batch.

        Args:
            input_spikes (torch.Tensor): Tensor of shape (B, input_size) for current time step.
            t (int): current simulation time step.

        Returns:
            torch.Tensor: Output spikes for the batch, shape (B, n_neurons).
        """
        # Compute the input current for the batch: shape (B, n_neurons)
        I = torch.matmul(input_spikes, self.weights.t())
        self.V = self.V + self.dt * ((-self.V / self.tau_m) + I)
        # Determine spikes: shape (B, n_neurons)
        spikes = (self.V >= self.V_thresh).float()
        # Reset neurons that fired
        mask = spikes.bool()
        self.V[mask] = self.V_reset
        # Update last_post_spike for cells that fired
        self.last_post_spike[mask] = t
        return spikes

    def update_stdp_batch(self, input_spikes, spikes, t):
        """
        A simplified vectorized STDP update for a batch.
        Note: Fully vectorizing STDP can be tricky. Here we compute a batch-averaged update.

        Args:
            input_spikes (torch.Tensor): Batch tensor, shape (B, input_size).
            spikes (torch.Tensor): Batch tensor, shape (B, n_neurons).
            t (int): current simulation time.
        """
        B = input_spikes.shape[0]
        # Create a mask where both input and output fired: shape (B, n_neurons, input_size)
        input_expanded = input_spikes.unsqueeze(1).expand(B, self.n_neurons, self.input_size)
        # For a proper implementation, you’d calculate the time difference per sample.
        # Here we compute an average time difference along the batch:
        dt_diff = t - self.last_pre_spike  # shape (B, input_size)
        dt_diff = dt_diff.unsqueeze(1)  # shape (B, 1, input_size)
        # Create STDP updates for the batch:
        # For simplicity, we treat all dt_diff >= 0 as potentiation:
        pos_mask = (dt_diff >= 0).float()
        # Compute weight updates (this is a simplification):
        dw = self.A_plus * torch.exp(-dt_diff / self.tau_plus) * pos_mask
        # Mask by where both input and output spiked:
        spike_mask = input_expanded * spikes.unsqueeze(2)  # shape (B, n_neurons, input_size)
        dw = dw * spike_mask
        # Average across the batch dimension:
        avg_dw = dw.mean(dim=0)  # shape (n_neurons, input_size)
        self.weights = self.weights + avg_dw
        # Ensure non-negative weights
        self.weights = torch.clamp(self.weights, min=0)
        # Update last_pre_spike (for batch, average update)
        fired_mask = (input_spikes == 1).float()
        self.last_pre_spike = self.last_pre_spike * (1 - fired_mask) + t * fired_mask

# -------------------------------
# Unsupervised Training Phase (STDP)
# -------------------------------
def unsupervised_training_batched(train_dataset, n_neurons=100, num_epochs=1, T=T, dt=dt, max_rate=max_rate,
                                  device=device, batch_size=64):
    """
    Unsupervised training using batched simulation.
    """
    input_size = 28 * 28  # for MNIST
    num_steps = int(T / dt)
    layer = LIFNeuronLayer(n_neurons=n_neurons, input_size=input_size, dt=dt, device=device)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_spike_sum = 0.0
        num_batches = 0

        print(f"\n--- Unsupervised Training Epoch {epoch + 1}/{num_epochs} ---")
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            # imgs shape: (B, 1, 28, 28)
            imgs = imgs.to(device)
            # Batch-poisson encode: shape (B, 784, T)
            spike_trains = poisson_encode_batch(imgs, T, max_rate, dt, device)
            B = spike_trains.shape[0]
            # Reset layer state for the entire batch:
            layer.reset_batch(B)
            output_spike_count = torch.zeros(B, n_neurons, device=device)

            for t in range(num_steps):
                # Get batched input spikes: shape (B, 784)
                input_spikes = spike_trains[:, :, t]
                # Forward pass: shape (B, n_neurons)
                spikes = layer.forward_batch(input_spikes, t)
                # Batch STDP update:
                layer.update_stdp_batch(input_spikes, spikes, t)
                output_spike_count += spikes

            batch_spike_sum = output_spike_count.sum().item()
            epoch_spike_sum += batch_spike_sum
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f"  Processed batch {batch_idx}/{len(dataloader)}")
        avg_spikes = epoch_spike_sum / (num_batches * B)
        weight_norm = torch.norm(layer.weights).item() / (n_neurons * input_size)
        epoch_duration = time.time() - epoch_start_time

        print(f"\nEpoch {epoch + 1} complete in {epoch_duration:.2f} sec.")
        print(f"   Average total spikes per image: {avg_spikes:.2f}")
        print(f"   Average weight norm: {weight_norm:.6f}")

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
        spike_train = poisson_encode_batch(img, T, max_rate, dt, device)
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
    spike_train = poisson_encode_batch(img, T, max_rate, dt, device)
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
    unsup_layer = unsupervised_training_batched(train_dataset, n_neurons=n_unsupervised_neurons,
                                                num_epochs=unsup_epochs, T=T,
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