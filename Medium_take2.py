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
from sklearn.decomposition import PCA

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
max_rate = 100.0  # Maximum firing rate used in Poisson encoding

# DoG filter parameters for retina-inspired encoding
sigma1 = 1.0
sigma2 = 2.0

# Global variables for debugging outputs
global_true_labels = []
global_predicted_labels = []


# -------------------------------
# Utility Functions
# -------------------------------
def dog_filter(image, sigma1=1.0, sigma2=2.0):
    """Apply Difference-of-Gaussians (DoG) filtering."""
    smooth1 = gaussian_filter(image, sigma=sigma1)
    smooth2 = gaussian_filter(image, sigma=sigma2)
    filtered = smooth1 - smooth2
    filtered = filtered - filtered.min()
    if filtered.max() > 0:
        filtered = filtered / filtered.max()
    return filtered


def poisson_encode_batch(images, T, max_rate, dt, device):
    """
    Batch Poisson encoding with DoG filtering.
    Expects images of shape (B, 1, H, W); returns tensor of shape (B, H*W, T).
    """
    batch_size = images.shape[0]
    images = images.squeeze(1)  # (B, H, W)
    spike_trains = []
    for b in range(batch_size):
        img_np = images[b].cpu().numpy()  # (H, W)
        filtered = dog_filter(img_np, sigma1, sigma2)
        filtered_tensor = torch.tensor(filtered, dtype=torch.float32, device=device)
        flat = filtered_tensor.view(-1)  # (H*W,)
        probability = flat * max_rate * dt / 1000.0
        rand_vals = torch.rand(flat.size(0), T, device=device)
        spike_train = (rand_vals < probability.unsqueeze(1)).float()
        spike_trains.append(spike_train)
    return torch.stack(spike_trains)


def poisson_encode(image, T, max_rate, dt, device):
    """
    Single-image Poisson encoding.
    Expects image as 2D tensor (H, W); returns tensor of shape (H*W, T).
    """
    flat = image.view(-1)
    probability = flat * max_rate * dt / 1000.0
    rand_vals = torch.rand(flat.size(0), T, device=device)
    spike_train = (rand_vals < probability.unsqueeze(1)).float()
    return spike_train


def plot_features_PCA(features, labels):
    """Plot a PCA scatter plot of the features colored by label."""
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features.numpy())
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels.numpy(), cmap='tab10', alpha=0.7)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA of Extracted Features")
    plt.colorbar(scatter, ticks=range(10), label="Digit")
    plt.show()


def compute_class_weights(labels):
    """Compute class weights as the inverse frequency of each class."""
    labels_np = labels.numpy()
    counts = np.bincount(labels_np)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# -------------------------------
# Single LIF Neuron Layer (used as building block)
# -------------------------------
class LIFNeuronLayer:
    def __init__(self, n_neurons, input_size, dt=2.0, tau_m=20.0, V_thresh_init=0.05, V_reset=0.0, device=device):
        self.device = device
        self.n_neurons = n_neurons
        self.input_size = input_size
        self.dt = dt
        self.tau_m = tau_m
        # Use a very low threshold to encourage firing
        self.V_thresh = torch.ones(n_neurons, device=device) * V_thresh_init
        self.V_reset = V_reset
        # Initialize weights with a larger scale to boost activation
        self.weights = torch.rand(n_neurons, input_size, device=device) * 1.0
        self.last_post_spike = None
        self.last_pre_spike = None
        # STDP parameters
        self.A_plus = 0.005
        self.A_minus = -0.015
        self.tau_plus = 20.0
        self.tau_minus = 20.0

    def reset_batch(self, batch_size):
        """Reset membrane potentials and spike timing for a batch."""
        self.V = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.last_post_spike = torch.full((batch_size, self.n_neurons), -1e8, device=self.device)
        self.last_pre_spike = torch.full((batch_size, self.input_size), -1e8, device=self.device)

    def forward_batch(self, input_spikes, t):
        """
        Batched forward pass with lateral inhibition.
        Input: (B, input_size), returns: (B, n_neurons).
        """
        I = self.gain * torch.matmul(input_spikes, self.weights.t())
        # Apply lateral inhibition (alpha = 0.05)
        alpha = 0.05
        inhibition = alpha * (I.sum(dim=1, keepdim=True) - I)
        I = I - inhibition
        self.V = self.V + self.dt * ((-self.V / self.tau_m) + I)
        thresh = self.V_thresh.unsqueeze(0).expand_as(self.V)
        spikes = (self.V >= thresh).float()
        mask = spikes.bool()
        self.V[mask] = self.V_reset
        self.last_post_spike[mask] = t
        return spikes

    def reset(self):
        self.reset_batch(1)

    def forward(self, input_spikes, t):
        return self.forward_batch(input_spikes.unsqueeze(0), t).squeeze(0)

    def update_stdp_batch(self, input_spikes, spikes, t):
        """Simplified vectorized STDP update for a batch."""
        B = input_spikes.shape[0]
        input_expanded = input_spikes.unsqueeze(1).expand(B, self.n_neurons, self.input_size)
        dt_diff = t - self.last_pre_spike  # (B, input_size)
        dt_diff = dt_diff.unsqueeze(1)  # (B, 1, input_size)
        pos_mask = (dt_diff >= 0).float()
        dw = self.A_plus * torch.exp(-dt_diff / self.tau_plus) * pos_mask
        spike_mask = input_expanded * spikes.unsqueeze(2)
        dw = dw * spike_mask
        avg_dw = dw.mean(dim=0)
        self.weights = self.weights + avg_dw
        self.weights = torch.clamp(self.weights, min=0)
        fired_mask = (input_spikes == 1).float()
        self.last_pre_spike = self.last_pre_spike * (1 - fired_mask) + t * fired_mask

    def normalize_weights(self):
        """Normalize each neuron's weight vector to unit norm."""
        norm = self.weights.norm(dim=1, keepdim=True) + 1e-8
        self.weights = self.weights / norm

    def update_thresholds(self, firing_rates, target_rate=20.0, lr_thresh=0.01):
        """Adaptively update thresholds based on firing rates (with a small noise term)."""
        adjustment = 1 + lr_thresh * (firing_rates - target_rate) / target_rate
        noise = torch.FloatTensor(self.n_neurons).uniform_(0.99, 1.01).to(self.device)
        self.V_thresh = self.V_thresh * adjustment * noise
        self.V_thresh = torch.clamp(self.V_thresh, min=0.3, max=1.0)


# -------------------------------
# Multi-Layer SNN (Two Layers)
# -------------------------------
class MultiLayerSNN:
    def __init__(self, n_neurons1, n_neurons2, input_size, dt, tau_m, V_thresh_init, V_reset, device):
        # Layer 1 processes the raw input (e.g., 28*28 pixels)
        self.layer1 = LIFNeuronLayer(n_neurons1, input_size, dt, tau_m, V_thresh_init, V_reset, device)
        # Set a gain for layer1 (we set it here so that it becomes an attribute)
        self.layer1.gain = 20.0
        # Layer 2 processes the output of layer 1 (with input size = n_neurons1)
        self.layer2 = LIFNeuronLayer(n_neurons2, n_neurons1, dt, tau_m, V_thresh_init, V_reset, device)
        self.layer2.gain = 20.0
        self.device = device

    def reset_batch(self, batch_size):
        self.layer1.reset_batch(batch_size)
        self.layer2.reset_batch(batch_size)

    def forward_batch(self, input_spikes, t):
        """Forward pass through both layers."""
        spikes1 = self.layer1.forward_batch(input_spikes, t)
        spikes2 = self.layer2.forward_batch(spikes1, t)
        return spikes2

    def update_stdp_batch(self, input_spikes, t):
        """Update STDP for both layers sequentially at time t."""
        # Update layer 1 using the raw input
        spikes1 = self.layer1.forward_batch(input_spikes, t)
        self.layer1.update_stdp_batch(input_spikes, spikes1, t)
        # Update layer 2 using layer 1's output
        spikes2 = self.layer2.forward_batch(spikes1, t)
        self.layer2.update_stdp_batch(spikes1, spikes2, t)

    def normalize_weights(self):
        self.layer1.normalize_weights()
        self.layer2.normalize_weights()

    def update_thresholds(self, firing_rates, target_rate=20.0, lr_thresh=0.01):
        self.layer1.update_thresholds(firing_rates, target_rate, lr_thresh)
        self.layer2.update_thresholds(firing_rates, target_rate, lr_thresh)


# -------------------------------
# Unsupervised Training Phase (Multi-Layer) with Debugging
# -------------------------------
def unsupervised_training_batched_multi(train_dataset, n_neurons1, n_neurons2, num_epochs, T, dt, max_rate,
                                        device, batch_size):
    """
    Unsupervised training for a two-layer SNN.
    Returns a MultiLayerSNN model.
    """
    input_size = 28 * 28  # MNIST
    num_steps = int(T / dt)
    model = MultiLayerSNN(n_neurons1, n_neurons2, input_size, dt, tau_m=20.0, V_thresh_init=0.05, V_reset=0.0,
                          device=device)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # We'll track outputs from the second layer as our final features.
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_spike_sum = 0.0
        num_batches = 0
        neuron_spike_sum = torch.zeros(n_neurons2, device=device)
        total_images = 0

        print(f"\n--- Multi-Layer Unsupervised Training Epoch {epoch + 1}/{num_epochs} ---")
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)  # (B, 1, 28, 28)
            spike_trains = poisson_encode_batch(imgs, T, max_rate, dt, device)  # (B, 784, T)
            B = imgs.shape[0]
            total_images += B
            model.reset_batch(B)
            # We'll accumulate output spikes from layer2 for feature extraction.
            output_spike_count = torch.zeros(B, n_neurons2, device=device)

            for t in range(num_steps):
                # For each time step, run layer1 and then layer2.
                input_spikes = spike_trains[:, :, t]  # (B, 784)
                spikes1 = model.layer1.forward_batch(input_spikes, t)
                model.layer1.update_stdp_batch(input_spikes, spikes1, t)
                spikes2 = model.layer2.forward_batch(spikes1, t)
                model.layer2.update_stdp_batch(spikes1, spikes2, t)
                output_spike_count += spikes2

            neuron_spike_sum += output_spike_count.sum(dim=0)
            epoch_spike_sum += output_spike_count.sum().item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f"  Processed batch {batch_idx}/{len(dataloader)}")

        avg_spikes = epoch_spike_sum / (num_batches * B)
        weight_norm1 = torch.norm(model.layer1.weights).item() / (n_neurons1 * (28 * 28))
        weight_norm2 = torch.norm(model.layer2.weights).item() / (n_neurons2 * n_neurons1)
        epoch_duration = time.time() - epoch_start_time
        firing_rates = neuron_spike_sum / total_images

        print(f"\nEpoch {epoch + 1} complete in {epoch_duration:.2f} sec.")
        print(f"   Average total spikes per image (layer2): {avg_spikes:.2f}")
        print(f"   Average weight norm layer1: {weight_norm1:.6f}, layer2: {weight_norm2:.6f}")
        print(
            f"   Mean firing rate per neuron (layer2): {firing_rates.mean().item():.2f} spikes/image (std: {firing_rates.std().item():.2f})")

        # Debug plots for layer2
        weight_norms2 = [torch.norm(model.layer2.weights[i]).item() for i in range(n_neurons2)]
        plt.figure()
        plt.hist(weight_norms2, bins=20)
        plt.title(f"Layer2 Weight Norms - Epoch {epoch + 1}")
        plt.xlabel("Weight Norm")
        plt.ylabel("Frequency")
        plt.show()

        plt.figure()
        plt.hist(firing_rates.cpu().numpy(), bins=20)
        plt.title(f"Layer2 Firing Rates - Epoch {epoch + 1}")
        plt.xlabel("Average Spikes per Image")
        plt.ylabel("Number of Neurons")
        plt.show()

        model.normalize_weights()
        model.update_thresholds(firing_rates, target_rate=20.0, lr_thresh=0.01)

    return model


# -------------------------------
# Feature Extraction – Batched (Multi-Layer)
# -------------------------------
def extract_features_batched_multi(dataset, model, T=T, dt=dt, max_rate=max_rate, device=device, batch_size=256):
    """
    Extract features from the multi-layer SNN.
    Returns features (num_images, n_neurons2) and labels.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feature_list = []
    label_list = []
    num_steps = int(T / dt)

    for imgs, labels in dataloader:
        imgs = imgs.to(device)  # (B, 1, 28, 28)
        spike_trains = poisson_encode_batch(imgs, T, max_rate, dt, device)  # (B, 784, T)
        B = imgs.shape[0]
        model.reset_batch(B)
        batch_output_count = torch.zeros(B, model.layer2.n_neurons, device=device)

        for t in range(num_steps):
            input_spikes = spike_trains[:, :, t]  # (B, 784)
            spikes1 = model.layer1.forward_batch(input_spikes, t)
            spikes2 = model.layer2.forward_batch(spikes1, t)
            batch_output_count += spikes2

        feature_list.append(batch_output_count.cpu())
        label_list.append(labels)

    features = torch.cat(feature_list, dim=0)
    labels = torch.cat(label_list, dim=0)
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


def supervised_training(features, labels, num_epochs=10, lr=0.01):
    """
    Train a linear classifier on the extracted features.
    Incorporates class weights to address imbalance.
    """
    input_dim = features.shape[1]
    model = LinearClassifier(input_dim)
    model.to(device)

    class_weights = compute_class_weights(labels)
    print("Class weights: ", class_weights)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.SGD(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(features.to(device))
        loss = criterion(outputs, labels.to(device))
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs, 1)
        correct = (predicted.cpu() == labels).sum().item()
        accuracy = 100.0 * correct / labels.size(0)
        print(f"[Supervised] Epoch {epoch + 1}/{num_epochs} - Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

    return model


def evaluate_supervised(model, features, labels):
    """
    Evaluate the classifier and display the confusion matrix.
    """
    model.eval()
    with torch.no_grad():
        outputs = model(features.to(device))
        _, predicted = torch.max(outputs, 1)

        global global_true_labels, global_predicted_labels
        global_true_labels = labels.cpu().numpy().tolist()
        global_predicted_labels = predicted.cpu().numpy().tolist()

        cm = confusion_matrix(global_true_labels, global_predicted_labels)
        plt.figure(figsize=(10, 8))
        classes = [str(i) for i in range(10)]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=classes, yticklabels=classes)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix for Classifier")
        plt.show()

        correct = (predicted.cpu() == labels).sum().item()
        accuracy = 100.0 * correct / labels.size(0)
        print(f"[Supervised] Evaluation Accuracy: {accuracy:.2f}%")

    return accuracy


# -------------------------------
# Raster Plot Visualization for a Sample Image
# -------------------------------
def visualize_raster(model, img, T=T, dt=dt, max_rate=max_rate, device=device):
    """
    Visualize spiking activity (raster plot) for a single image.
    """
    img = img.squeeze().to(device)
    spike_train = poisson_encode(img, T, max_rate, dt, device)  # (784, T)
    num_steps = int(T / dt)
    model.reset_batch(1)
    # We will use the output from layer2
    spikes_over_time = torch.zeros(model.layer2.n_neurons, num_steps, device=device)

    for t in range(num_steps):
        input_spikes = spike_train[:, t]  # (784,)
        spikes1 = model.layer1.forward(input_spikes, t)
        spikes2 = model.layer2.forward(spikes1, t)
        spikes_over_time[:, t] = spikes2
    spikes_over_time = spikes_over_time.cpu().numpy()

    plt.figure(figsize=(10, 5))
    for neuron in range(model.layer2.n_neurons):
        times = [t for t in range(num_steps) if spikes_over_time[neuron, t] > 0]
        if times:
            plt.vlines(times, neuron + 0.5, neuron + 1.5)
    plt.xlabel("Time (ms)")
    plt.ylabel("Neuron index")
    plt.title("Raster Plot of Output Neuron Activity (Layer 2)")
    plt.show()


# -------------------------------
# Main Script Execution
# -------------------------------
if __name__ == "__main__":
    # Data Augmentation and Transformation for MNIST
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # Phase 1: Unsupervised Training for Multi-Layer SNN
    n_neurons1 = 100  # Neurons in layer 1
    n_neurons2 = 50  # Neurons in layer 2 (final feature layer)
    unsup_epochs = 10
    batch_size = 256  # Set batch size to 256
    start_time = time.time()
    multi_snn = unsupervised_training_batched_multi(train_dataset, n_neurons1, n_neurons2,
                                                    num_epochs=unsup_epochs, T=T, dt=dt, max_rate=max_rate,
                                                    device=device, batch_size=batch_size)
    print(f"\nUnsupervised training completed in {(time.time() - start_time):.2f} seconds.")

    # Phase 2: Feature Extraction (Multi-Layer SNN)
    print("\nExtracting features from multi-layer SNN (training set)...")
    train_features, train_labels = extract_features_batched_multi(train_dataset, multi_snn, T=T, dt=dt,
                                                                  max_rate=max_rate, device=device,
                                                                  batch_size=batch_size)
    print(f"Extracted training feature shape: {train_features.shape}")

    print("\nExtracting features from multi-layer SNN (test set)...")
    test_features, test_labels = extract_features_batched_multi(test_dataset, multi_snn, T=T, dt=dt,
                                                                max_rate=max_rate, device=device, batch_size=batch_size)
    print(f"Extracted test feature shape: {test_features.shape}")

    # Debug: PCA visualization of extracted features
    plot_features_PCA(train_features, train_labels)

    # Phase 3: Supervised Training (Linear Readout)
    sup_epochs = 10
    lr = 0.01
    print("\nTraining supervised read-out (linear classifier) on extracted features...")
    classifier = supervised_training(train_features, train_labels, num_epochs=sup_epochs, lr=lr)
    evaluate_supervised(classifier, test_features, test_labels)

    # Phase 4: Raster Plot Visualization for a Sample Image
    sample_img, sample_label = test_dataset[0]
    print(f"\nVisualizing spiking activity for sample image with true label: {sample_label}")
    visualize_raster(multi_snn, sample_img, T=T, dt=dt, max_rate=max_rate, device=device)