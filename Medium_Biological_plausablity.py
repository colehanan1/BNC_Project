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
max_rate = 100.0  # Maximum firing rate (Hz)

# DoG filter parameters for retina-inspired encoding
sigma1 = 1.0
sigma2 = 2.0

# Global variables for debugging
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
    Batch version of Poisson encoding with DoG filtering.
    Input shape: (B, 1, H, W). Returns tensor of shape (B, H*W, T).
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
    Expects 2D image tensor (H, W), returns tensor of shape (H*W, T).
    """
    flat = image.view(-1)
    probability = flat * max_rate * dt / 1000.0
    rand_vals = torch.rand(flat.size(0), T, device=device)
    spike_train = (rand_vals < probability.unsqueeze(1)).float()
    return spike_train


def plot_features_PCA(features, labels):
    """Plot 2D PCA scatter of extracted features colored by label."""
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features.numpy())
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], c=labels.numpy(), cmap='tab10', alpha=0.7)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title('PCA of Extracted Features')
    plt.colorbar(scatter, ticks=range(10), label='Digit')
    plt.show()


def compute_class_weights(labels):
    """
    Compute class weights as the inverse frequency of each class.
    """
    labels_np = labels.numpy()
    counts = np.bincount(labels_np)
    weights = 1.0 / (counts + 1e-8)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32).to(device)


# -------------------------------
# LIF Neuron Layer with STDP, Lateral Inhibition, Adaptive Thresholds, and Weight Normalization
# -------------------------------
class LIFNeuronLayer:
    def __init__(self, n_neurons, input_size, dt=2.0, tau_m=20.0, V_thresh_init=0.1, V_reset=0.0, device=device):
        self.device = device
        self.n_neurons = n_neurons
        self.input_size = input_size
        self.dt = dt
        self.tau_m = tau_m
        # Lower threshold for easier firing
        self.V_thresh = torch.ones(n_neurons, device=device) * V_thresh_init
        self.V_reset = V_reset
        # Increased initial weight scale to [0, 1.0]
        self.weights = torch.rand(n_neurons, input_size, device=device) * 1.0
        self.last_post_spike = None
        self.last_pre_spike = None
        # Tuned STDP parameters
        self.A_plus = 0.005
        self.A_minus = -0.015
        self.tau_plus = 20.0
        self.tau_minus = 20.0
        # Add input gain to scale up the feedforward current
        self.gain = 10.0

    def reset_batch(self, batch_size):
        """Reset membrane potentials and spike timing for a batch."""
        self.V = torch.zeros(batch_size, self.n_neurons, device=self.device)
        self.last_post_spike = torch.full((batch_size, self.n_neurons), -1e8, device=self.device)
        self.last_pre_spike = torch.full((batch_size, self.input_size), -1e8, device=self.device)

    def forward_batch(self, input_spikes, t):
        """
        Batched forward pass with lateral inhibition.
        Input: (B, input_size). Returns: (B, n_neurons).
        """
        # Compute input current with gain
        I = self.gain * torch.matmul(input_spikes, self.weights.t())
        # Reduced lateral inhibition strength
        alpha = 0.05
        inhibition = alpha * (I.sum(dim=1, keepdim=True) - I)
        I = I - inhibition
        # Debug: Uncomment to print mean input current, if desired.
        # print("Mean input current:", I.mean().item())
        self.V = self.V + self.dt * ((-self.V / self.tau_m) + I)
        thresh = self.V_thresh.unsqueeze(0).expand_as(self.V)
        spikes = (self.V >= thresh).float()
        mask = spikes.bool()
        self.V[mask] = self.V_reset
        self.last_post_spike[mask] = t
        return spikes

    def reset(self):
        """Reset for a single sample."""
        self.reset_batch(1)

    def forward(self, input_spikes, t):
        """Single-sample forward pass."""
        return self.forward_batch(input_spikes.unsqueeze(0), t).squeeze(0)

    def update_stdp_batch(self, input_spikes, spikes, t):
        """
        Simplified vectorized STDP update for a batch.
        """
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
        """
        Adapt each neuron's threshold based on its average firing rate.
        """
        adjustment = 1 + lr_thresh * (firing_rates - target_rate) / target_rate
        self.V_thresh = self.V_thresh * adjustment
        self.V_thresh = torch.clamp(self.V_thresh, min=0.3, max=1.5)


# -------------------------------
# Unsupervised Training Phase (Batched) with Debugging and Adaptive Mechanisms
# -------------------------------
def unsupervised_training_batched(train_dataset, n_neurons=100, num_epochs=10, T=T, dt=dt, max_rate=max_rate,
                                  device=device, batch_size=128):
    """
    Unsupervised training using STDP with lateral inhibition, weight normalization, and adaptive thresholds.
    Extra debugging outputs are provided.
    """
    input_size = 28 * 28  # for MNIST
    num_steps = int(T / dt)
    layer = LIFNeuronLayer(n_neurons=n_neurons, input_size=input_size, dt=dt, device=device)
    dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        epoch_spike_sum = 0.0
        num_batches = 0
        neuron_spike_sum = torch.zeros(n_neurons, device=device)
        total_images = 0

        print(f"\n--- Unsupervised Training Epoch {epoch + 1}/{num_epochs} ---")
        for batch_idx, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.to(device)  # (B, 1, 28, 28)
            spike_trains = poisson_encode_batch(imgs, T, max_rate, dt, device)  # (B, 784, T)
            B = imgs.shape[0]
            total_images += B
            layer.reset_batch(B)
            output_spike_count = torch.zeros(B, n_neurons, device=device)

            for t in range(num_steps):
                input_spikes = spike_trains[:, :, t]  # (B, 784)
                spikes = layer.forward_batch(input_spikes, t)  # (B, n_neurons)
                layer.update_stdp_batch(input_spikes, spikes, t)
                output_spike_count += spikes

            neuron_spike_sum += output_spike_count.sum(dim=0)
            epoch_spike_sum += output_spike_count.sum().item()
            num_batches += 1

            if batch_idx % 50 == 0:
                print(f"  Processed batch {batch_idx}/{len(dataloader)}")

        avg_spikes = epoch_spike_sum / (num_batches * B)
        weight_norm = torch.norm(layer.weights).item() / (n_neurons * input_size)
        epoch_duration = time.time() - epoch_start_time
        firing_rates = neuron_spike_sum / total_images

        print(f"\nEpoch {epoch + 1} complete in {epoch_duration:.2f} sec.")
        print(f"   Average total spikes per image: {avg_spikes:.2f}")
        print(f"   Average weight norm: {weight_norm:.6f}")
        print(
            f"   Mean firing rate per neuron: {firing_rates.mean().item():.2f} spikes/image (std: {firing_rates.std().item():.2f})")

        # Debug Plot: Histogram of Weight Norms
        weight_norms = [torch.norm(layer.weights[i]).item() for i in range(n_neurons)]
        plt.figure()
        plt.hist(weight_norms, bins=20)
        plt.title(f"Weight Norms Histogram - Epoch {epoch + 1}")
        plt.xlabel("Weight Norm")
        plt.ylabel("Frequency")
        plt.show()

        # Debug Plot: Histogram of Neuron Firing Rates
        plt.figure()
        plt.hist(firing_rates.cpu().numpy(), bins=20)
        plt.title(f"Firing Rates Histogram - Epoch {epoch + 1}")
        plt.xlabel("Average Spikes per Image")
        plt.ylabel("Number of Neurons")
        plt.show()

        layer.normalize_weights()
        layer.update_thresholds(firing_rates, target_rate=20.0, lr_thresh=0.01)

    return layer


# -------------------------------
# Feature Extraction – Batched
# -------------------------------
def extract_features_batched(dataset, layer, T=T, dt=dt, max_rate=max_rate, device=device, batch_size=128):
    """
    Extract spike count features from the unsupervised SNN for every image using batch processing.
    Returns features (num_images, n_neurons) and labels.
    """
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    feature_list = []
    label_list = []
    num_steps = int(T / dt)

    for imgs, labels in dataloader:
        imgs = imgs.to(device)  # (B, 1, 28, 28)
        spike_trains = poisson_encode_batch(imgs, T, max_rate, dt, device)  # (B, 784, T)
        B = imgs.shape[0]
        layer.reset_batch(B)
        batch_output_count = torch.zeros(B, layer.n_neurons, device=device)

        for t in range(num_steps):
            input_spikes = spike_trains[:, :, t]  # (B, 784)
            spikes = layer.forward_batch(input_spikes, t)  # (B, n_neurons)
            batch_output_count += spikes

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
    Train a simple linear classifier on the extracted features.
    Incorporates class weighting to address imbalance if present.
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
    Evaluate the classifier on the test set and display a confusion matrix.
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
def visualize_raster(layer, img, T=T, dt=dt, max_rate=max_rate, device=device):
    """
    Visualize spiking activity (raster plot) for a single image.
    """
    img = img.squeeze().to(device)
    spike_train = poisson_encode(img, T, max_rate, dt, device)  # (784, T)
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
    # Data augmentation and transformation for MNIST.
    transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor()
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transforms.ToTensor())

    # Phase 1: Unsupervised Training (STDP with Adaptive Mechanisms)
    n_unsupervised_neurons = 100
    unsup_epochs = 10
    start_time = time.time()
    unsup_layer = unsupervised_training_batched(train_dataset, n_neurons=n_unsupervised_neurons,
                                                num_epochs=unsup_epochs, T=T, dt=dt, max_rate=max_rate,
                                                device=device, batch_size=128)
    print(f"\nUnsupervised training completed in {(time.time() - start_time):.2f} seconds.")

    # Phase 2: Feature Extraction (Batched)
    print("\nExtracting features from unsupervised layer (training set)...")
    train_features, train_labels = extract_features_batched(train_dataset, unsup_layer, T=T, dt=dt,
                                                            max_rate=max_rate, device=device, batch_size=128)
    print(f"Extracted training feature shape: {train_features.shape}")

    print("\nExtracting features from unsupervised layer (test set)...")
    test_features, test_labels = extract_features_batched(test_dataset, unsup_layer, T=T, dt=dt,
                                                          max_rate=max_rate, device=device, batch_size=128)
    print(f"Extracted test feature shape: {test_features.shape}")

    # Debug: PCA visualization of extracted features.
    plot_features_PCA(train_features, train_labels)

    # Phase 3: Supervised Training (Linear Classifier)
    sup_epochs = 10
    lr = 0.01
    print("\nTraining supervised read-out (linear classifier) on extracted features...")
    classifier = supervised_training(train_features, train_labels, num_epochs=sup_epochs, lr=lr)
    evaluate_supervised(classifier, test_features, test_labels)

    # Phase 4: Raster Plot Visualization for a Sample Image
    sample_img, sample_label = test_dataset[0]
    print(f"\nVisualizing spiking activity for sample image with true label: {sample_label}")
    visualize_raster(unsup_layer, sample_img, T=T, dt=dt, max_rate=max_rate, device=device)
