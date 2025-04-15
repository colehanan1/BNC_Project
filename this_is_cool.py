import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import gc


# =============================================================================
# Batched Poisson Encoding Function
# =============================================================================
def poisson_encode_batch(images, T=100, max_rate=100):
    """
    Vectorized Poisson encoding for a batch of flattened images.

    Args:
        images: Tensor of shape (B, input_size) with pixel values in [0, 1]
        T: Number of timesteps.
        max_rate: Maximum firing rate scaling factor.

    Returns:
        spikes: Tensor of shape (T, B, input_size) with binary spikes (0/1)
    """
    B, input_size = images.shape
    # Create random numbers on same device.
    rand_tensor = torch.rand(T, B, input_size, device=images.device)
    images_expanded = images.unsqueeze(0)  # shape: (1, B, input_size)
    spikes = (rand_tensor < (images_expanded * max_rate)).float()
    return spikes


# =============================================================================
# Batched Dynamic Threshold LIF Neuron Model with Forward-Step
# =============================================================================
class BatchedDynamicThresholdLIF(nn.Module):
    def __init__(self, num_neurons, tau_m=20.0, v_rest=-65.0, threshold0=-50.0,
                 tau_thresh=100.0, beta=5.0, dt=1.0):
        """
        num_neurons: Number of neurons in the layer.
        tau_m: Membrane time constant.
        v_rest: Resting membrane potential.
        threshold0: Baseline threshold value.
        tau_thresh: Time constant for threshold recovery.
        beta: Increment added to threshold upon spiking.
        dt: Time step.
        """
        super(BatchedDynamicThresholdLIF, self).__init__()
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.threshold0 = threshold0
        self.tau_thresh = tau_thresh
        self.beta = beta
        self.dt = dt
        # State variables: will be initialized for each batch during forward.
        self.membrane_potential = None  # Shape: (B, num_neurons)
        self.dynamic_threshold = None  # Shape: (B, num_neurons)

    def init_state(self, batch_size, device):
        self.membrane_potential = torch.full((batch_size, self.num_neurons), self.v_rest, device=device)
        self.dynamic_threshold = torch.full((batch_size, self.num_neurons), self.threshold0, device=device)

    def forward_step(self, net_current):
        """
        Performs a single time step update.

        Args:
            net_current: Tensor of shape (B, num_neurons) representing the net input current.

        Returns:
            current_spikes: Tensor of shape (B, num_neurons) of binary spikes.
        """
        B = net_current.size(0)
        if self.membrane_potential is None or self.membrane_potential.size(0) != B:
            self.init_state(B, net_current.device)
        # Update membrane potential
        self.membrane_potential = self.membrane_potential + (
                    self.v_rest - self.membrane_potential) / self.tau_m + net_current * self.dt
        # Determine spikes based on dynamic threshold
        current_spikes = (self.membrane_potential >= self.dynamic_threshold).float()
        # Update dynamic threshold: increase by beta where spikes occur and decay toward baseline.
        self.dynamic_threshold = self.dynamic_threshold + self.beta * current_spikes - (
                    (self.dynamic_threshold - self.threshold0) / self.tau_thresh) * self.dt
        # Reset the membrane potential for neurons that spiked.
        self.membrane_potential = torch.where(current_spikes.bool(),
                                              torch.full_like(self.membrane_potential, self.v_rest),
                                              self.membrane_potential)
        return current_spikes

    def reset_state(self):
        if self.membrane_potential is not None:
            self.membrane_potential.fill_(self.v_rest)
            self.dynamic_threshold.fill_(self.threshold0)


# =============================================================================
# Batched SNN Model with Recurrent Inhibition and Dynamic Threshold
# =============================================================================
class SNNModel(nn.Module):
    def __init__(self, input_size=784, num_neurons=100, output_size=10, T=100):
        """
        input_size: Dimension of flattened MNIST image (28x28=784).
        num_neurons: Number of hidden neurons.
        output_size: Number of output classes (10 for MNIST).
        T: Number of timesteps for simulation.
        """
        super(SNNModel, self).__init__()
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.output_size = output_size
        self.T = T

        # Learnable weight matrix mapping input to hidden neurons.
        self.input_weights = nn.Parameter(torch.rand(input_size, num_neurons))
        # Hidden layer: batched dynamic threshold LIF neurons.
        self.hidden_layer = BatchedDynamicThresholdLIF(num_neurons)
        # Fixed recurrent inhibitory matrix: shape (num_neurons, num_neurons).
        # Off-diagonals set to -0.5, diagonals 0.
        W_rec = -0.5 * (torch.ones(num_neurons, num_neurons) - torch.eye(num_neurons))
        self.register_buffer("W_rec", W_rec)
        # Learnable weight matrix mapping hidden spikes to output logits.
        self.output_weights = nn.Parameter(torch.rand(num_neurons, output_size))

    def forward(self, input_spikes):
        """
        Simulates the network for T timesteps.

        Args:
            input_spikes: Tensor of shape (T, B, input_size)

        Returns:
            logits: Tensor of shape (B, output_size)
            hidden_spike_record: Tensor of shape (T, B, num_neurons) (for visualization)
        """
        T, B, _ = input_spikes.shape
        self.hidden_layer.reset_state()
        previous_hidden = torch.zeros(B, self.num_neurons, device=input_spikes.device)
        hidden_spike_record = []

        for t in range(T):
            # Compute feedforward input: shape (B, num_neurons)
            input_current = torch.matmul(input_spikes[t], self.input_weights)
            # Compute recurrent inhibitory input from previous hidden spikes.
            recurrent_current = torch.matmul(previous_hidden, self.W_rec)
            # Total net current:
            net_current = input_current + recurrent_current
            # Single timestep update:
            current_hidden = self.hidden_layer.forward_step(net_current)
            hidden_spike_record.append(current_hidden)
            previous_hidden = current_hidden

        hidden_spikes_tensor = torch.stack(hidden_spike_record, dim=0)  # shape: (T, B, num_neurons)
        spike_counts = hidden_spikes_tensor.sum(dim=0)  # shape: (B, num_neurons)
        logits = torch.matmul(spike_counts, self.output_weights)  # shape: (B, output_size)
        return logits, hidden_spikes_tensor

    def reset(self):
        self.hidden_layer.reset_state()


# =============================================================================
# Data Loading (MNIST 28x28)
# =============================================================================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
batch_size = 16
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# For troubleshooting multiprocessing issues on macOS, you can set num_workers=0.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

# =============================================================================
# Device Setup: Use MPS (M2 Apple GPU)
# =============================================================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# =============================================================================
# Initialize Model, Loss Function, and Optimizer
# =============================================================================
model = SNNModel(input_size=784, num_neurons=100, output_size=10, T=100).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


# =============================================================================
# Training Function (Batched)
# =============================================================================
def train_snn(model, dataloader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for idx, (imgs, labels) in enumerate(dataloader):
            imgs = imgs.view(imgs.size(0), -1).to(device)  # shape: (B, 784)
            input_spikes = poisson_encode_batch(imgs, T=model.T, max_rate=100)  # shape: (T, B, 784)
            model.reset()
            logits, _ = model(input_spikes)
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] Batch [{idx + 1}/{len(dataloader)}] Loss: {total_loss / (idx + 1):.4f}")
            gc.collect()
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {total_loss / len(dataloader):.4f}")


# =============================================================================
# Evaluation Function (Batched) & Visualization
# =============================================================================
def evaluate_snn(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    hidden_patterns = []  # For storing hidden spike count patterns.
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            input_spikes = poisson_encode_batch(imgs, T=model.T, max_rate=100)
            model.reset()
            logits, hidden_spike_record = model(input_spikes)
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
            spike_counts = hidden_spike_record.sum(dim=0)  # shape: (B, num_neurons)
            hidden_patterns.append(spike_counts.cpu())
    accuracy = 100 * correct / total
    print(f"SNN Accuracy on test set: {accuracy:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=testset.classes, yticklabels=testset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

    hidden_patterns = torch.cat(hidden_patterns, dim=0)  # shape: (N_total, num_neurons)
    plt.figure(figsize=(8, 4))
    plt.imshow(hidden_patterns.T, aspect='auto', cmap='viridis')
    plt.colorbar(label='Spike Count')
    plt.xlabel("Sample index")
    plt.ylabel("Hidden neuron index")
    plt.title("Hidden Layer Spike Count Patterns")
    plt.show()


# =============================================================================
# Main Execution
# =============================================================================
if __name__ == "__main__":
    # It's crucial to protect the main module for multiprocessing (especially on macOS)
    train_snn(model, trainloader, num_epochs=2)
    evaluate_snn(model, testloader)