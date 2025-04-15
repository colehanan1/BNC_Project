import numpy as np
import torch
import torch.nn as nn
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
        images: Tensor of shape (B, input_size) with pixel values assumed to be normalized in [0, 1].
        T: Number of timesteps.
        max_rate: Scaling factor for spike probability.

    Returns:
        spikes: Tensor of shape (T, B, input_size) with binary (0/1) spike values.
    """
    B, input_size = images.shape
    # Create a random tensor of shape (T, B, input_size) on the same device as images.
    rand_tensor = torch.rand(T, B, input_size, device=images.device)
    # Expand images to shape (T, B, input_size)
    images_expanded = images.unsqueeze(0)  # shape: (1, B, input_size)
    spikes = (rand_tensor < (images_expanded * max_rate)).float()
    return spikes


# =============================================================================
# Batched Dynamic Threshold LIF Neuron Model
# =============================================================================
class BatchedDynamicThresholdLIF(nn.Module):
    def __init__(self, num_neurons, tau_m=20.0, v_rest=-65.0, threshold0=-50.0,
                 tau_thresh=100.0, beta=5.0):
        """
        num_neurons: Number of neurons in this layer.
        tau_m: Membrane time constant.
        v_rest: Resting membrane potential.
        threshold0: Baseline threshold value.
        tau_thresh: Time constant for threshold recovery.
        beta: Increase in threshold following a spike.
        """
        super(BatchedDynamicThresholdLIF, self).__init__()
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.v_rest = v_rest
        self.threshold0 = threshold0
        self.tau_thresh = tau_thresh
        self.beta = beta
        # State will be initialized in the forward pass
        self.membrane_potential = None  # shape: (B, num_neurons)
        self.dynamic_threshold = None  # shape: (B, num_neurons)

    def init_state(self, batch_size, device):
        self.membrane_potential = torch.full((batch_size, self.num_neurons), self.v_rest, device=device)
        self.dynamic_threshold = torch.full((batch_size, self.num_neurons), self.threshold0, device=device)

    def forward(self, weighted_input, dt=1.0):
        """
        weighted_input: Tensor of shape (T, B, num_neurons) representing the input current for each timestep.
        dt: Time step.

        Returns:
            spikes_tensor: Tensor of shape (T, B, num_neurons) with binary spikes.
        """
        T, B, N = weighted_input.shape
        # Initialize state if needed
        if self.membrane_potential is None or self.membrane_potential.size(0) != B:
            self.init_state(B, weighted_input.device)

        spikes_list = []
        for t in range(T):
            # Update membrane potential with leak and input:
            self.membrane_potential = self.membrane_potential + \
                                      (self.v_rest - self.membrane_potential) / self.tau_m + weighted_input[t]
            # Determine which neurons spike at this timestep:
            current_spikes = (self.membrane_potential >= self.dynamic_threshold).float()
            spikes_list.append(current_spikes)
            # Update dynamic threshold: increase where spikes occurred and decay toward baseline.
            self.dynamic_threshold = self.dynamic_threshold + self.beta * current_spikes - \
                                     ((self.dynamic_threshold - self.threshold0) / self.tau_thresh) * dt
            # Reset membrane potentials of neurons that spiked.
            self.membrane_potential = torch.where(current_spikes.bool(),
                                                  torch.full_like(self.membrane_potential, self.v_rest),
                                                  self.membrane_potential)
        spikes_tensor = torch.stack(spikes_list, dim=0)  # shape: (T, B, num_neurons)
        return spikes_tensor

    def reset_state(self):
        if self.membrane_potential is not None:
            self.membrane_potential.fill_(self.v_rest)
            self.dynamic_threshold.fill_(self.threshold0)


# =============================================================================
# Batched SNN Model with Dynamic Threshold
# =============================================================================
class SNNModel(nn.Module):
    def __init__(self, input_size=784, num_neurons=100, output_size=10, T=100):
        """
        input_size: Dimension of flattened MNIST image (28x28=784).
        num_neurons: Number of neurons in the hidden layer.
        output_size: Number of output classes (10 for MNIST).
        T: Number of timesteps for simulation.
        """
        super(SNNModel, self).__init__()
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.output_size = output_size
        self.T = T

        # Learnable weights mapping input (784) to hidden neurons (100)
        self.input_weights = nn.Parameter(torch.rand(input_size, num_neurons))
        # Hidden layer: dynamic threshold neurons (batched).
        self.hidden_layer = BatchedDynamicThresholdLIF(num_neurons)
        # Learnable output weights mapping hidden spike counts to logits.
        self.output_weights = nn.Parameter(torch.rand(num_neurons, output_size))

    def forward(self, input_spikes):
        """
        input_spikes: Tensor of shape (T, B, input_size) representing the spike train for each sample.

        Returns:
            logits: Tensor of shape (B, output_size)
        """
        # Compute weighted input: multiply each timestep's input with the input_weights.
        # weighted_input shape: (T, B, num_neurons)
        weighted_input = torch.matmul(input_spikes, self.input_weights)
        # Pass weighted input through the dynamic threshold hidden layer.
        hidden_spikes = self.hidden_layer(weighted_input)  # shape: (T, B, num_neurons)
        # Sum over the time dimension to get spike counts per neuron per sample.
        spike_counts = hidden_spikes.sum(dim=0)  # shape: (B, num_neurons)
        # Compute output logits via the output weights.
        output = torch.matmul(spike_counts, self.output_weights)  # shape: (B, output_size)
        return output

    def reset(self):
        self.hidden_layer.reset_state()


# =============================================================================
# Data Loading (MNIST 28x28)
# =============================================================================
transform = transforms.Compose([
    transforms.ToTensor(),  # MNIST images are 1x28x28
    transforms.Normalize((0.5,), (0.5,))
])
batch_size = 64
trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

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
            # imgs: shape (B, 1, 28, 28) -> flatten to (B, 784)
            imgs = imgs.view(imgs.size(0), -1).to(device)
            # Poisson encode the batch: output shape (T, B, 784)
            input_spikes = poisson_encode_batch(imgs, T=model.T, max_rate=100)

            # Reset the model's state
            model.reset()

            # Forward pass: get output logits of shape (B, output_size)
            logits = model(input_spikes)
            loss = criterion(logits, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            gc.collect()

            if (idx + 1) % 10 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}] Batch [{idx + 1}/{len(dataloader)}] Loss: {total_loss / (idx + 1):.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {total_loss / len(dataloader):.4f}")


# =============================================================================
# Evaluation Function (Batched)
# =============================================================================
def evaluate_snn(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.view(imgs.size(0), -1).to(device)
            input_spikes = poisson_encode_batch(imgs, T=model.T, max_rate=100)
            model.reset()
            logits = model(input_spikes)
            predicted = torch.argmax(logits, dim=1)
            total += labels.size(0)
            correct += (predicted.cpu() == labels).sum().item()
            all_preds.extend(predicted.cpu().tolist())
            all_labels.extend(labels.tolist())
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


# =============================================================================
# Run Training and Evaluation
# =============================================================================
train_snn(model, trainloader, num_epochs=10)
evaluate_snn(model, testloader)
