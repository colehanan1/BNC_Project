import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix


# ---------------------------------------------
# Poisson Encoding Function
# ---------------------------------------------
def poisson_encode(image, T=100, max_rate=100):
    """
    Encode a 1D numpy array (flattened image) into a (T, input_size) binary spike matrix.
    image: numpy array of shape (input_size,)
    Returns: numpy array of shape (T, input_size) with 0/1 values.
    """
    pixels = image.flatten()  # shape (input_size,)
    # Assuming pixel values normalized in [0,1]: use max_rate as the probability scaler.
    spikes = np.random.rand(T, pixels.size) < (pixels * max_rate)
    return spikes.astype(np.float32)


# ---------------------------------------------
# Leaky Integrate-and-Fire (LIF) Neuron Model
# ---------------------------------------------
class LIFNeuronModel(nn.Module):
    def __init__(self, num_neurons, tau_m=20.0, threshold=-50.0, v_rest=-65.0):
        """
        num_neurons: number of neurons in this layer
        tau_m: membrane time constant
        threshold: firing threshold
        v_rest: resting membrane potential
        """
        super(LIFNeuronModel, self).__init__()
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.threshold = threshold
        self.v_rest = v_rest
        # Register state as buffers so they move with the device.
        self.register_buffer("membrane_potential", torch.full((num_neurons,), v_rest))

    def forward(self, weighted_input, dt=1.0):
        """
        weighted_input: tensor of shape (T, num_neurons) representing the input current at each timestep.
        dt: time step size.
        Returns: spikes over T timesteps, tensor shape: (T, num_neurons)
        """
        T = weighted_input.size(0)
        spikes_list = []
        # Iterate over each time step
        for t in range(T):
            self.membrane_potential = self.membrane_potential + (self.v_rest - self.membrane_potential) / self.tau_m + \
                                      weighted_input[t] * dt
            current_spikes = (self.membrane_potential >= self.threshold).float()
            spikes_list.append(current_spikes)
            # Reset the membrane potential for neurons that fired
            self.membrane_potential = torch.where(current_spikes.bool(),
                                                  torch.full_like(self.membrane_potential, self.v_rest),
                                                  self.membrane_potential)
        spikes_tensor = torch.stack(spikes_list, dim=0)
        return spikes_tensor

    def reset_state(self):
        # Reset membrane potential to resting value.
        self.membrane_potential.fill_(self.v_rest)


# ---------------------------------------------
# Spiking Neural Network (SNN) Model
# ---------------------------------------------
class SNNModel(nn.Module):
    def __init__(self, input_size=784, num_neurons=100, output_size=10, T=100):
        """
        input_size: dimension of flattened MNIST image (28x28=784)
        num_neurons: number of neurons in the hidden spiking layer
        output_size: number of classes (10 for MNIST)
        T: number of timesteps for simulation per image
        """
        super(SNNModel, self).__init__()
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.output_size = output_size
        self.T = T

        # Learnable weights mapping input spikes (784) to hidden neurons (100)
        self.input_weights = nn.Parameter(torch.rand(input_size, num_neurons))
        # Hidden layer: LIF neurons
        self.hidden_layer = LIFNeuronModel(num_neurons)
        # Learnable weights mapping hidden spike counts to output logits (10 classes)
        self.output_weights = nn.Parameter(torch.rand(num_neurons, output_size))

    def forward(self, input_spikes):
        """
        input_spikes: tensor of shape (T, input_size)
        Returns: logits of shape (output_size,)
        """
        # Map input spikes to the hidden layer using the learnable input connection.
        weighted_input = torch.matmul(input_spikes, self.input_weights)  # shape: (T, num_neurons)
        # Process through LIF hidden layer to produce spikes.
        hidden_spikes = self.hidden_layer(weighted_input)  # shape: (T, num_neurons)
        # Sum spikes over time (activity) of each hidden neuron.
        spike_counts = hidden_spikes.sum(dim=0)  # shape: (num_neurons,)
        # Map spike counts to output logits.
        output = torch.matmul(spike_counts, self.output_weights)  # shape: (output_size,)
        return output

    def reset(self):
        self.hidden_layer.reset_state()


# ---------------------------------------------
# Data Loading (MNIST 28x28, no resize)
# ---------------------------------------------
transform = transforms.Compose([
    transforms.ToTensor(),  # Shape: 1x28x28 (grayscale)
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
# Using batch_size = 1 simplifies the state management in our spiking simulation.
trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

# ---------------------------------------------
# Device Setup: Use MPS for M2 Apple GPU if available.
# ---------------------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# ---------------------------------------------
# Initialize Model, Loss, Optimizer
# ---------------------------------------------
model = SNNModel(input_size=784, num_neurons=100, output_size=10, T=100).to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


# ---------------------------------------------
# Training Function
# ---------------------------------------------
def train_snn(model, dataloader, num_epochs=10):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for idx, (img, label) in enumerate(dataloader):
            # Flatten the image: original shape (1, 1, 28, 28) -> (784,)
            img = img.view(-1)
            # Convert to numpy on CPU (if necessary) for encoding.
            img_np = img.cpu().numpy()
            # Poisson encode the image: returns shape (T, 784)
            spikes_np = poisson_encode(img_np, T=model.T, max_rate=100)
            # Convert to tensor and move to device.
            input_spikes = torch.tensor(spikes_np, dtype=torch.float32).to(device)

            # Reset the model state for each new sample.
            model.reset()

            # Forward pass.
            logits = model(input_spikes)  # shape: (output_size,)
            loss = criterion(logits.unsqueeze(0), label.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if (idx + 1) % 500 == 0:
                print(
                    f"Epoch [{epoch + 1}/{num_epochs}], Sample [{idx + 1}/{len(dataloader)}], Loss: {total_loss / (idx + 1):.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}] Average Loss: {total_loss / len(dataloader):.4f}")


# ---------------------------------------------
# Evaluation Function
# ---------------------------------------------
def evaluate_snn(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for img, label in dataloader:
            img = img.view(-1)
            img_np = img.cpu().numpy()
            spikes_np = poisson_encode(img_np, T=model.T, max_rate=100)
            input_spikes = torch.tensor(spikes_np, dtype=torch.float32).to(device)
            model.reset()
            logits = model(input_spikes)
            predicted = torch.argmax(logits)
            total += 1
            correct += (predicted.item() == label.item())
            all_preds.append(predicted.item())
            all_labels.append(label.item())
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


# ---------------------------------------------
# Run Training and Evaluation
# ---------------------------------------------
train_snn(model, trainloader, num_epochs=10)
evaluate_snn(model, testloader)
