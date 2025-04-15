import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import random


# Define the LIF Neuron Model for SNN
class LIFNeuronModel(nn.Module):
    def __init__(self, num_neurons, tau_m=20.0, threshold=-50.0, v_rest=-65.0):
        super(LIFNeuronModel, self).__init__()
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.threshold = threshold
        self.v_rest = v_rest

        self.v = torch.full((num_neurons,), v_rest)
        self.refractory_period = torch.zeros((num_neurons,))
        self.membrane_potential = self.v.clone()

    def forward(self, input_spikes, dt=1.0):
        # Update membrane potential using LIF equations
        self.membrane_potential = self.membrane_potential + (
                    self.v_rest - self.membrane_potential) / self.tau_m + input_spikes * dt
        spikes = (self.membrane_potential >= self.threshold).float()  # Generate spikes
        self.membrane_potential[spikes == 1] = self.v_rest  # Reset membrane potential after spike
        return spikes


# Poisson Encoding of MNIST Images (rate coding)
def poisson_encode(image, T=100, max_rate=100):
    pixels = image.flatten()
    spikes = np.random.rand(T, len(pixels)) < pixels * max_rate / 255
    return spikes.astype(int)


# Define STDP for weight updates
def stdp_update(pre_spikes, post_spikes, weights, tau_plus=20.0, tau_minus=20.0, A_plus=0.005, A_minus=0.005):
    for i in range(len(pre_spikes)):
        for j in range(len(post_spikes)):
            delta_t = pre_spikes[i] - post_spikes[j]  # time difference between spikes
            if delta_t > 0:  # pre spike before post spike (LTP)
                weights[i, j] += A_plus * np.exp(-delta_t / tau_plus)
            elif delta_t < 0:  # post spike before pre spike (LTD)
                weights[i, j] -= A_minus * np.exp(delta_t / tau_minus)
    return np.clip(weights, 0, 1)  # Clamp weights to stay in range [0,1]


# Simple SNN Model using LIF Neurons and STDP
class SNNModel(nn.Module):
    def __init__(self, input_size=784, num_neurons=100, output_size=10):
        super(SNNModel, self).__init__()
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.output_size = output_size

        # Create LIF neurons for the hidden layer
        self.hidden_layer = LIFNeuronModel(num_neurons)

        # Output layer with learnable parameters (weights)
        self.output_weights = nn.Parameter(torch.rand(num_neurons, output_size))  # This is now a learnable parameter

    def forward(self, input_spikes):
        # Process input spikes through the hidden layer
        hidden_spikes = self.hidden_layer(input_spikes)

        # Compute the output spikes based on the weighted sum
        output_spikes = torch.matmul(hidden_spikes, self.output_weights)
        return output_spikes

    def reset(self):
        self.hidden_layer.membrane_potential.fill_(self.hidden_layer.v_rest)  # Reset hidden layer membrane potential


# Data loading and encoding
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# Load MNIST dataset
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Check if MPS is available (for Apple GPUs with MPS backend)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize the model
model = SNNModel(input_size=784, num_neurons=100, output_size=10).to(device)  # Move model to MPS

# Optimizer (simple for learning weights)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


# Training the SNN
def train_snn(model, trainloader, num_epochs=10):
    for epoch in range(num_epochs):
        total_loss = 0
        for i, data in enumerate(trainloader):
            inputs, labels = data
            inputs = inputs.view(-1, 784).to(device)  # Flatten image to 1D vector and move to MPS device

            # Poisson encoding for input spikes
            input_spikes = torch.tensor(poisson_encode(inputs.cpu().numpy(), T=100), dtype=torch.float32).to(device)

            # Reset the hidden layer's membrane potential at the start of each batch
            model.reset()

            # Forward pass: get output spikes
            output_spikes = model(input_spikes)

            # Calculate the loss (based on spike activity)
            loss = nn.CrossEntropyLoss()(output_spikes, labels.to(device))

            # Backprop and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(trainloader)}")


# Evaluate the SNN on the test set
def evaluate_snn(model, testloader):
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in testloader:
            inputs, labels = data
            inputs = inputs.view(-1, 784).to(device)  # Flatten image and move to MPS device

            # Poisson encoding for input spikes
            input_spikes = torch.tensor(poisson_encode(inputs.numpy(), T=100), dtype=torch.float32).to(device)

            # Reset the hidden layer's membrane potential
            model.reset()

            # Forward pass: get output spikes
            output_spikes = model(input_spikes)

            # Get predicted class based on maximum spike activity
            _, predicted = torch.max(output_spikes, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

            # Save predictions and labels for confusion matrix
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * correct / total
    print(f"SNN Accuracy on test set: {accuracy:.2f}%")

    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=testset.classes, yticklabels=testset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()


# Train the model
train_snn(model, trainloader, num_epochs=10)

# Evaluate the model
evaluate_snn(model, testloader)
