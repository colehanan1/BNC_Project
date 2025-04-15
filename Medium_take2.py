import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define the LIF Neuron Model
class LIFNeuronModel(nn.Module):
    def __init__(self, num_neurons, tau_m=20.0, threshold=-50.0, v_rest=-65.0):
        super(LIFNeuronModel, self).__init__()
        self.num_neurons = num_neurons
        self.tau_m = tau_m
        self.threshold = threshold
        self.v_rest = v_rest
        self.v = torch.full((num_neurons,), v_rest)
        self.membrane_potential = self.v.clone()

    def forward(self, input_spikes, dt=1.0):
        self.membrane_potential += (self.v_rest - self.membrane_potential) / self.tau_m + input_spikes * dt
        spikes = (self.membrane_potential >= self.threshold).float()
        self.membrane_potential[spikes == 1] = self.v_rest
        return spikes

# Define the SNN Model
class SNNModel(nn.Module):
    def __init__(self, input_size=784, num_neurons=100, output_size=10):
        super(SNNModel, self).__init__()
        self.input_size = input_size
        self.num_neurons = num_neurons
        self.output_size = output_size
        self.hidden_layer = LIFNeuronModel(num_neurons)
        self.output_weights = nn.Parameter(torch.rand(num_neurons, output_size))

    def forward(self, input_spikes):
        hidden_spikes = self.hidden_layer(input_spikes)
        output_spikes = torch.matmul(hidden_spikes, self.output_weights)
        return output_spikes

    def reset(self):
        self.hidden_layer.membrane_potential.fill_(self.hidden_layer.v_rest)

# Data loading and encoding
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Initialize the model
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = SNNModel(input_size=784, num_neurons=100, output_size=10).to(device)

# Optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training the SNN
def train_snn(model, trainloader, num_epochs=10):
    for epoch in range(num_epochs):
        total_loss = 0
        for inputs, labels in trainloader:
            inputs = inputs.view(-1, 784).to(device)
            input_spikes = inputs  # Using rate coding directly

            model.reset()
            output_spikes = model(input_spikes)

            loss = nn.CrossEntropyLoss()(output_spikes, labels.to(device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss / len(trainloader)}")

# Evaluate the SNN on the test set
def evaluate_snn(model, testloader):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs = inputs.view(-1, 784).to(device)
            input_spikes = inputs  # Using rate coding directly

            model.reset()
            output_spikes = model(input_spikes)

            _, predicted = torch.max(output_spikes, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=trainset.classes, yticklabels=trainset.classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()

# Train and evaluate the model
train_snn(model, trainloader, num_epochs=10)
evaluate_snn(model, testloader)
