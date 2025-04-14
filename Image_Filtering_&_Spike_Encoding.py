import os
os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import norse.torch as norse
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("MPS backend is available. Using Apple GPU.")
else:
    device = torch.device("cpu")
    print("MPS backend not available. Using CPU.")

# Hyperparameters
num_epochs = 10
batch_size = 64
learning_rate = 1e-3

# CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),  # Convert to grayscale
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data',
                                             train=True,
                                             transform=transform,
                                             download=True)

test_dataset = torchvision.datasets.CIFAR10(root='./data',
                                            train=False,
                                            transform=transform,
                                            download=True)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)

# Define the spiking neural network
class SNNModel(nn.Module):
    def __init__(self):
        super(SNNModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.lif1 = norse.LIFCell()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.lif2 = norse.LIFCell()
        self.pool = nn.AvgPool2d(2)
        self.fc1 = nn.Linear(64 * 16 * 16, 100)
        self.lif3 = norse.LIFCell()
        self.fc2 = nn.Linear(100, 10)
        self.lif4 = norse.LIFCell()

    def forward(self, x):
        seq_length = 100  # Simulation time steps
        batch_size = x.size(0)

        # Initialize states using a dummy input tensor
        dummy_input1 = torch.zeros(batch_size, 32, 32, 32, device=x.device)
        dummy_input2 = torch.zeros(batch_size, 64, 32, 32, device=x.device)
        dummy_input3 = torch.zeros(batch_size, 100, device=x.device)
        dummy_input4 = torch.zeros(batch_size, 10, device=x.device)

        mem1 = self.lif1.initial_state(dummy_input1)
        mem2 = self.lif2.initial_state(dummy_input2)
        mem3 = self.lif3.initial_state(dummy_input3)
        mem4 = self.lif4.initial_state(dummy_input4)

        spk_out = torch.zeros(batch_size, 10, device=x.device)

        for t in range(seq_length):
            z = self.conv1(x)
            z, mem1 = self.lif1(z, mem1)
            z = self.conv2(z)
            z, mem2 = self.lif2(z, mem2)
            z = self.pool(z)
            z = z.view(batch_size, -1)
            z = self.fc1(z)
            z, mem3 = self.lif3(z, mem3)
            z = self.fc2(z)
            z, mem4 = self.lif4(z, mem4)
            spk_out += z

        return spk_out / seq_length

# Initialize the model and move it to the appropriate device
model = SNNModel().to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(num_epochs):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")

# Evaluation
model.eval()
all_preds = []
all_labels = []
with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            yticklabels=train_dataset.classes)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()