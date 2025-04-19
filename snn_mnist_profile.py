#!/usr/bin/env python3
"""
snn_mnist_profile.py

Fully connected SNN for MNIST with snnTorch:
- 7×7 Poisson rate encoding
- Single hidden LIF layer → output LIF layer
- Surrogate-gradient backpropagation
- Profiling epoch runtimes
- Input/spike/membrane visualizations
- Dynamic tuning via CLI (tau, hidden size, lr, T, epochs)
- Device: mps (Apple MPS), cuda (NVIDIA), or cpu
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import matplotlib.pyplot as plt
import snntorch as snn
from snntorch import spikegen as spkgen
from snntorch import surrogate

# Device selection (MPS → CUDA → CPU) :contentReference[oaicite:4]{index=4}
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():    return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Poisson 7×7 encoder
pool7 = nn.AvgPool2d(4)  # 28→7 :contentReference[oaicite:5]{index=5}
def poisson_encode(img, num_steps):
    # img: [B,1,28,28] → intensities [B,49]
    inten = pool7(img).view(img.size(0), -1)
    inten = inten / inten.max(dim=1, keepdim=True)[0]  # normalize to [0,1]
    # generate Poisson spike trains [T, B, 49]
    return spkgen.rate(inten, num_steps=num_steps)

# SNN model: FC→LIF→FC→LIF :contentReference[oaicite:6]{index=6}
class SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta1, beta2):
        super().__init__()
        self.fc1  = nn.Linear(num_inputs,  num_hidden)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=surrogate.fast_sigmoid())
        self.fc2  = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps):
        # x has shape [T, B, input_size]
        B = x.size(1)
        # Initialize states
        mem1 = self.lif1.init_leaky()  # returns [B, hidden_size]
        mem2 = self.lif2.init_leaky()  # returns [B, output_size]        # Traces for visualization
        mem1_trace, spk1_trace = [], []

        spk2_rec = torch.zeros(num_steps, B, self.fc2.out_features, device=device)
        for t in range(num_steps):
            cur1    = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)  # works with mem1 from init_leaky()
            spk1_trace.append(spk1[:,0].detach().cpu().numpy())  # first sample
            mem1_trace.append(mem1[0].item())

            cur2     = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec[t] = spk2

        return torch.stack(spk1_trace), torch.stack(mem1_trace), spk2_rec

# Visualization utilities :contentReference[oaicite:7]{index=7}
def plot_intensity_hist(images, fname="intensity_hist.png"):
    inten = pool7(images).detach().cpu().numpy().flatten()
    plt.figure()
    plt.hist(inten, bins=30)
    plt.title("Input Intensity Histogram")
    plt.savefig(fname); plt.close()

def plot_spike_raster(spk_trace, fname="spike_raster.png"):
    plt.figure(figsize=(6,4))
    plt.imshow(spk_trace.T, aspect='auto', interpolation='nearest')
    plt.xlabel("Time step"); plt.ylabel("Neuron")
    plt.title("Hidden Layer Spike Raster (sample 0)")
    plt.savefig(fname); plt.close()

def plot_membrane(mem_trace, fname="membrane_trace.png"):
    plt.figure()
    plt.plot(mem_trace)
    plt.xlabel("Time step"); plt.ylabel("Membrane Potential")
    plt.title("Hidden Neuron 0 Membrane Potential")
    plt.savefig(fname); plt.close()

# Training loop with profiling & rate logging
def train(model, train_loader, loss_fn, optimizer, args):
    model.train()
    for epoch in range(args.epochs):
        t_start = time.time()
        total_loss, total_correct, total_spikes = 0.0, 0, 0
        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes = poisson_encode(imgs, args.T).to(device)

            optimizer.zero_grad()
            spk_trace, mem_trace, spk2 = model(spikes, args.T)
            out = spk2.sum(dim=0)  # sum over time
            loss = loss_fn(out, lbls)
            loss.backward(); optimizer.step()

            preds = out.argmax(dim=1)
            total_correct += (preds == lbls).sum().item()
            total_loss    += loss.item()
            total_spikes  += spk_trace.sum()

            # First‑batch visualizations
            if epoch==0 and batch_idx==0:
                plot_intensity_hist(imgs)
                plot_spike_raster(spk_trace.numpy())
                plot_membrane(mem_trace)

        epoch_time = time.time() - t_start
        avg_loss = total_loss / len(train_loader)
        acc = total_correct / len(train_loader.dataset)
        avg_rate = total_spikes.item() / (len(train_loader.dataset) * args.T)
        print(f"Epoch {epoch+1}/{args.epochs} — time: {epoch_time:.2f}s, loss: {avg_loss:.4f}, "
              f"acc: {acc:.4f}, rate: {avg_rate:.4f}")

# Evaluate accuracy
def evaluate_model(model, test_loader, args):
    model.eval()
    correct = 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes = poisson_encode(imgs, args.T).to(device)
            _, _, spk2 = model(spikes, args.T)
            out = spk2.sum(dim=0)
            correct += (out.argmax(dim=1) == lbls).sum().item()
    acc = correct / len(test_loader.dataset)
    print(f"Test Accuracy: {acc:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tau",    type=float, default=10.0, help="membrane time constant")  # ms
    parser.add_argument("--hidden", type=int,   default=50,   help="hidden neuron count")
    parser.add_argument("--lr",     type=float, default=1e-3, help="learning rate")
    parser.add_argument("--T",      type=int,   default=100,  help="time steps (ms)")
    parser.add_argument("--epochs", type=int,   default=5,    help="training epochs")
    args = parser.parse_args()

    # Compute decay factor beta = exp(-dt/τ) with dt=1 ms :contentReference[oaicite:8]{index=8}
    beta = torch.exp(torch.tensor(-1.0 / args.tau))
    # Same for output layer
    beta2 = beta

    # Data loaders
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(".", train=True,  download=True, transform=tf)
    test_ds  = torchvision.datasets.MNIST(".", train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=64, shuffle=False)

    # Model, loss, optimizer
    model = SNN(49, args.hidden, 10, beta, beta2).to(device)
    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train & Test
    train(model, train_loader, loss_fn, optimizer, args)
    evaluate_model(model, test_loader, args)

if __name__ == "__main__":
    main()
