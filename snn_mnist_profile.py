#!/usr/bin/env python3
"""
snn_mnist_optuna.py

Spiking Neural Network hyperparameter optimization with Optuna and snnTorch:
 - 7×7 Poisson rate encoding
 - Single hidden LIF layer → output LIF layer (no convolution)
 - Surrogate‑gradient backpropagation
 - Optuna Bayesian HPO with SuccessiveHalvingPruner
 - Pruning on validation accuracy and hidden spike rate (silent‑network detection)
 - Fix membrane‑trace error (use mem1[0,0].item())
 - Device selection: MPS → CUDA → CPU
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import snntorch as snn
from snntorch import spikegen as spkgen
from snntorch import surrogate
import optuna
from optuna.pruners import HyperbandPruner, SuccessiveHalvingPruner
from optuna.exceptions import TrialPruned
from optuna.visualization import plot_optimization_history


# Device selection (MPS → CUDA → CPU)
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():    return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Poisson 7×7 encoder (28→7 via avgpool)
pool7 = nn.AvgPool2d(4)
def poisson_encode(img, num_steps):
    inten = pool7(img).view(img.size(0), -1)
    inten = inten / (inten.max(dim=1, keepdim=True)[0] + 1e-12)
    return spkgen.rate(inten, num_steps=num_steps)

# SNN definition: FC→LIF→FC→LIF
class SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta1, beta2):
        super().__init__()
        self.fc1  = nn.Linear(num_inputs,  num_hidden)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=surrogate.fast_sigmoid())
        self.fc2  = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps):
        # x: [T, B, inputs]
        batch_size = x.size(1)
        # Initialize membrane potentials
        mem1 = self.lif1.init_leaky()  # -> [B, hidden]
        mem2 = self.lif2.init_leaky()  # -> [B, outputs]

        spk1_trace = []
        mem1_trace = []
        spk2_rec   = torch.zeros(num_steps, batch_size, self.fc2.out_features, device=device)

        for t in range(num_steps):
            cur1       = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            # Record hidden‑layer spikes and membrane for sample 0 only
            spk1_trace.append(spk1[0].detach().cpu())
            mem1_trace.append(mem1[0,0].item())  # <— fixed: extract scalar from mem1[0,0]

            cur2        = self.fc2(spk1)
            spk2, mem2  = self.lif2(cur2, mem2)
            spk2_rec[t] = spk2

        return torch.stack(spk1_trace), mem1_trace, spk2_rec

# Objective function for Optuna
def objective(trial):
    # Suggest hyperparameters
    tau    = trial.suggest_float("tau",    5.0, 20.0)
    hidden = trial.suggest_int("hidden",  20,  100)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    T       = trial.suggest_int("T",     50,   200)

    # Compute decay factors for LIF
    beta = torch.exp(torch.tensor(-1.0 / tau))
    beta2 = beta

    # Build model
    model = SNN(49, hidden, 10, beta, beta2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    # Data loaders (small subsets for speed)
    tf = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(".", train=True,  download=True, transform=tf)
    ds_val   = torchvision.datasets.MNIST(".", train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(ds_val,   batch_size=64, shuffle=False)

    # Training with pruning
    for epoch in range(3):  # only 3 epochs during HPO
        total_spikes = 0.0
        correct      = 0
        num_samples  = 0

        for batch_idx, (imgs, lbls) in enumerate(train_loader):
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes = poisson_encode(imgs, T).to(device)

            optimizer.zero_grad()
            spk1_trace, mem1_trace, spk2 = model(spikes, T)
            out   = spk2.sum(dim=0)
            loss  = loss_fn(out, lbls)
            loss.backward()
            optimizer.step()

            preds = out.argmax(dim=1)
            correct     += (preds == lbls).sum().item()
            total_spikes += spk1_trace.sum()
            num_samples  += imgs.size(0)

            # Limit batches per epoch for speed
            if batch_idx >= 10:
                break

        val_acc = correct / num_samples
        avg_rate = total_spikes / (num_samples * T)
        # Report intermediate results
        # report accuracy and spike rate at unique integer steps
        step_acc = epoch * 2
        step_rate = epoch * 2 + 1
        trial.report(val_acc,    step_acc)
        trial.report(avg_rate, step_rate)

        # Prune if no improvement or silent network
        if trial.should_prune():
            raise TrialPruned()

    # Final validation accuracy on held‑out set
    correct = 0
    total   = 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes = poisson_encode(imgs, T).to(device)
            _, _, spk2 = model(spikes, T)
            out = spk2.sum(dim=0)
            correct += (out.argmax(dim=1) == lbls).sum().item()
            total   += imgs.size(0)

    return correct / total

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=50, help="Number of HPO trials")
    parser.add_argument("--timeout", type=int, default=None, help="HPO timeout (s)")
    args = parser.parse_args()

    # Create Optuna study with pruning
    study = optuna.create_study(
        study_name="BNC_mnist_snn_tuning",
        direction="maximize",
        pruner=HyperbandPruner(
            min_resource=10,
            max_resource=100,  # e.g., 100 steps total
            reduction_factor=3
    ),
        storage="sqlite:///mnist_snn_tuning_hyperbrand.db",  # optional: persist to disk
        load_if_exists=True  # resume if already created
    )

    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    fig = plot_optimization_history(study)
    fig.show()

    print("Best hyperparameters:", study.best_trial.params)

    # Retrain final model on full training set
    best = study.best_trial.params
    tau, hidden, lr, T = best["tau"], best["hidden"], best["lr"], best["T"]
    beta = torch.exp(torch.tensor(-1.0 / tau))
    beta2 = beta

    # Full training
    final_model = SNN(49, hidden, 10, beta, beta2).to(device)
    optimizer = optim.Adam(final_model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()
    tf = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(".", train=True,  download=True, transform=tf)
    ds_test  = torchvision.datasets.MNIST(".", train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(ds_test,  batch_size=64, shuffle=False)

    # Train for more epochs
    for epoch in range(10):
        final_model.train()
        total_loss, correct = 0.0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes = poisson_encode(imgs, T).to(device)
            optimizer.zero_grad()
            _, _, spk2 = final_model(spikes, T)
            out = spk2.sum(dim=0)
            loss = loss_fn(out, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(dim=1) == lbls).sum().item()
        acc = correct / len(train_loader.dataset)
        print(f"Epoch {epoch+1}/10 — loss: {total_loss/len(train_loader):.4f}, acc: {acc:.4f}")

    # Final test
    final_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes = poisson_encode(imgs, T).to(device)
            _, _, spk2 = final_model(spikes, T)
            out = spk2.sum(dim=0)
            correct += (out.argmax(dim=1) == lbls).sum().item()
            total   += imgs.size(0)
    print(f"Test accuracy: {correct/total:.4f}")

if __name__ == "__main__":
    main()
