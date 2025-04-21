#!/usr/bin/env python3
"""
snn_mnist_optuna_with_plots.py

Spiking Neural Network hyperparameter optimization with Optuna and snnTorch,
extended with visualizations:
 - 7×7 Poisson rate encoding
 - Single hidden LIF layer → output LIF layer (no convolution)
 - Surrogate‐gradient backpropagation
 - Optuna Bayesian HPO with SuccessiveHalvingPruner
 - Pruning on validation accuracy and hidden spike rate
 - Fix membrane‐trace error (use mem1[0,0].item())
 - Device selection: MPS → CUDA → CPU
 - Post‐training plots: confusion matrix, Optuna plots, epoch vs accuracy,
   spike rasters per class, weight histograms, spike activity before/after,
   per‐class accuracy table
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
from optuna.pruners import SuccessiveHalvingPruner
from optuna.exceptions import TrialPruned

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
from optuna.exceptions import ExperimentalWarning
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
# Suppress Optuna experimental warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)
import numpy as np
import pandas as pd

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
        batch_size = x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_trace = []
        mem1_trace = []
        spk2_rec   = torch.zeros(num_steps, batch_size, self.fc2.out_features, device=device)

        for t in range(num_steps):
            cur1       = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_trace.append(spk1[0].detach().cpu())
            mem1_trace.append(mem1[0,0].item())
            cur2        = self.fc2(spk1)
            spk2, mem2  = self.lif2(cur2, mem2)
            spk2_rec[t] = spk2

        return torch.stack(spk1_trace), mem1_trace, spk2_rec

# Objective for Optuna HPO
def objective(trial):
    tau    = trial.suggest_float("tau",    5.0, 20.0)
    hidden = trial.suggest_int("hidden",  20,  100)
    lr     = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    T       = trial.suggest_int("T",     50,   200)

    beta  = torch.exp(torch.tensor(-1.0 / tau))
    beta2 = beta
    model = SNN(49, hidden, 10, beta, beta2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    tf = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(".", train=True,  download=True, transform=tf)
    ds_val   = torchvision.datasets.MNIST(".", train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
    val_loader   = torch.utils.data.DataLoader(ds_val,   batch_size=64, shuffle=False)

    for epoch in range(3):
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
            if batch_idx >= 10: break

        val_acc  = correct / num_samples
        avg_rate = total_spikes / (num_samples * T)
        trial.report(val_acc,  epoch*2)
        trial.report(avg_rate, epoch*2+1)
        if trial.should_prune(): raise TrialPruned()

    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in val_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes      = poisson_encode(imgs, T).to(device)
            _, _, spk2  = model(spikes, T)
            out = spk2.sum(dim=0)
            correct += (out.argmax(dim=1) == lbls).sum().item()
            total   += imgs.size(0)
    return correct / total

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=100)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()

    study = optuna.create_study(
        study_name="BNC_mnist_snn_tuning",
        direction="maximize",
        pruner=SuccessiveHalvingPruner(),
        storage="sqlite:///mnist_snn_tuning.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    print("Best hyperparameters:", study.best_trial.params)

    best    = study.best_trial.params
    tau, hidden, lr, T = best["tau"], best["hidden"], best["lr"], best["T"]
    beta    = torch.exp(torch.tensor(-1.0 / tau))
    beta2   = beta
    model   = SNN(49, hidden, 10, beta, beta2).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn   = nn.CrossEntropyLoss()

    tf = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(".", train=True,  download=True, transform=tf)
    ds_test  = torchvision.datasets.MNIST(".", train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(ds_test,  batch_size=64, shuffle=False)

    # Training history
    epoch_accs = []
    weight_histories = []
    for epoch in range(10):
        model.train()
        total_loss, correct = 0.0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes = poisson_encode(imgs, T).to(device)
            optimizer.zero_grad()
            _, _, spk2 = model(spikes, T)
            out  = spk2.sum(dim=0)
            loss = loss_fn(out, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (out.argmax(dim=1) == lbls).sum().item()
        acc = correct / len(train_loader.dataset)
        epoch_accs.append(acc)
        weight_histories.append(model.fc1.weight.detach().cpu().numpy().ravel())
        print(f"Epoch {epoch+1}/10 — loss: {total_loss/len(train_loader):.4f}, acc: {acc:.4f}")

    # Final test & collect preds for confusion and per-class
    model.eval()
    y_true, y_pred = [], []
    # pre-training spike counts
    counts_before = np.zeros(model.fc1.out_features)
    # post-training spike counts
    counts_after  = np.zeros(model.fc1.out_features)
    fixed_loader = torch.utils.data.DataLoader(ds_test, batch_size=64, shuffle=False)

    # Compute before learning counts
    untrained = SNN(49, hidden, 10, beta, beta2).to(device)
    idx = 0
    for imgs, lbls in fixed_loader:
        spikes = poisson_encode(imgs, T).to(device)
        spk1, _, _ = untrained(spikes, T)
        counts_before += spk1.sum(axis=0).numpy()  # fix sum over time axis to match hidden-size
        idx += imgs.size(0)
        if idx>1000: break
    counts_before /= idx

    correct, total = 0, 0
    idx = 0
    for imgs, lbls in fixed_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        spikes     = poisson_encode(imgs, T).to(device)
        spk1, _, spk2 = model(spikes, T)
        counts_after  += spk1.sum(axis=0).cpu().numpy()  # fix sum over time axis to match hidden-size
        out = spk2.sum(dim=0)
        y_true.extend(lbls.cpu().numpy())
        y_pred.extend(out.argmax(dim=1).cpu().numpy())
        idx += imgs.size(0)
        if idx>1000: break
    counts_after /= idx

    # 1. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
    ax.set_title("Confusion Matrix")
    plt.show()

    # 2. Epoch vs Accuracy
    plt.figure()
    plt.plot(range(1, len(epoch_accs)+1), epoch_accs, marker='o')  # use dynamic length
    plt.xlabel("Epoch"); plt.ylabel("Accuracy")
    plt.title("Epoch vs Training Accuracy")
    plt.grid(True); plt.show()

                    # 3. Optuna plots
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
        plot_terminator_improvement,
    )
    # Define plotting functions
    plot_funcs = [
        plot_optimization_history,
        lambda st: plot_parallel_coordinate(st, ['tau','hidden','lr','T']),
        plot_param_importances,
        plot_slice,
        plot_terminator_improvement,
    ]
    # Generate and display each Optuna plot
    for fn in plot_funcs:
        ax = fn(study)
        # Handle single Axes or array of Axes
        if isinstance(ax, (list, tuple, np.ndarray)):
            for subax in np.array(ax).ravel():
                subax.figure.show()
        else:
            ax.figure.show()

        # 4. Spike raster per digit
    spike_trains = {i: [] for i in range(10)}
    for imgs, lbls in test_loader:
        spikes = poisson_encode(imgs, T).to(device)
        spk1, _, _ = model(spikes, T)
        for j, lbl in enumerate(lbls.cpu().numpy()):
            if len(spike_trains[lbl]) == 0:
                spike_trains[lbl] = spk1[:, j].cpu().numpy()
        # Exit loop once we have spike trains for all 10 digits
        if all(len(v) > 0 for v in spike_trains.values()):
            break
    # Plot rasters
    fig, axs = plt.subplots(5, 2, figsize=(10, 12), sharex=True)
    for d, ax in enumerate(axs.flatten()):
        ax.eventplot(np.where(spike_trains[d] == 1)[0], orientation='horizontal')
        ax.set_title(f"Digit {d}")
    plt.tight_layout()
    plt.show()

    # 5. Weight hist evolution Weight hist evolution
    plt.figure(figsize=(8,6))
    for i,w in enumerate(weight_histories):
        hist,bins = np.histogram(w, bins=50, range=(-0.5,0.5), density=True)
        plt.plot(bins[:-1], hist, label=f"Ep {i+1}")
    plt.legend(); plt.title("FC1 Weight Distribution"); plt.show()

    # 6. Spike activity before vs after
    indices = np.arange(len(counts_before))
    width=0.4
    plt.figure(figsize=(10,4))
    plt.bar(indices-width/2, counts_before, width, label='Before')
    plt.bar(indices+width/2, counts_after,  width, label='After')
    plt.xlabel("Neuron"); plt.ylabel("Avg spike count")
    plt.legend(); plt.title("Spike Activity Before vs After"); plt.show()

    # 7. Accuracy table
    accs={}
    for d in range(10):
        idxs=[i for i,y in enumerate(y_true) if y==d]
        accs[d]= sum(y_pred[i]==y_true[i] for i in idxs)/len(idxs)
    overall = sum(y_pred[i]==y_true[i] for i in range(len(y_true)))/len(y_true)
    df = pd.DataFrame({"Digit":list(accs.keys())+['Overall'],
                       "Accuracy":list(accs.values())+[overall]})
    print(df.to_markdown(index=False))

            # Save the trained model's state_dict
    save_path = "snn_mnist_final.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model state_dict saved to {save_path}")
    print(f"Final test accuracy: {correct/total:.4f}")


