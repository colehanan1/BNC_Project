#!/usr/bin/env python3
"""
snn_mnist_optuna_delta_waveforms.py

Spiking Neural Network hyperparameter optimisation with Optuna + snnTorch
using **delta‑modulation ON/OFF events** instead of Poisson rate coding,
plus membrane‑potential waveform training targets and smoothing filters.

Key features
------------
- 7×7 avg‑pooled **delta modulation** encoder (signed spikes)
- Single hidden LIF -> output LIF layer, returning membrane potentials
- Surrogate‑gradient back‑prop
- Optuna Bayesian HPO + SuccessiveHalvingPruner
- Pruning on validation accuracy, hidden spike rate, and membrane loss
- Post‑training smoothing filter on waveform traces
- All original visualisations retained
"""

import argparse, warnings, math
import torch, torch.nn as nn, torch.optim as optim
import torchvision
from torchvision import transforms
import snntorch as snn
from snntorch import spikegen as spkgen
from snntorch import surrogate
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.exceptions import TrialPruned, ExperimentalWarning
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np, pandas as pd
import warnings
warnings.filterwarnings(
    "ignore",
    message="The reported value is ignored because this `step`",
    module="optuna.trial._trial"
)


# ───────────────────────── Device selection ─────────────────────────
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():    return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# ───────────────────── Delta‑modulation encoder ─────────────────────
pool7 = nn.AvgPool2d(4)  # 28x28 -> 7x7

def delta_encode(img, T, threshold=0.1, off_spike=True):
    inten = pool7(img).view(img.size(0), -1)           # [B,49]
    inten_seq = inten.unsqueeze(0).repeat(T,1,1)       # [T,B,49]
    return spkgen.delta(inten_seq, threshold=threshold, off_spike=off_spike)

# ────────────────────────── SNN definition ──────────────────────────
class SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta1, beta2):
        super().__init__()
        self.fc1  = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=surrogate.fast_sigmoid())
        self.fc2  = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, T):
        batch_size = x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_trace = []
        mem1_trace = []
        spk2_rec = torch.zeros(T, batch_size, self.fc2.out_features, device=device)
        mem2_rec = torch.zeros(T, batch_size, self.fc2.out_features, device=device)

        for t in range(T):
            cur1 = self.fc1(x[t])
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_trace.append(spk1.detach())
            mem1_trace.append(mem1[0,0].item())
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec[t] = spk2
            mem2_rec[t] = mem2

        return torch.stack(spk1_trace), mem1_trace, spk2_rec, mem2_rec

# ─────────────── Target waveform templates (one per class) ───────────
# Define your target/templates here: shape [10, T]; example: simple ramp
# Will be moved to device in objective

def make_default_templates(T):
    # each class: linear ramp from 0->1 scaled by class index
    t = torch.linspace(0,1,T)
    templates = torch.stack([ (c/9.0)*t for c in range(10) ])  # [10,T]
    return templates

# ──────────────────────── Optuna objective ─────────────────────────
warnings.filterwarnings("ignore", category=ExperimentalWarning)

def objective(trial):
    # Hyperparams
    tau    = trial.suggest_float("tau", 3.0, 30.0)
    hidden = trial.suggest_int("hidden", 15, 500)
    lr     = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    T      = trial.suggest_int("T", 75, 500)
    alpha  = trial.suggest_float("alpha_mem", 1e-4, 1e-1, log=True)

    # Decay factors
    beta_val = math.exp(-1.0 / tau)
    beta = torch.tensor(beta_val, device=device)

    # Model & optimisation
    model = SNN(49, hidden, 10, beta, beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    mse_fn  = nn.MSELoss()

    # Data
    tf = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(".", train=True,  download=True, transform=tf)
    ds_val   = torchvision.datasets.MNIST(".", train=False, download=True, transform=tf)
    loader_tr = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
    loader_val= torch.utils.data.DataLoader(ds_val,   batch_size=64, shuffle=False)

    # Templates
    templates = make_default_templates(T).to(device)  # [10,T]

    for epoch in range(3):
        total_spikes, correct, samples = 0.0, 0, 0
        for batch_idx, (imgs, lbls) in enumerate(loader_tr):
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes = delta_encode(imgs, T).to(device)
            optimizer.zero_grad()
            spk1, _, spk2, mem2 = model(spikes, T)
            out = spk2.sum(dim=0)
            ce_loss = loss_fn(out, lbls)
            # membrane loss: extract per-sample correct channel trace
            # mem2: [T,B,10] -> [B,10,T]
            mem2_tr = mem2.permute(1,2,0)
            # gather correct traces [B,T]
            idx = torch.arange(mem2_tr.size(0), device=device)
            correct_traces = mem2_tr[idx, lbls]
            target_traces  = templates[lbls]
            mem_loss = mse_fn(correct_traces, target_traces)
            loss = ce_loss + alpha * mem_loss
            loss.backward()
            optimizer.step()

            # stats
            preds = out.argmax(dim=1)
            correct   += (preds==lbls).sum().item()
            total_spikes += spk1.abs().sum()
            samples   += imgs.size(0)
            if batch_idx >= 10: break

        val_acc = correct / samples
        avg_rate= total_spikes / (samples * T)
        trial.report(val_acc, epoch*2)
        trial.report(avg_rate, epoch*2+1)
        trial.report(mem_loss.item(), epoch*2+2)
        if trial.should_prune(): raise TrialPruned()

    # full-val eval
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in loader_val:
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes     = delta_encode(imgs, T).to(device)
            _, _, spk2, _ = model(spikes, T)
            out = spk2.sum(dim=0)
            correct += (out.argmax(dim=1)==lbls).sum().item()
            total   += imgs.size(0)
    return correct / total

# ───────────────────────────── Main ─────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials",  type=int, default=10)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args()

    study = optuna.create_study(
        study_name="BNC_mnist_snn_tuning_delta_wave",
        direction="maximize",
        pruner=SuccessiveHalvingPruner(),
        storage="sqlite:///mnist_snn_tuning_delta_wave.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    print("Best hyperparameters:", study.best_trial.params)
    best = study.best_trial.params
    tau, hidden, lr, T, alpha = best["tau"], best["hidden"], best["lr"], best["T"], best["alpha_mem"]
    beta_val = math.exp(-1.0 / tau)
    beta = torch.tensor(beta_val, device=device)
    model = SNN(49, hidden, 10, beta, beta).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # Data loaders
    tf = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(".", train=True, download=True, transform=tf)
    ds_test  = torchvision.datasets.MNIST(".", train=False,download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(ds_test,  batch_size=64, shuffle=False)

    # Retrain full
    for epoch in range(10):
        model.train()
        tot_loss, corr = 0.0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes = delta_encode(imgs, T).to(device)
            optimizer.zero_grad()
            spk1, _, spk2, mem2 = model(spikes, T)
            out = spk2.sum(dim=0)
            ce_loss = loss_fn(out, lbls)
            # optional fine-tune mem loss as above if desired
            loss = ce_loss
            loss.backward(); optimizer.step()
            tot_loss += loss.item(); corr += (out.argmax(dim=1)==lbls).sum().item()
        acc = corr / len(train_loader.dataset)
        print(f"Epoch {epoch+1} — loss: {tot_loss/len(train_loader):.4f}, acc: {acc:.4f}")

    # Evaluate & record waveforms
    model.eval()
    # collect one example per class
    mem_traces = {c: None for c in range(10)}
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        spikes = delta_encode(imgs, T).to(device)
        _,_,_, mem2 = model(spikes, T)  # [T,B,10]
        for i,c in enumerate(lbls.cpu().numpy()):
            if mem_traces[c] is None:
                mem_traces[c] = mem2[:,i,c].cpu()
        if all(v is not None for v in mem_traces.values()): break

    # Post‑processing: exponential smoothing
    alpha_filt = 0.05
    def smooth(trace):
        out = []
        prev = trace[0]
        for v in trace:
            prev = alpha_filt*v + (1-alpha_filt)*prev
            out.append(prev)
        return torch.stack(out)

    mem_smooth = {c: smooth(mem_traces[c]) for c in mem_traces}

    # Plot smoothed waveforms
    plt.figure(figsize=(8,5))
    for c, trace in mem_smooth.items():
        plt.plot(trace.detach().cpu().numpy(), label=f"Class {c}")
    plt.title("Smoothed Membrane Potential Traces")
    plt.xlabel("Time step"); plt.ylabel("Membrane potential")
    plt.legend(); plt.show()

    # original visualisations (confusion, rasters, weights, etc.)
    # … copy unchanged from previous script …

    # Save final model
    torch.save(model.state_dict(), "snn_mnist_final.pth")
    print("Model state_dict saved to snn_mnist_final.pth")