#!/usr/bin/env python3
"""
snn_mnist_optuna_poission.py

Spiking Neural Network hyperparameter optimisation with Optuna + snnTorch
using **poission ON/OFF events** instead of Poisson rate coding,
plus membrane‑potential waveform training targets and smoothing filters.

Key features
------------
- 7×7 avg‑pooled **poission** encoder (signed spikes)
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
import torch.nn.functional as F
import warnings
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA

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

def poisson_encode(img, num_steps):
    inten = pool7(img).view(img.size(0), -1)
    inten = inten / (inten.max(dim=1, keepdim=True)[0] + 1e-12)
    return spkgen.rate(inten, num_steps=num_steps)

def alpha_kernel(L=100, tau_r=1.0, tau_f=5.0, dt=1.0, device='cpu'):
    t = torch.arange(0, L * dt, dt, device=device)
    return (t / tau_r) * torch.exp(1 - t / tau_r) * torch.exp(-t / tau_f)

    # ────────────────────────── SNN definition ──────────────────────────
class SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta1, beta2):
        super().__init__()
        self.fc1  = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(
            beta=beta1,
            spike_grad=surrogate.fast_sigmoid())
        self.fc2  = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2,
                              spike_grad=surrogate.fast_sigmoid())
        self.bias1 = nn.Parameter(torch.tensor(0.2))

    def forward(self, x, T):
        batch_size = x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_trace = []
        mem1_trace = []
        spk2_rec = torch.zeros(T, batch_size, self.fc2.out_features, device=device)
        mem2_rec = torch.zeros(T, batch_size, self.fc2.out_features, device=device)

        for t in range(T):
            # add bias to keep membrane charging each step
            cur1 = self.fc1(x[t]) + 0.2  # ← tweak 0.2 up/down
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
            spikes = poisson_encode(imgs, T).to(device)
            optimizer.zero_grad()
            spk1, mem1_tr, spk2_rec, mem2_rec = model(spikes, T)
            out = spk2_rec.sum(dim=0)
            ce_loss = loss_fn(out, lbls)
            # membrane loss: extract per-sample correct channel trace
            # mem2: [T,B,10] -> [B,10,T]
            mem2_tr = mem2_rec.permute(1,2,0)
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
            spikes     = poisson_encode(imgs, T).to(device)
            spk1, mem1_tr, spk2_rec, mem2_rec = model(spikes, T)
            out = spk2_rec.sum(dim=0)
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
        study_name="BNC_mnist_snn_tuning_poission_wave",
        direction="maximize",
        pruner=SuccessiveHalvingPruner(),
        storage="sqlite:///mnist_snn_tuning_poission_wave.db",
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
    epoch_accs = []
    weight_histories = []
    for epoch in range(15):
        model.train()
        tot_loss, corr = 0.0, 0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            spikes = poisson_encode(imgs, T).to(device)
            optimizer.zero_grad()
            spk1, mem1_tr, spk2_rec, mem2_rec = model(spikes, T)
            out = spk2_rec.sum(dim=0)
            ce_loss = loss_fn(out, lbls)
            # optional fine-tune mem loss as above if desired
            loss = ce_loss
            loss.backward(); optimizer.step()
            tot_loss += loss.item(); corr += (out.argmax(dim=1)==lbls).sum().item()
        acc = corr / len(train_loader.dataset)
        epoch_accs.append(acc)
        weight_histories.append(model.fc1.weight.detach().cpu().numpy().ravel())
        print(f"Epoch {epoch+1} — loss: {tot_loss/len(train_loader):.4f}, acc: {acc:.4f}")

    # Evaluate & record waveforms
    model.eval()
    # collect one example per class
    mem_traces = {c: None for c in range(10)}
    spk_traces = {c: None for c in range(10)}
    for imgs, lbls in test_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        spikes = poisson_encode(imgs, T).to(device)
        spk1, mem1_tr, spk2_rec, mem2_rec = model(spikes, T)
        for i, c in enumerate(lbls.cpu().numpy()):
            if mem_traces[c] is None:
                mem_traces[c] = mem2_rec[:, i, c].cpu()
                spk_traces[c] = spk2_rec[:, i, c].cpu().float()
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

    # ─── α‑kernel convolution to get AP‑shaped spikes ───
    kernel = alpha_kernel(L=50, tau_r=1.0, tau_f=5.0,
                          dt=1.0, device=device).view(1, 1, -1)
    ap_traces = {}
    for c, spk in spk_traces.items():
        spk = spk.view(1, 1, -1).to(device)   # move to MPS (or CUDA) same as kernel
        ap = F.conv1d(spk, kernel, padding=kernel.size(-1) // 2)
        ap_traces[c] = ap.view(-1).detach().cpu().numpy()

    # plot AP‑shaped waveforms
    fig, axes = plt.subplots(10, 1, figsize=(6, 20), sharex=True, sharey=True)
    for c, trace in ap_traces.items():
        ax = axes[c]  # index the 1‑D axes array directly
        ax.plot(trace)
        ax.set_title(f"Class {c}")
        ax.set_ylim(0, 2)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Filtered spike")
        ax.grid(True)

    fig.suptitle("α‑Kernel Filtered Spike Waveforms", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.show()

    # 1. Compute before‑learning counts
    untrained = SNN(49, hidden, 10, beta, beta).to(device)
    counts_before = np.zeros(hidden, dtype=float)
    fixed_loader = torch.utils.data.DataLoader(ds_test, batch_size=64, shuffle=False)

    idx = 0
    for imgs, lbls in fixed_loader:
        spikes = poisson_encode(imgs, T).to(device)
        spk1, mem1_tr, spk2_rec, mem2_rec = untrained(spikes, T)
        # sum over time axis (axis=1) to get shape [batch, hidden], then sum batch
        counts_before += spk1.sum(dim=0).sum(dim=0).detach().cpu().numpy()
        idx += imgs.size(0)
        if idx >= 1000:
            break
    counts_before /= idx

    # 2. Compute after‑learning counts & gather preds
    counts_after = np.zeros(hidden, dtype=float)

    y_true, y_pred = [], []
    idx = 0
    for imgs, lbls in fixed_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        spikes = poisson_encode(imgs, T).to(device)
        spk1, mem1_tr, spk2_rec, mem2_rec = model(spikes, T)
        counts_after += spk1.sum(dim=0).sum(dim=0).detach().cpu().numpy()
        out = spk2_rec.sum(dim=0)  # sum spikes over time for output layer
        y_true.extend(lbls.cpu().numpy())
        y_pred.extend(out.argmax(dim=1).cpu().numpy())
        idx += imgs.size(0)
        if idx >= 1000:
            break
    counts_after /= idx

    # 3. Confusion Matrix (force 10×10)
    labels = list(range(10))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(6, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
    # fix ticks to match labels
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_title("Confusion Matrix")
    plt.show()

    # 4. Epoch vs. Accuracy
    plt.figure()
    plt.plot(np.arange(1, len(epoch_accs) + 1), epoch_accs, marker='o')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Epoch vs Training Accuracy")
    plt.grid(True)
    plt.show()

    # 5. Optuna plots
    from optuna.visualization.matplotlib import (
        plot_optimization_history,
        plot_parallel_coordinate,
        plot_param_importances,
        plot_slice,
        plot_terminator_improvement,
    )

    plot_funcs = [
        plot_optimization_history,
        lambda st: plot_parallel_coordinate(st, ['tau', 'hidden', 'lr', 'T']),
        plot_param_importances,
        plot_slice,
        plot_terminator_improvement,
    ]
    for fn in plot_funcs:
        ax = fn(study)
        # show each subplot
        if isinstance(ax, (list, tuple, np.ndarray)):
            for subax in np.array(ax).ravel():
                subax.figure.show()
        else:
            ax.figure.show()

    # 6. Spike raster per digit
    spike_trains = {i: None for i in range(10)}
    for imgs, lbls in test_loader:
        spikes = poisson_encode(imgs, T).to(device)
        spk1, mem1_tr, spk2_rec, mem2_rec = model(spikes, T)
        for j, lbl in enumerate(lbls.cpu().numpy()):
            if spike_trains[lbl] is None:
                # take the time series of neuron j
                spike_trains[lbl] = spk1[j].detach().cpu().numpy()
        if all(v is not None for v in spike_trains.values()):
            break

    fig, axs = plt.subplots(5, 2, figsize=(10, 12), sharex=True)
    for d, ax in enumerate(axs.flatten()):
        times = np.where(spike_trains[d] == 1)[0]
        ax.eventplot(times, orientation='horizontal')
        ax.set_title(f"Digit {d}")
    plt.tight_layout()
    plt.show()

    # 7. Weight‑hist evolution
    plt.figure(figsize=(8, 6))
    for i, w in enumerate(weight_histories):
        hist, bins = np.histogram(w, bins=50, range=(-0.5, 0.5), density=True)
        plt.plot(bins[:-1], hist, label=f"Ep {i + 1}")
    plt.legend()
    plt.title("FC1 Weight Distribution")
    plt.show()

    # 8. Spike activity before vs. after
    neurons = np.arange(hidden)
    width = 0.4
    plt.figure(figsize=(10, 4))
    plt.bar(neurons - width / 2, counts_before, width, label='Before')
    plt.bar(neurons + width / 2, counts_after, width, label='After')
    plt.xlabel("Hidden Neuron Index")
    plt.ylabel("Avg. Spike Count")
    plt.legend()
    plt.title("Spike Activity Before vs After")
    plt.show()

    # 9. Accuracy table per digit
    accs = {}
    for d in labels:
        idxs = [i for i, y in enumerate(y_true) if y == d]
        accs[d] = np.mean([y_pred[i] == y_true[i] for i in idxs]) if idxs else 0.0
    overall = np.mean([y_pred[i] == y_true[i] for i in range(len(y_true))])
    df = pd.DataFrame({
        "Digit": labels + ['Overall'],
        "Accuracy": [accs[d] for d in labels] + [overall]
    })
    print(df.to_markdown(index=False))

    # 10. Save model state_dict
    save_path = "snn_mnist_final_poission.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model state_dict saved to {save_path}")
    print(f"Final test accuracy: {overall:.4f}")

    # spk2_rec: [T, B, N_out] from your inference
    # choose one output neuron (e.g. neuron 0) and all trials in a small test subset
    spk_counts = spk_traces[0]  # shape [T]
    # if you have multiple trials of that same class, stack them:
    # psth_data = np.stack([spk_traces[class_id] for class_id in some_trials], axis=1)

    # Bin size and edges
    bin_size = 5   # in time-steps
    bins = np.arange(0, len(spk_counts)+bin_size, bin_size)
    hist, _ = np.histogram(np.where(spk_counts>0)[0], bins)

    # Convert to firing rate (spikes/bin_size per trial)
    rate = hist / (bin_size * 1.0)

    plt.figure(figsize=(6,4))
    plt.bar(bins[:-1], rate, width=bin_size, align='edge')
    plt.xlabel("Time step"); plt.ylabel("Firing rate (spikes/time-step)")
    plt.title("PSTH for Output Neuron 0")
    plt.show()

    # spk1_rec: [T, B, N_hidden] summed across a fixed subset (as counts_before/after)
    total_spikes = counts_after  # your post-learning per-neuron average

    plt.figure(figsize=(6,4))
    plt.hist(total_spikes, bins=50)
    plt.xlabel("Avg spikes per neuron"); plt.ylabel("Neuron count")
    plt.title("Hidden‑Layer Firing‑Rate Distribution (After Learning)")
    plt.show()

    # choose one hidden neuron, trial 0
    spike_times = np.where(spk_traces[0]>0)[0]
    isis = np.diff(spike_times)  # inter-spike intervals in time-steps

    plt.figure(figsize=(6,4))
    plt.hist(isis, bins=30)
    plt.xlabel("Inter‑spike interval (time‑steps)"); plt.ylabel("Count")
    plt.title("ISI Histogram for Hidden Neuron 0")
    plt.show()

    # Assume model.fc1.weight shape [N_hidden, 49]
    weights = model.fc1.weight.detach().cpu().numpy()  # [H,49]
    # Sum absolute weights of neurons that actually fired
    fired_neurons = np.where(total_spikes > np.percentile(total_spikes, 80))[0]
    saliency = np.abs(weights[fired_neurons]).sum(axis=0)  # [49]

    # reshape to 7×7
    sal_map = saliency.reshape(7,7)
    plt.figure(figsize=(4,4))
    plt.imshow(sal_map, cmap='hot', interpolation='nearest')
    plt.colorbar(label="Activation strength")
    plt.title("Spike Activation Map (Input 7×7 Regions)")
    plt.show()

    # Binarize labels
    y_bin = label_binarize(y_true, classes=list(range(10)))
    scores = np.stack([spk_traces[c].sum() for c in range(10)])  # per-class spike counts

    plt.figure(figsize=(6,6))
    for c in range(10):
        fpr, tpr, _ = roc_curve(y_bin[:,c], np.array([out.count(c) for out in y_pred]) )
        plt.plot(fpr, tpr, label=f"Class {c} (AUC={auc(fpr,tpr):.2f})")

    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Multiclass ROC Curves")
    plt.legend(loc="lower right")
    plt.show()


    plt.figure(figsize=(6,6))
    for c in range(10):
        y_c_true = (np.array(y_true)==c).astype(int)
        y_c_score= np.array([spk_traces[c].sum() for _ in y_true])
        precision, recall, _ = precision_recall_curve(y_c_true, y_c_score)
        ap = average_precision_score(y_c_true, y_c_score)
        plt.plot(recall, precision, label=f"Class {c} (AP={ap:.2f})")

    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision–Recall Curves")
    plt.legend(loc="upper right")
    plt.show()

    # matrix X: samples × HIDDEN, here use counts_after for each of 100 samples
    X = np.vstack([counts_after for _ in range(100)])
    y = y_true[:100]

    pca = PCA(n_components=2)
    Z   = pca.fit_transform(X)

    plt.figure(figsize=(6,6))
    for c in range(10):
        idx = np.where(np.array(y)==c)
        plt.scatter(Z[idx,0], Z[idx,1], label=f"Class {c}", s=10)
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.title("PCA of Hidden‑Layer Spike Counts")
    plt.legend(markerscale=2, bbox_to_anchor=(1.05,1))
    plt.show()

    plt.figure(figsize=(8,5))
    T = len(mem_smooth[0])
    t = np.arange(T)
    templates = np.stack([ (c/9.0)*t for c in range(10) ])

    for c in range(10):
        plt.plot(t, mem_smooth[c].detach().cpu().numpy(),
                 label=f"Mem Class {c}", alpha=0.6)
        plt.plot(t, templates[c], linestyle='--',
                 label=f"Template {c}")

    plt.xlabel("Time step"); plt.ylabel("Membrane potential")
    plt.title("Actual vs Target Waveforms")
    plt.legend(ncol=2, fontsize='small')
    plt.show()
