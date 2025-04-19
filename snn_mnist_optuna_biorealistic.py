#!/usr/bin/env python3
"""
snn_mnist_optuna_biorealistic.py

Enhanced MNIST SNN with dynamic spatiotemporal encoding, AdEx neurons,
heterogeneous time constants, and mild recurrence. Integrated with
Optuna HPO and full plotting hooks:
 1. Confusion matrix
 2. Epoch vs. accuracy
 3. Optuna HPO visualizations
 4. Spike raster per digit
 5. Weight distribution evolution
 6. Spike activity before vs after learning
 7. Per-class accuracy table
Also saves the trained model state_dict.
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import snntorch as snn
from snntorch import spikegen as spkgen
from snntorch import surrogate
# Removed neuron.AdEx (not available)
# Use Synaptic conductance-based LIF model instead
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.exceptions import TrialPruned, ExperimentalWarning
import warnings
warnings.filterwarnings("ignore", category=ExperimentalWarning)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optuna.visualization.matplotlib import (
    plot_optimization_history,
    plot_parallel_coordinate,
    plot_param_importances,
    plot_slice,
    plot_terminator_improvement
)

# Device selection

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():    return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Spatiotemporal sliding-bar encoder
BAR_WIDTH = 5
NUM_POS   = 28 - BAR_WIDTH + 1

def sliding_bar_encode(imgs, num_steps):
    B = imgs.size(0)
    rates = []
    for t in range(num_steps):
        pos = t % NUM_POS
        mask = torch.zeros_like(imgs)
        mask[:, :, :, pos:pos+BAR_WIDTH] = 1.0
        patch = imgs * mask
        inten = nn.functional.avg_pool2d(patch, 4).view(B, -1)
        inten = inten / (inten.max(dim=1, keepdim=True)[0] + 1e-12)
        rates.append(inten)
    rates = torch.stack(rates)  # [T, B, features]
    return spkgen.rate(rates, num_steps=num_steps)  # pass correct num_steps to avoid NoneType error

# Bio-inspired SNN: FC→AdEx→FC→AdEx with recurrence & hetero taus
class BioSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, tau_min, tau_max):
        super().__init__()
        # Hidden layer: heterogeneous synaptic/membrane time constants
        taus1 = torch.linspace(tau_min, tau_max, num_hidden, device=device)
        alpha1 = torch.exp(-1.0 / taus1)
        beta1  = alpha1
        self.fc1  = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Synaptic(alpha=alpha1, beta=beta1, spike_grad=surrogate.fast_sigmoid())
        # Recurrent output layer
        self.fc_rec = nn.Linear(num_hidden + num_outputs, num_outputs)
        # Output layer: heterogeneous time constants
        taus2 = torch.linspace(tau_min, tau_max, num_outputs, device=device)
        alpha2 = torch.exp(-1.0 / taus2)
        beta2  = alpha2
        self.lif2 = snn.Synaptic(alpha=alpha2, beta=beta2, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x, num_steps):
        B = x.size(1)
        # Initialize synaptic and membrane states
        # initialize synaptic & membrane state manually for batch
        syn1 = torch.zeros(B, self.fc1.out_features, device=device)
        mem1 = torch.zeros(B, self.fc1.out_features, device=device)  # initialize synaptic & membrane state
        # initialize synaptic & membrane state for output layer manually
        syn2 = torch.zeros(B, self.fc_rec.out_features, device=device)
        mem2 = torch.zeros(B, self.fc_rec.out_features, device=device)  # initialize synaptic & membrane state
        rec_spk = torch.zeros(B, self.fc_rec.out_features, device=device)

        trace1      = []
        mem1_trace  = []
        rec_trace2  = torch.zeros(num_steps, B, self.fc_rec.out_features, device=device)

        for t in range(num_steps):
            # Hidden layer
            cur1       = self.fc1(x[t])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
            trace1.append(spk1[0].detach().cpu())
            mem1_trace.append(mem1[0].mean().item())  # record mean membrane potential for batch 0  # record membrane of neuron 0 in batch 0  # mem1 is [B,features], index batch 0 then feature 0
            # Output layer with recurrence
            inp2       = torch.cat([spk1, rec_spk], dim=1)
            cur2       = self.fc_rec(inp2)
            spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
            rec_spk    = spk2
            rec_trace2[t] = spk2

        return torch.stack(trace1), mem1_trace, rec_trace2

# Optuna objective
def objective(trial):
    tau_min = trial.suggest_float("tau_min", 5.0, 10.0)
    tau_max = trial.suggest_float("tau_max", 10.0, 30.0)
    hidden  = trial.suggest_int("hidden", 20, 100)
    lr      = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    T       = trial.suggest_int("T", 50, 200)

    model = BioSNN(49, hidden, 10, tau_min, tau_max).to(device)
    opt   = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    tf = transforms.ToTensor()
    train_ds = torchvision.datasets.MNIST('.', train=True,  download=True, transform=tf)
    val_ds   = torchvision.datasets.MNIST('.', train=False, download=True, transform=tf)
    tr = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    va = torch.utils.data.DataLoader(val_ds,   batch_size=64, shuffle=False)

    for epoch in range(3):
        spikes_sum, corr, n = 0.0, 0, 0
        for i,(imgs,labels) in enumerate(tr):
            imgs,labels = imgs.to(device), labels.to(device)
            spikes = sliding_bar_encode(imgs, T).to(device)
            opt.zero_grad()
            t1,_,spk2 = model(spikes, T)
            out = spk2.sum(0)
            loss = loss_fn(out, labels)
            loss.backward(); opt.step()
            corr += (out.argmax(1)==labels).sum().item()
            spikes_sum += t1.sum().item(); n += imgs.size(0)
            if i>=10: break
        acc = corr/n
        rate = spikes_sum/(n*T)
        trial.report(acc, epoch*2)
        trial.report(rate,epoch*2+1)
        if trial.should_prune(): raise TrialPruned()

    c,t = 0,0
    with torch.no_grad():
        for imgs,labels in va:
            imgs,labels = imgs.to(device),labels.to(device)
            spikes = sliding_bar_encode(imgs,T).to(device)
            _,_,spk2 = model(spikes,T)
            c += (spk2.sum(0).argmax(1)==labels).sum().item()
            t += labels.size(0)
    return c/t

# Main & plotting
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--timeout',type=int, default=None)
    args = p.parse_args()

    study = optuna.create_study(
        study_name='BNC_mnist_snn_tuning', direction='maximize',
        pruner=SuccessiveHalvingPruner(), storage='sqlite:///mnist.db', load_if_exists=True
    )
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)
    print('Best hyperparameters:', study.best_trial.params)

    # Retrain final model
    best = study.best_trial.params
    model = BioSNN(49, best['hidden'], 10, best['tau_min'], best['tau_max']).to(device)
    opt   = optim.Adam(model.parameters(), lr=best['lr'])
    loss_fn = nn.CrossEntropyLoss()
    tf = transforms.ToTensor()
    train_ds = torchvision.datasets.MNIST('.', train=True,  download=True, transform=tf)
    test_ds  = torchvision.datasets.MNIST('.', train=False, download=True, transform=tf)
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=64, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test_ds,  batch_size=64, shuffle=False)

    # Training loop with history
    epoch_accs = []
    weight_histories = []
    for e in range(10):
        tot_loss, corr = 0.0, 0
        for imgs,labels in train_loader:
            imgs,labels=imgs.to(device),labels.to(device)
            spikes = sliding_bar_encode(imgs, best['T']).to(device)
            opt.zero_grad()
            _,_,spk2 = model(spikes, best['T'])
            out = spk2.sum(0)
            loss = loss_fn(out,labels)
            loss.backward(); opt.step()
            tot_loss += loss.item()
            corr += (out.argmax(1)==labels).sum().item()
        acc = corr/len(train_ds)
        epoch_accs.append(acc)
        weight_histories.append(model.fc1.weight.detach().cpu().numpy().ravel())
        print(f'Epoch {e+1}: loss {tot_loss/len(train_loader):.4f} acc {acc:.4f}')

    # Save model
    save_path = 'snn_mnist_bio.pth'
    torch.save(model.state_dict(), save_path)
    print(f'Model state_dict saved to {save_path}')

    # 1. Confusion Matrix
    y_true, y_pred = [], []
    with torch.no_grad():
        for imgs,labels in test_loader:
            imgs,labels=imgs.to(device),labels.to(device)
            spikes=sliding_bar_encode(imgs,best['T']).to(device)
            _,_,spk2=model(spikes,best['T'])
            out=spk2.sum(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(out.argmax(1).cpu().numpy())
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=range(10))
    fig, ax = plt.subplots(figsize=(6,6))
    disp.plot(ax=ax, cmap=plt.cm.Blues)
    ax.set_title('Confusion Matrix')
    plt.show()

    # 2. Epoch vs Accuracy
    plt.figure()
    plt.plot(range(1,len(epoch_accs)+1), epoch_accs, marker='o')
    plt.xlabel('Epoch'); plt.ylabel('Accuracy')
    plt.title('Epoch vs Training Accuracy')
    plt.grid(True); plt.show()

    # 3. Optuna plots
    plot_funcs = [
        plot_optimization_history,
        lambda st: plot_parallel_coordinate(st,['tau_min','tau_max','hidden','lr','T']),
        plot_param_importances,
        plot_slice,
        plot_terminator_improvement,
    ]
    for fn in plot_funcs:
        ax = fn(study)
        # handle Axes or array of Axes
        if hasattr(ax,'figure'):
            ax.figure.show()
        else:
            for subax in np.asarray(ax).ravel(): subax.figure.show()

    # 4. Spike raster per digit
    spike_trains = {i:[] for i in range(10)}
    for imgs,labels in test_loader:
        imgs,labels=imgs.to(device),labels.to(device)
        spikes=sliding_bar_encode(imgs,best['T']).to(device)
        tr1,_,_ = model(spikes,best['T'])
        for j,lbl in enumerate(labels.cpu().numpy()):
            if not spike_trains[lbl]: spike_trains[lbl] = tr1[:,j].cpu().numpy()
        if all(spike_trains.values()): break
    fig, axs = plt.subplots(5,2,figsize=(10,12),sharex=True)
    for d,ax in enumerate(axs.flatten()):
        ax.eventplot(np.where(spike_trains[d]==1)[0],orientation='horizontal')
        ax.set_title(f'Digit {d}')
    plt.tight_layout(); plt.show()

    # 5. Weight distribution evolution
    plt.figure(figsize=(8,6))
    for i,w in enumerate(weight_histories):
        hist,bins = np.histogram(w,bins=50,range=(-0.5,0.5),density=True)
        plt.plot(bins[:-1],hist,label=f'Ep{i+1}')
    plt.legend(); plt.title('FC1 Weight Distribution'); plt.show()

    # 6. Spike activity before vs after
    counts_before = np.zeros_like(weight_histories[0])
    counts_after  = np.zeros_like(counts_before)
    cnt=0
    for imgs,labels in test_loader:
        imgs,labels=imgs.to(device),labels.to(device)
        spikes=sliding_bar_encode(imgs,best['T']).to(device)
        tr1,_,_ = model(spikes,best['T'])
        counts_after += tr1.sum(axis=0).cpu().numpy()
        cnt += imgs.size(0)
        if cnt>=1000: break
    counts_after /= cnt
    # before learning
    untrained = BioSNN(49,best['hidden'],10,best['tau_min'],best['tau_max']).to(device)
    cnt=0
    for imgs,_ in test_loader:
        imgs=imgs.to(device)
        spikes=sliding_bar_encode(imgs,best['T']).to(device)
        tr1,_,_ = untrained(spikes,best['T'])
        counts_before += tr1.sum(axis=0).cpu().numpy()
        cnt += imgs.size(0)
        if cnt>=1000: break
    counts_before /= cnt
    indices = np.arange(len(counts_before)); width=0.4
    plt.figure(figsize=(10,4))
    plt.bar(indices-width/2, counts_before, width, label='Before')
    plt.bar(indices+width/2, counts_after,  width, label='After')
    plt.xlabel('Neuron'); plt.ylabel('Avg spike count')
    plt.legend(); plt.title('Spike Activity Before vs After'); plt.show()

    # 7. Per-class accuracy table
    accs = {}
    for d in range(10):
        idxs = [i for i,y in enumerate(y_true) if y==d]
        accs[d] = sum(y_pred[i]==y_true[i] for i in idxs) / len(idxs)
    overall = sum(y_pred[i]==y_true[i] for i in range(len(y_true))) / len(y_true)
    df = pd.DataFrame({
        'Digit': list(accs.keys()) + ['Overall'],
        'Accuracy': list(accs.values()) + [overall]
    })
    print(df.to_markdown(index=False))
