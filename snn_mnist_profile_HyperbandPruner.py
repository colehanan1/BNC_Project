#!/usr/bin/env python3
"""
snn_mnist_optuna.py

Spiking Neural Network hyperparameter optimization with Optuna and snnTorch:
 - 7×7 Poisson rate encoding
 - Single hidden LIF layer → output LIF layer (no convolution)
 - Surrogate‑gradient backpropagation
 - Optuna Bayesian HPO with HyperbandPruner
 - Pruning on validation accuracy and hidden spike rate (silent‑network detection)
 - Fix membrane‑trace error (use mem1[0,0].item())
 - Device selection: MPS → CUDA → CPU
 - After HPO and final training, generate:
     • Optuna optimization history plot
     • Epoch vs. training accuracy plot
     • Confusion matrix on test set
     • Hidden‑layer spike‑raster per digit
     • Weight‑distribution evolution
     • Spike‑activity before vs. after learning
     • Per‑class accuracy table
"""

import argparse
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import transforms
import snntorch as snn
from snntorch import spikegen as spkgen
from snntorch import surrogate
import optuna
from optuna.pruners import HyperbandPruner
from optuna.exceptions import TrialPruned
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Device selection
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
            cur1, mem1 = self.lif1(self.fc1(x[t]), mem1)
            spk1 = cur1 if hasattr(cur1, 'dtype') else None  # placeholder
            spk1 = (cur1 >= self.lif1.threshold).float() if spk1 is None else spk1
            spk1_trace.append(spk1[0].detach().cpu())
            mem1_trace.append(mem1[0,0].item())
            cur2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk2_rec[t] = cur2 if hasattr(cur2, 'dtype') else (cur2 >= self.lif2.threshold).float()
        return torch.stack(spk1_trace), mem1_trace, spk2_rec

# Optuna objective
def objective(trial):
    tau    = trial.suggest_float("tau",    5.0, 20.0)
    hidden = trial.suggest_int("hidden",  20,  100)
    lr     = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    T      = trial.suggest_int("T",     50,   200)
    beta   = torch.exp(torch.tensor(-1.0 / tau))
    model  = SNN(49, hidden, 10, beta, beta).to(device)
    opt    = optim.Adam(model.parameters(), lr=lr)
    loss_fn= nn.CrossEntropyLoss()
    tf     = transforms.Compose([transforms.ToTensor()])
    ds_train = torchvision.datasets.MNIST(".", train=True,  download=True, transform=tf)
    ds_val   = torchvision.datasets.MNIST(".", train=False, download=True, transform=tf)
    loader_train = torch.utils.data.DataLoader(ds_train, batch_size=64, shuffle=True)
    loader_val   = torch.utils.data.DataLoader(ds_val,   batch_size=64, shuffle=False)
    for epoch in range(3):
        total_spikes = 0; correct = 0; num = 0
        for i,(img,lbl) in enumerate(loader_train):
            img,lbl = img.to(device),lbl.to(device)
            spikes  = poisson_encode(img, T).to(device)
            opt.zero_grad()
            spk1, _, spk2 = model(spikes, T)
            out = spk2.sum(0)
            loss= loss_fn(out, lbl)
            loss.backward(); opt.step()
            preds = out.argmax(1)
            correct += (preds==lbl).sum().item()
            total_spikes += spk1.sum().item()
            num += lbl.size(0)
            if i>=10: break
        val_acc  = correct/num
        avg_rate = total_spikes/(num*T)
        trial.report(val_acc,  epoch*2)
        trial.report(avg_rate, epoch*2+1)
        if trial.should_prune(): raise TrialPruned()
    # final
    correct=0; total=0
    with torch.no_grad():
        for img,lbl in loader_val:
            img,lbl = img.to(device),lbl.to(device)
            spikes  = poisson_encode(img, T).to(device)
            _,_,spk2= model(spikes,T)
            out=spk2.sum(0)
            correct+= (out.argmax(1)==lbl).sum().item()
            total+=lbl.size(0)
    return correct/total

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--timeout",type=int, default=None)
    args=parser.parse_args()

    study=optuna.create_study(
        study_name="BNC_mnist_snn_tuning",
        direction="maximize",
        pruner=HyperbandPruner(min_resource=10, max_resource=100, reduction_factor=3),
        storage="sqlite:///mnist_snn_tuning.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=args.trials, timeout=args.timeout)

    # Optuna plots
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig1.show(); fig2.show()

    print("Best params:", study.best_trial.params)
    tau, hidden, lr, T = study.best_trial.params.values()
    beta = torch.exp(torch.tensor(-1.0/tau))

    # Final training
    model= SNN(49, hidden, 10, beta, beta).to(device)
    opt  = optim.Adam(model.parameters(), lr=lr)
    loss_fn=nn.CrossEntropyLoss()
    tf=transforms.Compose([transforms.ToTensor()])
    ds_tr= torchvision.datasets.MNIST(".",train=True,download=True,transform=tf)
    ds_te= torchvision.datasets.MNIST(".",train=False,download=True,transform=tf)
    ld_tr= torch.utils.data.DataLoader(ds_tr, batch_size=64, shuffle=True)
    ld_te= torch.utils.data.DataLoader(ds_te, batch_size=64, shuffle=False)

    epoch_acc=[]; weight_hist=[]
    for e in range(10):
        model.train(); total, corr = 0.0, 0
        for img,lbl in ld_tr:
            img,lbl = img.to(device),lbl.to(device)
            spikes  = poisson_encode(img, T).to(device)
            opt.zero_grad()
            _,_,spk2 = model(spikes,T)
            out=spk2.sum(0)
            loss=loss_fn(out,lbl)
            loss.backward(); opt.step()
            corr+= (out.argmax(1)==lbl).sum().item()
            total+=loss.item()
        acc=corr/len(ds_tr)
        epoch_acc.append(acc)
        weight_hist.append(model.fc1.weight.detach().cpu().flatten().numpy().copy())
        print(f"Epoch {e+1} acc {acc:.4f}")

    # Epoch vs accuracy
    plt.figure(); plt.plot(range(1,11), epoch_acc, marker='o')
    plt.xlabel("Epoch"); plt.ylabel("Accuracy"); plt.title("Epoch vs. Training Accuracy"); plt.show()

    # Weight evolution
    plt.figure()
    for i,w in enumerate(weight_hist):
        hist,bins=np.histogram(w, bins=50, range=(-0.5,0.5), density=True)
        plt.plot(bins[:-1], hist, label=f"Ep{i+1}")
    plt.legend(); plt.title("Weight Distribution Evolution"); plt.show()

    # Confusion matrix
    y_t,y_p=[],[]
    model.eval()
    with torch.no_grad():
        for img,lbl in ld_te:
            img,lbl=img.to(device),lbl.to(device)
            spikes=poisson_encode(img,T).to(device)
            _,_,spk2=model(spikes,T)
            out=spk2.sum(0)
            y_t.extend(lbl.cpu().numpy()); y_p.extend(out.argmax(1).cpu().numpy())
    cm=confusion_matrix(y_t,y_p,labels=list(range(10)))
    disp=ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=list(range(10)))
    fig,ax=plt.subplots(figsize=(6,6))
    disp.plot(ax=ax,cmap=plt.cm.Blues); plt.title("Confusion Matrix"); plt.show()

    # Spike raster per digit
    spike_trains={d:None for d in range(10)}
    with torch.no_grad():
        for img,lbl in ld_te:
            img,lbl=img.to(device),lbl.to(device)
            spikes=poisson_encode(img,T).to(device)
            spk1,_,_ = model(spikes,T)
            for i,lab in enumerate(lbl.cpu().numpy()):
                if spike_trains[lab] is None:
                    spike_trains[lab]=spk1[:,i].cpu().numpy()
            if all(v is not None for v in spike_trains.values()):
                break
    fig,axs=plt.subplots(5,2,figsize=(10,12),sharex=True,sharey=True)
    for d,ax in enumerate(axs.flatten()):
        times=np.where(spike_trains[d]==1)[0]
        ax.eventplot(times, orientation='horizontal', colors='k')
        ax.set_title(f"Digit {d}")
    plt.tight_layout(); plt.show()

    # Spike activity before vs after
    def get_counts(m):
        cnt=np.zeros(m.fc1.out_features)
        with torch.no_grad():
            for img,_ in ld_te:
                img=img.to(device)
                spikes=poisson_encode(img,T).to(device)
                spk1,_,_ = m(spikes,T)
                cnt+=spk1.sum(1).cpu().numpy()
        return cnt/len(ds_te)
    before_counts=get_counts(SNN(49,hidden,10,beta,beta).to(device))
    after_counts =get_counts(model)
    idx=np.arange(len(before_counts))
    width=0.4
    plt.figure(figsize=(10,4))
    plt.bar(idx-width/2, before_counts, width, label="Before")
    plt.bar(idx+width/2, after_counts,  width, label="After")
    plt.xlabel("Neuron"); plt.ylabel("Avg Spike Count")
    plt.title("Spike Activity Before vs. After Learning"); plt.legend(); plt.show()

    # Per-class accuracy table
    accs={}
    for d in range(10):
        inds=[i for i,y in enumerate(y_t) if y==d]
        accs[d]=sum(y_p[i]==y_t[i] for i in inds)/len(inds)
    overall=sum(y_p[i]==y_t[i] for i in range(len(y_t)))/len(y_t)
    df=pd.DataFrame({
        "Digit": list(accs.keys())+["Overall"],
        "Accuracy": list(accs.values())+[overall]
    })
    print(df.to_markdown(index=False))

if __name__=="__main__":
    main()
