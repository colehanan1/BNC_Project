#!/usr/bin/env python3
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from torchvision import datasets, transforms
import snntorch as snn
from snntorch import spikegen as spkgen, surrogate

# ─── Configuration ───────────────────────────────────────────────
DEVICE     = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
STATE_PATH = "snn_mnist_final_poission.pth"
# (replace with your actual best Optuna values if desired)
TAU        = 12.532392193484942
HIDDEN     = 203
T_STEPS    = 186
BETA       = torch.tensor(math.exp(-1.0 / TAU), device=DEVICE)

# ─── Model definition ────────────────────────────────────────────
class SNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta1, beta2):
        super().__init__()
        self.fc1   = nn.Linear(num_inputs, num_hidden)
        self.lif1  = snn.Leaky(beta=beta1, spike_grad=surrogate.fast_sigmoid())
        self.fc2   = nn.Linear(num_hidden, num_outputs)
        self.lif2  = snn.Leaky(beta=beta2, spike_grad=surrogate.fast_sigmoid())
        self.bias1 = nn.Parameter(torch.tensor(0.2, device=DEVICE))

    def forward(self, x, T):
        B = x.size(1)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        spk1_trace = []
        spk2_rec   = torch.zeros(T, B, self.fc2.out_features, device=DEVICE)
        for t in range(T):
            cur1 = self.fc1(x[t]) + self.bias1
            spk1, mem1 = self.lif1(cur1, mem1)
            spk1_trace.append(spk1.detach())
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec[t] = spk2
        return torch.stack(spk1_trace), spk2_rec

# ─── Poisson encoder ─────────────────────────────────────────────
pool7 = nn.AvgPool2d(4)
def poisson_encode(img, num_steps):
    inten = pool7(img).view(img.size(0), -1)
    inten = inten / (inten.max(dim=1, keepdim=True)[0] + 1e-12)
    return spkgen.rate(inten, num_steps=num_steps)

# ─── α‑kernel for AP‑shaping ───────────────────────────────────────
def alpha_kernel(L=50, tau_r=1.0, tau_f=5.0, dt=1.0):
    t    = torch.arange(0, L*dt, dt, device=DEVICE)
    kern = (t/tau_r)*torch.exp(1 - t/tau_r)*torch.exp(-t/tau_f)
    return kern.view(1,1,-1)

# ─── Load trained model ───────────────────────────────────────────
model = SNN(49, HIDDEN, 10, BETA, BETA).to(DEVICE)
model.load_state_dict(torch.load(STATE_PATH, map_location=DEVICE))
model.eval()

# ─── Prepare test data loader ─────────────────────────────────────
test_ds = datasets.MNIST(".", train=False, download=True,
                         transform=transforms.ToTensor())
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=64, shuffle=False)

# ─── Collect one spike train per class ────────────────────────────
spk_traces = {}
y_true, y_pred = [], []
for imgs, lbls in test_loader:
    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
    spikes     = poisson_encode(imgs, T_STEPS).to(DEVICE)
    _, spk2    = model(spikes, T_STEPS)
    out        = spk2.sum(dim=0)
    preds      = out.argmax(dim=1)
    y_true.extend(lbls.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())
    for i, c in enumerate(lbls.cpu().numpy()):
        if c not in spk_traces:
            spk_traces[c] = spk2[:, i, c].cpu().float()
    if len(spk_traces) == 10:
        break

# ─── AP‑shaped waveforms via α‑kernel ─────────────────────────────
kernel    = alpha_kernel().to(DEVICE)
ap_traces = {}
for c, spk in spk_traces.items():
    spk = spk.view(1,1,-1).to(DEVICE)
    ap  = F.conv1d(spk, kernel, padding=kernel.size(-1)//2)
    ap_traces[c] = ap.view(-1).detach().cpu().numpy()

# ─── Plot α‑kernel waveforms (10×1 vertical) ─────────────────────
fig, axes = plt.subplots(10, 1, figsize=(6, 20), sharex=True, sharey=True)
for c in range(10):
    trace = ap_traces[c]
    ax    = axes[c]
    ax.plot(trace)
    ax.set_title(f"Class {c}")
    ax.set_ylim(0, 2)
    ax.set_xlabel("Time step")
    ax.set_ylabel("Filtered spike")
    ax.grid(True)
fig.suptitle("α‑Kernel Filtered Spike Waveforms", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()

# ─── Compute before/after spike counts ────────────────────────────
# before: untrained random weights
untrained   = SNN(49, HIDDEN, 10, BETA, BETA).to(DEVICE)
counts_before = np.zeros(HIDDEN, dtype=float)
counts_after  = np.zeros(HIDDEN, dtype=float)
cnt = 0
for imgs, _ in test_loader:
    imgs = imgs.to(DEVICE)
    spikes = poisson_encode(imgs, T_STEPS).to(DEVICE)
    spk1_u, _ = untrained(spikes, T_STEPS)
    spk1_t, _ = model(spikes, T_STEPS)
    counts_before += spk1_u.sum(dim=0).sum(dim=0).detach().cpu().numpy()
    counts_after  += spk1_t.sum(dim=0).sum(dim=0).detach().cpu().numpy()
    cnt += imgs.size(0)
    if cnt >= 1000: break
counts_before /= cnt
counts_after  /= cnt

# ─── Plot before vs after (subset first 50 neurons) ─────────────
neurons = np.arange(min(50, HIDDEN))
width   = 0.4
plt.figure(figsize=(10, 4))
plt.bar(neurons - width/2, counts_before[neurons], width, label='Before')
plt.bar(neurons + width/2, counts_after[neurons],  width, label='After')
plt.xlabel("Hidden Neuron Index"); plt.ylabel("Avg. Spike Count")
plt.legend(); plt.title("Spike Activity Before vs After"); plt.show()

# ─── Confusion Matrix ────────────────────────────────────────────
labels = list(range(10))
cm = confusion_matrix(y_true, y_pred, labels=labels)
fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=True)
ax.set_title("Confusion Matrix"); ax.set_xticks(labels); ax.set_yticks(labels)
plt.show()

# ─── Spike raster per digit ──────────────────────────────────────
spike_trains = {i: None for i in range(10)}
for imgs, lbls in test_loader:
    imgs = imgs.to(DEVICE)
    spikes = poisson_encode(imgs, T_STEPS).to(DEVICE)
    spk1, _ = model(spikes, T_STEPS)
    for j, d in enumerate(lbls.cpu().numpy()):
        if spike_trains[d] is None:
            spike_trains[d] = spk1[j].detach().cpu().numpy()
    if all(v is not None for v in spike_trains.values()):
        break

fig, axs = plt.subplots(5, 2, figsize=(10, 12), sharex=True)
for d, ax in enumerate(axs.flatten()):
    times = np.where(spike_trains[d] == 1)[0]
    ax.eventplot(times, orientation='horizontal'); ax.set_title(f"Digit {d}")
plt.tight_layout(); plt.show()

# ─── Accuracy table per digit ─────────────────────────────────────
accs = {d: np.mean([pred==true for pred,true in zip(y_pred,y_true) if true==d])
        for d in labels}
overall = np.mean([p==t for p,t in zip(y_pred,y_true)])
df = pd.DataFrame({
    "Digit": labels + ['Overall'],
    "Accuracy": [accs[d] for d in labels] + [overall]
})
print(df.to_markdown(index=False))
