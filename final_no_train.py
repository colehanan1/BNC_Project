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
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.decomposition import PCA

def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():    return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")



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
spk_traces = {c: [] for c in range(10)}
y_true, y_pred = [], []
max_samples_per_class = 100  # Increase this number for more validation samples
class_counts = {c: 0 for c in range(10)}
for imgs, lbls in test_loader:
    imgs, lbls = imgs.to(DEVICE), lbls.to(DEVICE)
    spikes = poisson_encode(imgs, T_STEPS).to(device)
    spk1, mem1_tr, spk2_rec, mem2_rec = model(spikes, T_STEPS)
    out        = spk2_rec.sum(dim=0)
    preds      = out.argmax(dim=1)
    y_true.extend(lbls.cpu().numpy())
    y_pred.extend(preds.cpu().numpy())

    for i, c in enumerate(lbls.cpu().numpy()):
        if class_counts[c] < max_samples_per_class:

            spk_traces[c].append(spk2_rec[:, i, c].detach().cpu().float())
            class_counts[c] += 1

        # Check if we have enough samples for all classes
    if all(count >= max_samples_per_class for count in class_counts.values()):
        break

# ─── AP‑shaped waveforms via α‑kernel ─────────────────────────────
kernel    = alpha_kernel().to(DEVICE)
ap_traces = {}
for c, spk in spk_traces.items():
    if spk:
        spk_tensor = spk[0]  # Or torch.stack(spk_list).mean(dim=0)
        spk = spk_tensor.view(1, 1, -1).to(DEVICE)
        ap  = F.conv1d(spk, kernel, padding=kernel.size(-1)//2)
        ap_traces[c] = ap.view(-1).detach().cpu().numpy()

# ─── Plot α‑kernel waveforms (3×3 grid, wider figure) ─────────────
fig, axes = plt.subplots(3, 3, figsize=(18, 8), sharex=True, sharey=True)  # Increased width
axes = axes.flatten()  # Flatten the 2D array of axes for easier iteration

for c in range(10):
    if c < len(axes):  # Ensure we don't access out-of-bounds indices
        ax = axes[c]
        trace = ap_traces[c]
        ax.plot(trace)
        ax.set_title(f"Class {c}")
        ax.set_ylim(0, 1.2)
        ax.set_xlabel("Time step")
        ax.set_ylabel("Filtered spike")
        ax.grid(True)

# Turn off any unused subplots
for ax in axes[9:]:
    ax.axis('off')

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
    spk1_u, mem1_u, spk2_u, mem2_u = untrained(spikes, T_STEPS)
    spk1_t, mem1_t, spk2_t, mem2_t = model(spikes, T_STEPS)
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
    spk1, _, _, _ = model(spikes, T_STEPS)
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


if isinstance(spk_traces[0], list) and spk_traces[0]:
    # Use the first example from class 0
    spk_data = spk_traces[0][0].numpy()
else:
    spk_data = spk_traces[0]

spike_times = np.where(spk_data > 0)[0]
isis = np.diff(spike_times)
# Bin size and edges
bin_size = 5   # in time-steps
bins = np.arange(0, len(spk_data)+bin_size, bin_size)
hist, _ = np.histogram(np.where(spk_data>0)[0], bins)

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
# Fix: Add detach() before converting to numpy
scores = np.stack([torch.stack(spk_traces[c]).mean(dim=0).sum().detach().numpy() for c in range(10)])

plt.figure(figsize=(6,6))
for c in range(10):
    y_scores = np.array([1 if pred == c else 0 for pred in y_pred])
    fpr, tpr, _ = roc_curve(y_bin[:,c], y_scores)
    plt.plot(fpr, tpr, label=f"Class {c} (AUC={auc(fpr,tpr):.2f})")

plt.plot([0,1],[0,1],'k--')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Multiclass ROC Curves")
plt.legend(loc="lower right")
plt.show()

plt.figure(figsize=(6,6))
for c in range(10):
    y_c_true = (np.array(y_true)==c).astype(int)
    y_c_score = np.array([torch.stack(spk_traces[c]).mean(dim=0).sum().detach().numpy() for _ in y_true])
    precision, recall, _ = precision_recall_curve(y_c_true, y_c_score)
    ap = average_precision_score(y_c_true, y_c_score)
    plt.plot(recall, precision, label=f"Class {c} (AP={ap:.2f})")

plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision–Recall Curves")
plt.legend(loc="upper right")
plt.show()

# matrix X: samples × HIDDEN, here use counts_after for each of 100 samples
X = np.vstack([counts_after for _ in range(100)])
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

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

mem_traces = {c: None for c in range(10)}
spk_traces = {c: None for c in range(10)}
for imgs, lbls in test_loader:
    imgs, lbls = imgs.to(device), lbls.to(device)
    spikes = poisson_encode(imgs, T_STEPS).to(device)
    spk1, mem1_tr, spk2_rec, mem2_rec = model(spikes, T_STEPS)
    for i, c in enumerate(lbls.cpu().numpy()):
        if mem_traces[c] is None:
            mem_traces[c] = mem2_rec[:, i, c].cpu()
            spk_traces[c] = spk2_rec[:, i, c].cpu().float()
    if all(v is not None for v in mem_traces.values()): break

alpha_filt = 0.05
def smooth(trace):
    out = []
    prev = trace[0]
    for v in trace:
        prev = alpha_filt * v + (1 - alpha_filt) * prev
        out.append(prev)
    return torch.stack(out)

mem_smooth = {c: smooth(mem_traces[c]) for c in mem_traces}

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


# ─── Manually pick a digit image and predict ──────────────────────
import random

# Select a random digit image from test set
img_idx = random.randint(0, len(test_ds) - 1)
img, label = test_ds[img_idx]

img_input = img.unsqueeze(0).to(DEVICE)  # add batch dim

# Encode and run through model
encoded = poisson_encode(img_input, T_STEPS).to(DEVICE)  # shape [T, B=1, 49]
_, spk2_out = model(encoded, T_STEPS)                    # spk2_out: [T, 1, 10]

# Prediction from summed spikes
out_sum = spk2_out.sum(dim=0)     # [1, 10]
pred = out_sum.argmax(dim=1).item()
print(f"True label: {label}, Predicted: {pred}")

# ─── α‑kernel convolution for AP-shaped waveform ───────────────────
kernel = alpha_kernel().to(DEVICE)
ap_waveforms = []
for c in range(10):
    spikes = spk2_out[:, 0, c].view(1, 1, -1)  # [1,1,T]
    ap = F.conv1d(spikes, kernel, padding=kernel.size(-1) // 2)
    ap_waveforms.append(ap.view(-1).detach().cpu().numpy())

# ─── Plot all 10 class neuron responses ───────────────────────────
fig, axes = plt.subplots(10, 1, figsize=(6, 20), sharex=True, sharey=True)
for c in range(10):
    ax = axes[c]
    ax.plot(ap_waveforms[c])
    ax.set_title(f"Class {c}")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Filtered spike")
    ax.grid(True)
axes[-1].set_xlabel("Time step")
fig.suptitle("α‑Kernel Filtered Spike Waveforms for Single Input", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
plt.figure(figsize=(2,2))
plt.imshow(img.squeeze(), cmap='gray')
plt.title(f"Digit Image (Label: {label}, Pred: {pred})")
plt.axis('off')
plt.show()



