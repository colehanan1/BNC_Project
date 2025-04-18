#!/usr/bin/env python3
"""
retina_snn_mnist_conv_full.py

Spiking CNN + R‑STDP for MNIST, with:

 - Retina-inspired 7×7 Poisson encoding
 - Conv2d input→hidden (dynamic hidden size)
 - Hidden LIF neurons + placeholder STDP
 - Reward‑modulated STDP on output
 - Logging every 10 train steps
 - Interim eval + weight & receptive‑field plots every snapshot interval
 - Final accuracy‑vs‑samples curve + confusion matrix (ticks fixed)

Requires Python 3.9+ and:
    pip install torch torchvision matplotlib numpy scikit-learn
"""

import argparse
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Device selection
def get_device():
    if torch.backends.mps.is_available(): return torch.device("mps")
    if torch.cuda.is_available():    return torch.device("cuda")
    return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Retina‑inspired Poisson encoder
def encode_image_to_spikes(img, T=250, rate=100):
    x = img.unsqueeze(0)             # (1,1,28,28)
    x = F.avg_pool2d(x, 2, 2)         # (1,1,14,14)
    x = F.avg_pool2d(x, 2, 2)         # (1,1,7,7)
    inten = x.squeeze()               # (7,7)
    p = torch.clamp(inten.flatten() * (rate/1000), 0, 1)
    p = p.view(-1,1).expand(-1, T)    # (49,T)
    return (torch.rand(p.shape, device=img.device) < p).float(), inten

class SpikingNetwork:
    def __init__(self,
                 maps=8, kernel=5, stride=2,
                 tau_m=20., tau_ref=2,
                 A_plus=0.02, A_minus=0.01, w_max=0.5,
                 eta=0.01, alpha_e=0.95,
                 snapshot_every=1000):
        # STDP & R‑STDP params
        self.A_plus, self.A_minus, self.w_max = A_plus, A_minus, w_max
        self.tau_trace = 50.; self.alpha_trace = np.exp(-1/self.tau_trace)
        self.eta_rstdp = eta; self.alpha_e = alpha_e

        # Conv input→hidden
        self.conv = torch.nn.Conv2d(1, maps, kernel, stride=stride, bias=False).to(device)
        torch.nn.init.uniform_(self.conv.weight, 0, 0.1)
        # infer hidden size
        with torch.no_grad():
            d = torch.zeros(1,1,7,7, device=device)
            out = self.conv(d)
            self.N_exc = int(np.prod(out.shape[1:]))

        # LIF params
        self.v_thr, self.v_reset = 1.0, 0.0
        self.tau_m, self.tau_ref = tau_m, tau_ref
        self.alpha_v = np.exp(-1/self.tau_m)

        # state & traces
        self.reset_state()
        self.trace_pre  = torch.zeros(self.N_exc, device=device)
        self.trace_post = torch.zeros(self.N_exc, device=device)
        self.e_trace    = torch.zeros(self.N_exc, 10, device=device)

        # output weights
        self.W_out = torch.randn(self.N_exc, 10, device=device) * 0.1

        # snapshot interval
        self.snapshot_every = snapshot_every

    def reset_state(self):
        self.v   = torch.zeros(self.N_exc, device=device)
        self.ref = torch.zeros(self.N_exc, dtype=torch.int32, device=device)

    def simulate_step(self, inp_spikes):
        # conv→hidden drive
        maps = inp_spikes.view(1,1,7,7)
        o = self.conv(maps)            # (1, maps, H, W)
        I = o.view(-1)                 # (N_exc,)

        # membrane update + refractoriness
        self.v = self.alpha_v * self.v + I
        self.v[self.ref > 0] = self.v_reset

        # spike generation
        s = (self.v >= self.v_thr) & (self.ref == 0)
        self.v[s] = self.v_reset
        self.ref[s] = int(self.tau_ref)
        self.ref[self.ref > 0] -= 1

        return s.float()

    def stdp_update(self, pre, post):
        # placeholder STDP
        self.trace_pre  = self.trace_pre * self.alpha_trace + pre
        self.trace_post = self.trace_post * self.alpha_trace + post
        # (conv weights update omitted)

    def update_eligibility(self, hidden_spk, out_spk):
        self.e_trace = self.alpha_e * self.e_trace + hidden_spk.unsqueeze(1) * out_spk.unsqueeze(0)

    def apply_rstdp(self, reward):
        self.W_out += self.eta_rstdp * reward * self.e_trace

    def train(self, ds, N=10000, T=250, rate=100):
        acc_log = []
        weight_snaps = []
        rf_snaps     = []

        for i in range(N):
            img, lbl = ds[i]
            spikes, _ = encode_image_to_spikes(img.to(device), T, rate)
            self.reset_state()
            self.trace_pre.zero_(); self.trace_post.zero_(); self.e_trace.zero_()

            total = torch.zeros(self.N_exc, device=device)
            for t in range(T):
                s = self.simulate_step(spikes[:,t])
                self.stdp_update(s, s)
                # eligibility
                out_current = self.W_out.t() @ s
                out_spk     = (out_current >= self.v_thr).float()
                self.update_eligibility(s, out_spk)
                total += s

            # R‑STDP
            votes = (total.unsqueeze(1) * self.W_out).sum(0)
            pred  = votes.argmax().item()
            r     = 1.0 if pred == lbl else -1.0
            self.apply_rstdp(r)

            # log every 10
            if (i+1) % 10 == 0:
                print(f"[Train] processed {i+1}/{N}")

            # snapshot every snapshot_every
            if (i+1) % self.snapshot_every == 0:
                weight_snaps.append(self.conv.weight.detach().cpu().numpy().copy())
                rf = self.conv.weight.detach().cpu().clone()
                rf_snaps.append(rf[:9])
                acc = self.evaluate(ds, N=500, T=T, rate=rate, record=False)
                acc_log.append((i+1, acc))
                print(f"[Train] snapshot @ {i+1}: {acc:.2f}%")

        # plot weight histograms
        if weight_snaps:
            fig, axs = plt.subplots(1, len(weight_snaps),
                                    figsize=(4*len(weight_snaps), 3))
            for idx, w in enumerate(weight_snaps):
                axs[idx].hist(w.flatten(), bins=30)
                axs[idx].set_title(f"Step {(idx+1)*self.snapshot_every}")
            plt.suptitle("Conv Weight Distributions")
            plt.show()
        else:
            print("No weight snapshots to plot.")

        # plot receptive fields
        if rf_snaps:
            for idx, rf in enumerate(rf_snaps):
                fig, axes = plt.subplots(3, 3, figsize=(6,6))
                for j in range(9):
                    r, c = divmod(j, 3)
                    axes[r, c].imshow(rf[j, 0], cmap='gray')
                    axes[r, c].axis('off')
                plt.suptitle(f"Receptive Fields @ Step {(idx+1)*self.snapshot_every}")
                plt.show()
        else:
            print("No receptive-field snapshots to plot.")

        # accuracy vs. training samples
        if acc_log:
            xs, ys = zip(*acc_log)
            plt.figure(figsize=(6,4))
            plt.plot(xs, ys, '-o')
            plt.title("Accuracy vs. Training Samples")
            plt.xlabel("Training Steps")
            plt.ylabel("Accuracy (%)")
            plt.grid(True)
            plt.show()
        else:
            print("No accuracy snapshots to plot.")

    def evaluate(self, ds, N=1000, T=250, rate=100, record=True):
        correct = 0
        true, pred = [], []
        for i in range(N):
            img, lbl = ds[i]
            spikes, _ = encode_image_to_spikes(img.to(device), T, rate)
            self.reset_state()
            total = torch.zeros(self.N_exc, device=device)
            for t in range(T):
                s = self.simulate_step(spikes[:,t])
                total += s
            votes = (total.unsqueeze(1) * self.W_out).sum(0)
            p = votes.argmax().item()
            true.append(lbl); pred.append(p)
            if p == lbl:
                correct += 1
            if (i+1) % 100 == 0:
                print(f"[Test] accuracy: {correct/(i+1)*100:.2f}%")

        final = correct / N * 100
        print(f"[Test] final accuracy: {final:.2f}%")

        if record:
            cm = confusion_matrix(true, pred)
            disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
            disp.plot(cmap="Blues", xticks_rotation=45)
            ax = disp.ax_
            num = len(disp.display_labels)
            ax.set_xticks(np.arange(num))
            ax.set_yticks(np.arange(num))
            plt.title("Confusion Matrix")
            plt.tight_layout()
            plt.show()

        return final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples",     type=int, default=100)
    parser.add_argument("--test-samples",      type=int, default=20)
    parser.add_argument("--snapshot-interval", type=int, default=1000)
    args = parser.parse_args()

    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=tf)
    test_ds  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)

    snn = SpikingNetwork(snapshot_every=args.snapshot_interval)
    snn.train(train_ds, N=args.train_samples)
    snn.evaluate(test_ds, N=args.test_samples)

if __name__=="__main__":
    main()
