#!/usr/bin/env python3
"""
retina_snn_mnist_full.py

Biologically Inspired Spiking Neural Network for MNIST Classification
with:
 - Retina-like encoding (28×28 → 7×7 Poisson spikes)
 - Leaky Integrate-and-Fire intermediate layer (excitatory & inhibitory)
 - Unsupervised STDP learning
 - Output‑layer LIF potentials plotted per class (10‑panel time series)
 - Confusion matrix and final accuracy
 - Model save/load for resuming training or standalone evaluation

Requires Python 3.9 and:
    pip install torch torchvision matplotlib numpy scikit-learn
"""

import os
import argparse
import torch
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def encode_image_to_spikes(image, T=100, max_rate=100):
    x = image.unsqueeze(0)             # (1,1,28,28)
    x = F.avg_pool2d(x, 2, 2)          # (1,1,14,14)
    x = F.avg_pool2d(x, 2, 2)          # (1,1,7,7)
    intensities = x.squeeze()          # (7,7)
    p = intensities.flatten() * (max_rate / 1000.0)
    p = torch.clamp(p, 0, 1)
    p = p.view(-1, 1).expand(-1, T)    # (49, T)
    spikes = (torch.rand(p.shape, device=image.device) < p).float()
    return spikes, intensities


class SpikingNetwork:
    def __init__(self,
                 N_in=49, N_exc=100, N_inh=25, N_out=10,
                 tau_m_exc=20.0, tau_ref_exc=2,
                 tau_m_inh=10.0, tau_ref_inh=5,
                 tau_out=20.0,
                 A_plus=0.005, A_minus=0.005, w_max=1.0):
        # dimensions
        self.N_in, self.N_exc, self.N_inh, self.N_out = N_in, N_exc, N_inh, N_out

        # neuron & STDP params
        self.tau_m_exc, self.tau_ref_exc = tau_m_exc, tau_ref_exc
        self.tau_m_inh, self.tau_ref_inh = tau_m_inh, tau_ref_inh
        self.v_thr, self.v_reset = 1.0, 0.0
        self.alpha_exc = np.exp(-1.0/self.tau_m_exc)
        self.alpha_inh = np.exp(-1.0/self.tau_m_inh)

        self.A_plus, self.A_minus, self.w_max = A_plus, A_minus, w_max
        self.tau_trace = 20.0
        self.alpha_trace = np.exp(-1.0/self.tau_trace)

        # output layer leak
        self.tau_out = tau_out
        self.alpha_out = np.exp(-1.0/self.tau_out)

        # initialize weights
        scale = 0.2
        self.W_in2exc = torch.randn(N_in, N_exc, device=device) * scale
        self.W_in2inh = torch.randn(N_in, N_inh, device=device) * scale
        self.W_exc2inh = torch.randn(N_exc, N_inh, device=device) * scale
        self.W_inh2exc = torch.randn(N_inh, N_exc, device=device) * (-scale)

        # state vars
        self.reset_state()
        self.trace_pre = torch.zeros(N_in, device=device)
        self.trace_post = torch.zeros(N_exc, device=device)

    def reset_state(self):
        self.v_exc   = torch.zeros(self.N_exc, device=device)
        self.v_inh   = torch.zeros(self.N_inh, device=device)
        self.ref_exc = torch.zeros(self.N_exc, dtype=torch.int32, device=device)
        self.ref_inh = torch.zeros(self.N_inh, dtype=torch.int32, device=device)

    def simulate_step(self, input_spikes):
        I_exc = torch.mv(self.W_in2exc.t(), input_spikes)
        I_inh = torch.mv(self.W_in2inh.t(), input_spikes)
        self.v_exc = self.alpha_exc*self.v_exc + I_exc
        self.v_inh = self.alpha_inh*self.v_inh + I_inh
        self.v_exc[self.ref_exc>0] = self.v_reset
        self.v_inh[self.ref_inh>0] = self.v_reset
        spikes_exc = (self.v_exc>=self.v_thr)&(self.ref_exc==0)
        spikes_inh = (self.v_inh>=self.v_thr)&(self.ref_inh==0)
        self.v_exc[spikes_exc] = self.v_reset
        self.v_inh[spikes_inh] = self.v_reset
        self.ref_exc[spikes_exc] = self.tau_ref_exc
        self.ref_inh[spikes_inh] = self.tau_ref_inh
        self.ref_exc[self.ref_exc>0] -= 1
        self.ref_inh[self.ref_inh>0] -= 1
        return spikes_exc.float(), spikes_inh.float()

    def stdp_update(self, pre, post):
        self.trace_pre  = self.trace_pre  * self.alpha_trace + pre
        self.trace_post = self.trace_post * self.alpha_trace + post
        for j in range(self.N_exc):
            if post[j]>0:
                dw = self.A_plus * self.trace_pre * (self.w_max - self.W_in2exc[:, j])
                self.W_in2exc[:, j] += dw
        for i in range(self.N_in):
            if pre[i]>0:
                dw = self.A_minus * self.trace_post * self.W_in2exc[i, :]
                self.W_in2exc[i, :] -= dw
        self.W_in2exc = torch.clamp(self.W_in2exc, 0.0, self.w_max)

    def save_model(self, path, neuron_class):
        data = {
            'W_in2exc': self.W_in2exc.cpu(),
            'W_in2inh': self.W_in2inh.cpu(),
            'W_exc2inh': self.W_exc2inh.cpu(),
            'W_inh2exc': self.W_inh2exc.cpu(),
            'neuron_class': neuron_class.cpu()
        }
        torch.save(data, path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        data = torch.load(path, map_location=device)
        self.W_in2exc = data['W_in2exc'].to(device)
        self.W_in2inh = data['W_in2inh'].to(device)
        self.W_exc2inh = data['W_exc2inh'].to(device)
        self.W_inh2exc = data['W_inh2exc'].to(device)
        neuron_class = data.get('neuron_class', None)
        print(f"Model loaded from {path}")
        return neuron_class

    def train(self, train_ds, test_ds,
              num_samples=10000, T=100, max_rate=100,
              log_interval=2000):
        spikes_per_class = torch.zeros(self.N_out, self.N_exc, device=device)
        weight_snaps = [self.W_in2exc.cpu().flatten().numpy()]
        acc_log = []

        for idx in range(num_samples):
            img, label = train_ds[idx]
            img = img.to(device)
            spike_train, _ = encode_image_to_spikes(img, T, max_rate)
            self.reset_state()
            self.trace_pre.zero_(); self.trace_post.zero_()

            for t in range(T):
                pre = spike_train[:, t]
                s_exc, _ = self.simulate_step(pre)
                self.stdp_update(pre, s_exc)
                spikes_per_class[label] += s_exc

            if (idx + 1) % 10 == 0:
                print(f"[Train] processed {idx+1}/{num_samples}")

            if (idx + 1) % log_interval == 0:
                weight_snaps.append(self.W_in2exc.cpu().flatten().numpy())
                neuron_class = torch.argmax(spikes_per_class, dim=0)
                acc = self.evaluate(test_ds, neuron_class,
                                    num_samples=500, T=T, max_rate=max_rate,
                                    record_outputs=False)
                acc_log.append((idx+1, acc))
                print(f"[Train] eval @ {idx+1}: {acc:.2f}%")

        weight_snaps.append(self.W_in2exc.cpu().flatten().numpy())
        neuron_class = torch.argmax(spikes_per_class, dim=0)

        # Plot weight distributions (Fig.2)
        fig, axs = plt.subplots(1, len(weight_snaps), figsize=(5*len(weight_snaps),4))
        for i, w in enumerate(weight_snaps):
            axs[i].hist(w, bins=50, range=(0,self.w_max))
            title = "Init" if i==0 else ("Final" if i==len(weight_snaps)-1 else f"After {(i)*log_interval}")
            axs[i].set_title(title)
            axs[i].set_xlabel("w"); axs[i].set_ylabel("count")
        plt.tight_layout(); plt.show()

        # Plot receptive fields (Fig.3)
        fig, axes = plt.subplots(3,3, figsize=(9,9))
        for j in range(9):
            r,c = divmod(j,3)
            wf = self.W_in2exc[:,j].cpu().view(7,7)
            im = axes[r][c].imshow(wf, cmap='hot', vmin=0, vmax=self.w_max)
            axes[r][c].set_title(f"Neuron {j}→{neuron_class[j].item()}")
            axes[r][c].axis('off')
        fig.colorbar(im, ax=axes, fraction=0.02)
        plt.suptitle("Learned Receptive Fields"); plt.tight_layout(rect=[0,0,1,0.96]); plt.show()

        # Plot accuracy curve (Fig.5)
        xs, ys = zip(*acc_log)
        plt.figure(figsize=(6,4))
        plt.plot(xs, ys, marker='o')
        plt.title("Accuracy vs. Training Samples")
        plt.xlabel("Samples"); plt.ylabel("Accuracy (%)"); plt.grid(True); plt.show()

        return neuron_class

    def evaluate(self, test_ds, neuron_class,
                 num_samples=1000, T=100, max_rate=100,
                 record_outputs=True):
        W_out = torch.zeros(self.N_exc, self.N_out, device=device)
        for i in range(self.N_exc):
            W_out[i, neuron_class[i]] = 1.0

        true_labels, pred_labels = [], []
        volt_history = None

        for idx in range(num_samples):
            img, label = test_ds[idx]
            img = img.to(device)
            spike_train, _ = encode_image_to_spikes(img, T, max_rate)
            self.reset_state()
            v_out = torch.zeros(self.N_out, device=device)
            if record_outputs and idx==0:
                volt_history = torch.zeros(T, self.N_out)

            total_spikes = torch.zeros(self.N_exc, device=device)
            for t in range(T):
                pre = spike_train[:, t]
                s_exc, _ = self.simulate_step(pre)
                total_spikes += s_exc
                v_out = self.alpha_out*v_out + W_out.t() @ s_exc
                if record_outputs and idx==0:
                    volt_history[t] = v_out.detach().cpu()

            votes = torch.zeros(self.N_out, device=device)
            for j in range(self.N_exc):
                votes[neuron_class[j]] += total_spikes[j]
            pred = int(votes.argmax().item())

            true_labels.append(label)
            pred_labels.append(pred)

        acc = 100*np.mean(np.array(true_labels)==np.array(pred_labels))
        print(f"[Test] accuracy over {num_samples} samples: {acc:.2f}%")

        if record_outputs:
            # Fig.4 voltage traces
            time = np.arange(T)
            fig, axes = plt.subplots(10, 1, figsize=(8, 12), sharex=True)
            for j in range(self.N_out):
                ax = axes[j]
                ax.plot(time, volt_history[:, j],
                        color='red' if j==true_labels[0] else 'blue')
                ax.set_ylabel(f"Class {j}")
                ax.set_xlim(0, T)
                ax.set_xticks([])
            axes[-1].set_xlabel("Time (ms)")
            plt.suptitle(f"Output Neuron Potentials (Sample={true_labels[0]})")
            plt.tight_layout(rect=[0,0,1,0.95]); plt.show()

            # confusion matrix
            cm = confusion_matrix(true_labels, pred_labels)
            disp = ConfusionMatrixDisplay(cm, display_labels=list(range(self.N_out)))
            disp.plot(cmap="Blues", xticks_rotation=45)
            plt.title("Confusion Matrix"); plt.tight_layout(); plt.show()

        return acc


def main():
    parser = argparse.ArgumentParser(description="Retina-SNN MNIST")
    parser.add_argument("--load-model",     type=str, default=None,
                        help="path to saved model checkpoint")
    parser.add_argument("--save-model",     type=str, default="snn_checkpoint.pth",
                        help="where to save the trained model")
    parser.add_argument("--eval-only",      action="store_true",
                        help="skip training and only evaluate loaded model")
    parser.add_argument("--train-samples",  type=int, default=10000)
    parser.add_argument("--test-samples",   type=int, default=1000)
    parser.add_argument("--T",              type=int, default=100)
    parser.add_argument("--rate",           type=int, default=100)
    parser.add_argument("--snapshot-interval", type=int, default=2000)
    args = parser.parse_args()

    # load MNIST
    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST(root="./data", train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)

    snn = SpikingNetwork()

    # Load checkpoint if provided
    neuron_class = None
    if args.load_model and os.path.isfile(args.load_model):
        neuron_class = snn.load_model(args.load_model)

    if args.eval_only:
        if neuron_class is None:
            raise ValueError("Must provide --load-model when using --eval-only")
        # Just evaluate
        snn.evaluate(test_ds, neuron_class,
                     num_samples=args.test_samples,
                     T=args.T, max_rate=args.rate,
                     record_outputs=True)
        return

    # Else: train (possibly resuming)
    neuron_class = snn.train(train_ds, test_ds,
                             num_samples=args.train_samples,
                             T=args.T, max_rate=args.rate,
                             log_interval=args.snapshot_interval)

    # Save checkpoint
    snn.save_model(args.save_model, neuron_class)

    # Final eval
    snn.evaluate(test_ds, neuron_class,
                 num_samples=args.test_samples,
                 T=args.T, max_rate=args.rate,
                 record_outputs=True)


if __name__ == "__main__":
    main()
