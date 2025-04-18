#!/usr/bin/env python3
"""
retina_snn_mnist_rstdp_fixed.py

Biologically Inspired Spiking Neural Network for MNIST Classification
with Reward‑Modulated STDP (R‑STDP) at the output, plus logging:

 - Prints [Train] processed i/N every 10 training images
 - Prints [Train] eval @ i: XX.XX% every snapshot
 - Prints [Test] accuracy: XX.XX% every 100 test samples and final accuracy
 - Confusion matrix at the end

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
    """
    Encode a 28×28 MNIST image into:
      - spikes:     (49, T) Poisson spike tensor
      - intensities: 7×7 average‑pooled map (unused here)
    """
    x = image.unsqueeze(0)                                  # (1,1,28,28)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)            # (1,1,14,14)
    x = F.avg_pool2d(x, kernel_size=2, stride=2)            # (1,1,7,7)
    intensities = x.squeeze()                               # (7,7)
    p = torch.clamp(intensities.flatten() * (max_rate/1000), 0.0, 1.0)  # (49,)
    p = p.view(-1,1).expand(-1, T)                          # (49, T)
    spikes = (torch.rand(p.shape, device=image.device) < p).float()
    return spikes, intensities

class SpikingNetwork:
    def __init__(self,
                 N_in=49, N_exc=100, N_inh=25, N_out=10,
                 tau_m_exc=20.0, tau_ref_exc=2,
                 tau_m_inh=10.0, tau_ref_inh=5,
                 tau_out=20.0,
                 A_plus=0.005, A_minus=0.005, w_max=1.0,
                 eta_rstdp=0.01, alpha_e=0.95):
        # dimensions
        self.N_in, self.N_exc, self.N_inh, self.N_out = N_in, N_exc, N_inh, N_out

        # hidden-layer LIF params
        self.tau_m_exc, self.tau_ref_exc = tau_m_exc, tau_ref_exc
        self.tau_m_inh, self.tau_ref_inh = tau_m_inh, tau_ref_inh
        self.v_thr, self.v_reset = 1.0, 0.0
        self.alpha_exc = np.exp(-1.0/self.tau_m_exc)
        self.alpha_inh = np.exp(-1.0/self.tau_m_inh)

        # hidden-layer STDP
        self.A_plus, self.A_minus, self.w_max = A_plus, A_minus, w_max
        self.tau_trace = 20.0
        self.alpha_trace = np.exp(-1.0/self.tau_trace)

        # output-layer leak & R-STDP params
        self.tau_out = tau_out
        self.alpha_out = np.exp(-1.0/self.tau_out)
        self.eta_rstdp = eta_rstdp
        self.alpha_e = alpha_e
        self.e_trace = torch.zeros(N_exc, N_out, device=device)

        # initialize weights
        scale = 0.2
        self.W_in2exc = torch.randn(N_in, N_exc, device=device) * scale
        self.W_in2inh = torch.randn(N_in, N_inh, device=device) * scale
        self.W_exc2inh = torch.randn(N_exc, N_inh, device=device) * scale
        self.W_inh2exc = torch.randn(N_inh, N_exc, device=device) * (-scale)
        self.W_exc2out = torch.randn(N_exc, N_out, device=device) * 0.1

        # reset state and traces
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
        self.v_exc = self.alpha_exc * self.v_exc + I_exc
        self.v_inh = self.alpha_inh * self.v_inh + I_inh
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
        self.trace_pre  = self.trace_pre * self.alpha_trace + pre
        self.trace_post = self.trace_post * self.alpha_trace + post
        # LTP
        for j in range(self.N_exc):
            if post[j] > 0:
                dw = self.A_plus * self.trace_pre * (self.w_max - self.W_in2exc[:, j])
                self.W_in2exc[:, j] += dw
        # LTD
        for i in range(self.N_in):
            if pre[i] > 0:
                dw = self.A_minus * self.trace_post * self.W_in2exc[i, :]
                self.W_in2exc[i, :] -= dw
        torch.clamp_(self.W_in2exc, 0.0, self.w_max)

    def update_eligibility(self, hidden_spikes, output_spikes):
        self.e_trace *= self.alpha_e
        self.e_trace += hidden_spikes.unsqueeze(1) * output_spikes.unsqueeze(0)

    def apply_rstdp(self, reward):
        self.W_exc2out += self.eta_rstdp * reward * self.e_trace

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
            self.e_trace.zero_()

            # simulate hidden + compute eligibility
            total_spikes = torch.zeros(self.N_exc, device=device)
            for t in range(T):
                pre = spike_train[:, t]
                s_exc, _ = self.simulate_step(pre)
                self.stdp_update(pre, s_exc)
                out_current = self.W_exc2out.t() @ s_exc
                out_spike = (out_current >= self.v_thr).float()
                self.update_eligibility(s_exc, out_spike)
                total_spikes += s_exc
                spikes_per_class[label] += s_exc

            # supervised R‑STDP update
            votes = (total_spikes.unsqueeze(1) * self.W_exc2out).sum(dim=0)
            pred = votes.argmax().item()
            reward = 1.0 if pred == label else -1.0
            self.apply_rstdp(reward)

            if (idx+1) % 10 == 0:
                print(f"[Train] processed {idx+1}/{num_samples}")

            if (idx+1) % log_interval == 0:
                neuron_class = torch.argmax(spikes_per_class, dim=0)
                acc = self.evaluate(test_ds, neuron_class,
                                    num_samples=500, T=T, max_rate=max_rate,
                                    record_outputs=False)
                acc_log.append((idx+1, acc))
                print(f"[Train] eval @ {idx+1}: {acc:.2f}%")

        weight_snaps.append(self.W_in2exc.cpu().flatten().numpy())
        neuron_class = torch.argmax(spikes_per_class, dim=0)
        return neuron_class

    def evaluate(self, test_ds, neuron_class,
                 num_samples=1000, T=100, max_rate=100,
                 record_outputs=True):
        W_out = torch.zeros(self.N_exc, self.N_out, device=device)
        for i in range(self.N_exc):
            W_out[i, neuron_class[i]] = 1.0

        true_labels, pred_labels = [], []
        correct = 0

        for idx in range(num_samples):
            img, label = test_ds[idx]
            img = img.to(device)
            spike_train, _ = encode_image_to_spikes(img, T, max_rate)
            self.reset_state()

            total_spikes = torch.zeros(self.N_exc, device=device)
            v_out = torch.zeros(self.N_out, device=device)
            for t in range(T):
                pre = spike_train[:, t]
                s_exc, _ = self.simulate_step(pre)
                total_spikes += s_exc
                v_out = self.alpha_out * v_out + W_out.t() @ s_exc

            votes = (total_spikes.unsqueeze(1) * self.W_exc2out).sum(dim=0)
            pred = votes.argmax().item()

            true_labels.append(label)
            pred_labels.append(pred)
            if pred == label:
                correct += 1

            if (idx+1) % 100 == 0:
                acc_so_far = correct / (idx+1) * 100.0
                print(f"[Test] accuracy: {acc_so_far:.2f}%")

        final_acc = correct / num_samples * 100.0
        print(f"[Test] final accuracy: {final_acc:.2f}%")

        if record_outputs:
            cm = confusion_matrix(true_labels, pred_labels)
            disp = ConfusionMatrixDisplay(cm, display_labels=list(range(self.N_out)))
            disp.plot(cmap="Blues")
            plt.title("Confusion Matrix")
            plt.show()

        return final_acc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--test-samples",  type=int, default=1000)
    parser.add_argument("--T",             type=int, default=100)
    parser.add_argument("--rate",          type=int, default=100)
    parser.add_argument("--snapshot-interval", type=int, default=2000)
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=transform)
    test_ds  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=transform)

    snn = SpikingNetwork()
    neuron_class = snn.train(train_ds, test_ds,
                              num_samples=args.train_samples,
                              T=args.T, max_rate=args.rate,
                              log_interval=args.snapshot_interval)
    snn.evaluate(test_ds, neuron_class,
                 num_samples=args.test_samples,
                 T=args.T, max_rate=args.rate)

if __name__ == "__main__":
    main()
