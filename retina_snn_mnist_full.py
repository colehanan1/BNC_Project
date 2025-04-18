#!/usr/bin/env python3
"""
retina_snn_mnist_conv_rstdp.py

Biologically Inspired Spiking Neural Network for MNIST Classification
with:
 - Retina-inspired encoding (28×28 → 7×7 Poisson spikes)
 - Convolutional STDP input layer (5×5 kernels → feature maps)
 - Hidden-layer LIF neurons trained with unsupervised STDP
 - Reward‑modulated STDP (R‑STDP) supervised output layer
 - Frequent logging of training progress
 - Intermediate figures of weight histograms, receptive fields, and accuracy curves

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

# Device selection
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

device = get_device()
print(f"Using device: {device}")

# Reproducibility
torch.manual_seed(0)
np.random.seed(0)
random.seed(0)


def encode_image_to_spikes(image, T=250, max_rate=100):
    """
    Encode a 28×28 image into a 7×7×T Poisson spike train via retina pooling.
    Returns spikes (49×T) and the pooled intensity map (7×7).
    """
    x = image.unsqueeze(0)                       # (1,1,28,28)
    x = F.avg_pool2d(x, 2, 2)                     # (1,1,14,14)
    x = F.avg_pool2d(x, 2, 2)                     # (1,1,7,7)
    intensities = x.squeeze()                     # (7,7)
    p = torch.clamp(intensities.flatten() * (max_rate/1000.0), 0.0, 1.0)
    p = p.view(-1,1).expand(-1, T)                # (49, T)
    spikes = (torch.rand(p.shape, device=image.device) < p).float()
    return spikes, intensities


class SpikingNetwork:
    def __init__(self,
                 N_in_maps=8, in_kernel=5,
                 N_exc=200, N_inh=50, N_out=10,
                 tau_m_exc=20.0, tau_ref_exc=2,
                 tau_m_inh=10.0, tau_ref_inh=5,
                 tau_out=20.0,
                 A_plus=0.02, A_minus=0.01, w_max=0.5,
                 eta_rstdp=0.01, alpha_e=0.95):
        # dimensions
        self.N_in_maps = N_in_maps      # number of conv feature maps
        self.in_kernel = in_kernel      # conv kernel size
        self.N_exc = N_exc
        self.N_inh = N_inh
        self.N_out = N_out

        # hidden LIF params
        self.tau_m_exc, self.tau_ref_exc = tau_m_exc, tau_ref_exc
        self.tau_m_inh, self.tau_ref_inh = tau_m_inh, tau_ref_inh
        self.v_thr, self.v_reset = 1.0, 0.0
        self.alpha_exc = np.exp(-1.0/self.tau_m_exc)
        self.alpha_inh = np.exp(-1.0/self.tau_m_inh)

        # STDP hidden
        self.A_plus, self.A_minus, self.w_max = A_plus, A_minus, w_max
        self.tau_trace = 50.0
        self.alpha_trace = np.exp(-1.0/self.tau_trace)

        # output R-STDP
        self.tau_out = tau_out
        self.alpha_out = np.exp(-1.0/self.tau_out)
        self.eta_rstdp = eta_rstdp
        self.alpha_e = alpha_e

        # conv input→hidden
        self.conv = torch.nn.Conv2d(1, N_in_maps, in_kernel, stride=2, bias=False).to(device)
        torch.nn.init.uniform_(self.conv.weight, a=0.0, b=0.1)

        # state and traces
        self.reset_state()
        self.trace_pre = torch.zeros(N_exc, device=device)
        self.trace_post = torch.zeros(N_exc, device=device)
        self.e_trace = torch.zeros(N_exc, N_out, device=device)

        # weights
        self.W_exc2inh = torch.randn(N_exc, N_inh, device=device)*0.1
        self.W_inh2exc = -torch.randn(N_inh, N_exc, device=device)*0.1
        self.W_exc2out = torch.randn(N_exc, N_out, device=device)*0.1

    def reset_state(self):
        self.v_exc   = torch.zeros(self.N_exc, device=device)
        self.v_inh   = torch.zeros(self.N_inh, device=device)
        self.ref_exc = torch.zeros(self.N_exc, dtype=torch.int32, device=device)
        self.ref_inh = torch.zeros(self.N_inh, dtype=torch.int32, device=device)

    def simulate_step(self, input_spikes):
        # input→hidden via conv+flatten
        spikes_maps = input_spikes.view(1,1,7,7)      # reconstruct 7×7
        maps_out = self.conv(spikes_maps)             # (1, N_in_maps, H, W)
        hid_in = maps_out.view(-1, self.N_exc)        # flatten to (N_exc,)
        # hidden dynamics
        I_exc = hid_in
        I_inh = torch.mv(self.W_exc2inh.t(), (self.v_exc >= self.v_thr).float())
        self.v_exc = self.alpha_exc*self.v_exc + I_exc - I_inh
        self.v_inh = self.alpha_inh*self.v_inh + torch.mv(self.W_exc2inh.t(), (self.v_exc>=self.v_thr).float())
        # apply refractory
        self.v_exc[self.ref_exc>0]=self.v_reset
        self.v_inh[self.ref_inh>0]=self.v_reset
        # generate spikes
        s_exc = (self.v_exc>=self.v_thr)&(self.ref_exc==0)
        s_inh = (self.v_inh>=self.v_thr)&(self.ref_inh==0)
        # reset & refract
        self.v_exc[s_exc]=self.v_reset; self.ref_exc[s_exc]=self.tau_ref_exc
        self.v_inh[s_inh]=self.v_reset; self.ref_inh[s_inh]=self.tau_ref_inh
        self.ref_exc[self.ref_exc>0]-=1; self.ref_inh[self.ref_inh>0]-=1
        return s_exc.float(), s_inh.float()

    def stdp_update(self, pre, post):
        self.trace_pre  = self.trace_pre*self.alpha_trace + pre
        self.trace_post = self.trace_post*self.alpha_trace + post
        # hidden STDP on conv weights
        for k in range(self.N_exc):
            if post[k]>0:
                grad = self.A_plus * self.trace_pre[k]
                self.conv.weight.data[k//((7-(self.in_kernel-1))**2),
                    :, :, :] += grad * self.conv.weight.data[k//((7-(self.in_kernel-1))**2)]
        torch.clamp_(self.conv.weight.data, 0.0, 0.5)

    def update_eligibility(self, hidden_spikes, output_spikes):
        self.e_trace *= self.alpha_e
        self.e_trace += hidden_spikes.unsqueeze(1)*output_spikes.unsqueeze(0)

    def apply_rstdp(self, reward):
        self.W_exc2out += self.eta_rstdp*reward*self.e_trace

    def train(self, train_ds, num_samples=20000, T=250, rate=100,
              log_every=10, snapshot_every=1000):
        acc_log, weight_snaps = [], []
        for idx in range(num_samples):
            img, label = train_ds[idx]
            spikes, _ = encode_image_to_spikes(img.to(device), T, rate)
            self.reset_state()
            self.trace_pre.zero_(); self.trace_post.zero_(); self.e_trace.zero_()

            total_spikes = torch.zeros(self.N_exc, device=device)
            for t in range(T):
                pre = spikes[:,t]
                s_exc, s_inh = self.simulate_step(pre)
                self.stdp_update(pre.repeat(self.N_exc), s_exc)  # pseudo pre
                out_current = self.W_exc2out.t()@s_exc
                out_spike = (out_current>=self.v_thr).float()
                self.update_eligibility(s_exc, out_spike)
                total_spikes += s_exc

            votes = (total_spikes.unsqueeze(1)*self.W_exc2out).sum(dim=0)
            pred = votes.argmax().item()
            reward = 1.0 if pred==label else -1.0
            self.apply_rstdp(reward)

            if (idx+1)%log_every==0:
                print(f"[Train] processed {idx+1}/{num_samples}")

            if (idx+1)%snapshot_every==0:
                weight_snaps.append(self.conv.weight.data.cpu().numpy().copy())
                acc = self.evaluate(train_ds, torch.arange(self.N_out),
                                    num_samples=500, T=T, rate=rate, record=False)
                acc_log.append((idx+1,acc))
                print(f"[Train] snapshot @ {idx+1}: {acc:.2f}%")

        # plot weight snapshots
        fig, axs = plt.subplots(1,len(weight_snaps), figsize=(4*len(weight_snaps),4))
        for i,w in enumerate(weight_snaps):
            axs[i].hist(w.flatten(), bins=30)
            axs[i].set_title(f"Snap {i}")
        plt.show()

        # accuracy curve
        xs, ys = zip(*acc_log)
        plt.plot(xs,ys,'-o')
        plt.title("Training Accuracy over time")
        plt.show()

    def evaluate(self, ds, neuron_class, num_samples=1000, T=250, rate=100, record=True):
        correct=0; true, pred = [],[]
        for idx in range(num_samples):
            img, label = ds[idx]
            spikes,_ = encode_image_to_spikes(img.to(device), T, rate)
            self.reset_state()
            total_spikes = torch.zeros(self.N_exc, device=device)
            for t in range(T):
                s_exc,_=self.simulate_step(spikes[:,t])
                total_spikes+=s_exc
            votes=(total_spikes.unsqueeze(1)*self.W_exc2out).sum(dim=0)
            p=votes.argmax().item()
            true.append(label); pred.append(p)
            if p==label: correct+=1
            if (idx+1)%100==0:
                print(f"[Test] accuracy: {correct/(idx+1)*100:.2f}%")
        final=correct/num_samples*100
        print(f"[Test] final accuracy: {final:.2f}%")
        if record:
            cm=confusion_matrix(true,pred)
            disp=ConfusionMatrixDisplay(cm, display_labels=list(range(self.N_out)))
            disp.plot(cmap="Blues")
            plt.show()
        return final


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=20000)
    parser.add_argument("--test-samples",  type=int, default=2000)
    args=parser.parse_args()

    tf=transforms.Compose([transforms.ToTensor()])
    train_ds=torchvision.datasets.MNIST("./data",train=True,download=True,transform=tf)
    test_ds =torchvision.datasets.MNIST("./data",train=False,download=True,transform=tf)

    snn=SpikingNetwork()
    snn.train(train_ds, num_samples=args.train_samples)
    snn.evaluate(test_ds, neuron_class=torch.arange(snn.N_out), num_samples=args.test_samples)

if __name__=="__main__":
    main()
