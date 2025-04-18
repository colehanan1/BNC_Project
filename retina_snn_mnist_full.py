#!/usr/bin/env python3
"""
retina_snn_mnist_hybrid.py

Biologically inspired SNN + Logistic Regression readout for MNIST:
 - Retina encoding (28×28 → 7×7 Poisson spikes)
 - Hidden-layer LIF neurons trained with unsupervised STDP
 - Aggregate hidden spikes per image → train sklearn LogisticRegression
 - Confusion matrix + final accuracy reporting

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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

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

def encode_image_to_spikes(image, T=100, max_rate=100):
    """Convert 28×28 image → 7×7 Poisson spike train of shape (49, T)."""
    x = image.unsqueeze(0)
    x = F.avg_pool2d(x, 2, 2)
    x = F.avg_pool2d(x, 2, 2)
    intensities = x.squeeze()
    p = torch.clamp(intensities.flatten() * (max_rate/1000.0), 0, 1)
    p = p.view(-1,1).expand(-1, T)
    return (torch.rand(p.shape, device=image.device) < p).float()

class SpikingNetwork:
    def __init__(self, N_in=49, N_exc=200, N_inh=50,
                 tau_m_exc=20., tau_ref_exc=2,
                 tau_m_inh=10., tau_ref_inh=5.,
                 A_plus=0.02, A_minus=0.01, w_max=1.0):
        # Dimensions
        self.N_in, self.N_exc, self.N_inh = N_in, N_exc, N_inh
        # LIF parameters
        self.tau_m_exc, self.tau_ref_exc = tau_m_exc, tau_ref_exc
        self.tau_m_inh, self.tau_ref_inh = tau_m_inh, tau_ref_inh
        self.v_thr, self.v_reset = 1.0, 0.0
        self.alpha_exc = np.exp(-1.0/tau_m_exc)
        self.alpha_inh = np.exp(-1.0/tau_m_inh)
        # STDP parameters (tuned for better feature learning) :contentReference[oaicite:0]{index=0}
        self.A_plus, self.A_minus, self.w_max = A_plus, A_minus, w_max
        self.tau_trace = 20.
        self.alpha_trace = np.exp(-1.0/self.tau_trace)
        # Initialize weights
        scale = 0.1
        self.W_in2exc = torch.rand(N_in, N_exc, device=device) * scale
        self.W_in2inh = torch.rand(N_in, N_inh, device=device) * scale
        self.W_exc2inh = torch.rand(N_exc, N_inh, device=device) * scale
        self.W_inh2exc = -torch.rand(N_inh, N_exc, device=device) * scale
        # State and traces
        self.reset_state()
        self.trace_pre = torch.zeros(N_in, device=device)
        self.trace_post = torch.zeros(N_exc, device=device)

    def reset_state(self):
        self.v_exc   = torch.zeros(self.N_exc, device=device)
        self.v_inh   = torch.zeros(self.N_inh, device=device)
        self.ref_exc = torch.zeros(self.N_exc, dtype=torch.int32, device=device)
        self.ref_inh = torch.zeros(self.N_inh, dtype=torch.int32, device=device)

    def simulate_step(self, inp):
        I_exc = torch.mv(self.W_in2exc.t(), inp)
        I_inh = torch.mv(self.W_in2inh.t(), inp)
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
        # Update traces
        self.trace_pre  = self.trace_pre*self.alpha_trace + pre
        self.trace_post = self.trace_post*self.alpha_trace + post
        # LTP
        for j in range(self.N_exc):
            if post[j]>0:
                dw = self.A_plus*self.trace_pre*(self.w_max-self.W_in2exc[:,j])
                self.W_in2exc[:,j] += dw
        # LTD
        for i in range(self.N_in):
            if pre[i]>0:
                dw = self.A_minus*self.trace_post*self.W_in2exc[i,:]
                self.W_in2exc[i,:] -= dw
        torch.clamp_(self.W_in2exc, 0.0, self.w_max)

    def extract_features(self, dataset, num_samples, T, rate):
        X = np.zeros((num_samples, self.N_exc), dtype=np.float32)
        y = np.zeros((num_samples,), dtype=np.int32)
        for idx in range(num_samples):
            img, label = dataset[idx]
            spikes = encode_image_to_spikes(img.to(device), T, rate)
            self.reset_state()
            total_spikes = torch.zeros(self.N_exc, device=device)
            for t in range(T):
                s_exc, _ = self.simulate_step(spikes[:,t])
                self.stdp_update(spikes[:,t], s_exc)
                total_spikes += s_exc
            X[idx] = total_spikes.cpu().numpy()
            y[idx] = label
            if (idx+1)%1000==0:
                print(f"Extracted features {idx+1}/{num_samples}")
        return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-samples", type=int, default=10000)
    parser.add_argument("--test-samples",  type=int, default=2000)
    parser.add_argument("--T",             type=int, default=200)
    parser.add_argument("--rate",          type=int, default=100)
    args = parser.parse_args()

    # Load MNIST
    tf = transforms.Compose([transforms.ToTensor()])
    train_ds = torchvision.datasets.MNIST("./data", train=True,  download=True, transform=tf)
    test_ds  = torchvision.datasets.MNIST("./data", train=False, download=True, transform=tf)

    # Build and train hidden-layer SNN with STDP
    snn = SpikingNetwork()
    print("Extracting train features...")
    X_train, y_train = snn.extract_features(train_ds, args.train_samples, args.T, args.rate)
    print("Extracting test features...")
    X_test,  y_test  = snn.extract_features(test_ds,  args.test_samples,  args.T, args.rate)

    # Train logistic regression on spike features :contentReference[oaicite:1]{index=1}
    clf = LogisticRegression(solver='lbfgs', multi_class='multinomial', max_iter=500)
    clf.fit(X_train, y_train)  # learns to discriminate spiking patterns :contentReference[oaicite:2]{index=2}

    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Final Test Accuracy: {acc*100:.2f}%")  # should rise above chance as train_samples ↑ :contentReference[oaicite:3]{index=3}

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=list(range(10)))
    disp.plot(cmap='Blues')
    plt.title("Confusion Matrix")
    plt.show()

if __name__ == "__main__":
    main()
