#!/usr/bin/env python3
"""
Extended GPU-accelerated Spiking Neural Network Training with PSAC for MNIST

This script implements:
  - A Retina module that converts MNIST images into spike trains.
  - A GPU‐accelerated spiking network with a middle (liquid) layer,
    an output layer, and an actor–critic module.
  - A proper training loop over MNIST using a session‐scoped pytest fixture.
  - An evaluation routine with detailed debugging output.
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytest

# Detect Apple M2 GPU (MPS) or fallback to CPU
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -------------------- Configuration --------------------
CONFIG = {
    "simulation": {
        "dt": 1.0,
        "t_total": 1200,
        "t_skip": 200,
    },
    "neuron": {
        "v_rest": 0.0,
        "v_thresh": 18.0,
        "v_reset": 0.0,
    },
    "middle_layer": {
        "num_neurons": 5000,
        "ratio_exc": 0.8,
        "tau_m_exc": 20.0,
        "tau_m_inh": 10.0,
        "refractory_exc": 2.0,
        "refractory_inh": 1.0,
    },
    "output_layer": {
        "num_neurons": 10,
        "tau_m": 20.0,
        "refractory": 2.0,
    },
    "network": {
        "connection_prob": 0.2,
    },
    "actor_critic": {
        "num_critic_neurons": 20,
        "gamma": 0.99,
        "tau_r": 20.0,
    },
    "learning": {
        "learning_rate": 0.001,
    },
    "retina": {
        "input_size": (28, 28),
        "pool_size": 2,
        "stride": 2,
        "spike_rate_scaling": 50.0,
    },
    "training": {
        "num_epochs": 2,
        "batch_size": 1,
    }
}

# -------------------- Retina Module --------------------
class Retina:
    def __init__(self, config=CONFIG):
        self.config = config
        self.input_size = config["retina"]["input_size"]
        self.pool_size = config["retina"]["pool_size"]
        self.stride = config["retina"]["stride"]
        self.output_dim = (self.input_size[0] // self.stride,
                           self.input_size[1] // self.stride)

    def process_image(self, image):
        h, w = self.input_size
        pool_h, pool_w = self.pool_size, self.pool_size
        out_h, out_w = self.output_dim
        pooled = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                window = image[i*self.stride:i*self.stride+pool_h,
                               j*self.stride:j*self.stride+pool_w]
                pooled[i, j] = np.mean(window)
        return pooled / 255.0

    def generate_spike_train(self, rates, sim_time, dt):
        num_units = rates.size
        num_steps = int(sim_time / dt)
        train = np.zeros((num_units, num_steps), dtype=np.float32)
        scale = self.config["retina"]["spike_rate_scaling"]
        for u in range(num_units):
            rate = rates.flatten()[u]
            probs = np.random.rand(num_steps)
            train[u] = (probs < (rate * scale * dt / 1000.0)).astype(np.float32)
        return torch.tensor(train, device=device, dtype=torch.float32)

# -------------------- GPU Layer & Actor-Critic --------------------
class LayerGPU:
    def __init__(self, N, tau_m, v_rest, v_thresh, v_reset, refractory, device):
        self.N = N
        self.device = device
        self.tau_m = tau_m.to(device)
        self.v = torch.full((N,), v_rest, device=device)
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.refractory = refractory.to(device)
        self.last_spike = torch.full((N,), -1e6, device=device)
        self.spike_counts = torch.zeros((N,), device=device)

    def update(self, t, dt, I):
        nr = (t - self.last_spike) >= self.refractory
        dv = torch.zeros_like(self.v)
        dv[nr] = ((-self.v[nr] + I[nr]) * dt / self.tau_m[nr])
        self.v += dv
        sp = self.v >= self.v_thresh
        if sp.any():
            self.last_spike[sp] = t
            self.v[sp] = self.v_reset
            self.spike_counts[sp] += 1
        return sp.float()

    def reset(self):
        self.v.fill_(self.v_rest)
        self.last_spike.fill_(-1e6)
        self.spike_counts.zero_()

class ActorCriticGPU:
    def __init__(self, config, device):
        num = config["actor_critic"]["num_critic_neurons"]
        tau_m = torch.full((num,), 20.0, device=device)
        ref = torch.full((num,), 2.0, device=device)
        self.layer = LayerGPU(num, tau_m,
                              config["neuron"]["v_rest"],
                              config["neuron"]["v_thresh"],
                              config["neuron"]["v_reset"],
                              ref, device)
        self.last_val = 0.0

    def compute_value(self):
        return self.layer.spike_counts.mean().item()

    def update(self, reward, gamma):
        cur = self.compute_value()
        delta = reward + gamma * cur - self.last_val
        self.last_val = cur
        return delta

    def simulate(self, t, dt):
        I = torch.zeros((self.layer.N,), device=device)
        self.layer.update(t, dt, I)

    def reset(self):
        self.layer.reset()
        self.last_val = 0.0

# -------------------- PSAC Network --------------------
class PSACNetworkGPU:
    def __init__(self, cfg, device):
        self.cfg, self.dev = cfg, device
        self.dt = cfg["simulation"]["dt"]
        # Retina is CPU-based
        self.retina = Retina(cfg)
        # Middle layer
        M = cfg["middle_layer"]["num_neurons"]
        exc = int(M * cfg["middle_layer"]["ratio_exc"])
        tau_m = torch.empty(M, device=device)
        ref   = torch.empty(M, device=device)
        tau_m[:exc]=cfg["middle_layer"]["tau_m_exc"]
        ref[:exc]  =cfg["middle_layer"]["refractory_exc"]
        tau_m[exc:]=cfg["middle_layer"]["tau_m_inh"]
        ref[exc:]  =cfg["middle_layer"]["refractory_inh"]
        self.middle = LayerGPU(M, tau_m,
                               cfg["neuron"]["v_rest"],
                               cfg["neuron"]["v_thresh"],
                               cfg["neuron"]["v_reset"],
                               ref, device)
        # Output layer
        O = cfg["output_layer"]["num_neurons"]
        tau_o = torch.full((O,), cfg["output_layer"]["tau_m"], device=device)
        ref_o = torch.full((O,), cfg["output_layer"]["refractory"], device=device)
        self.output = LayerGPU(O, tau_o,
                               cfg["neuron"]["v_rest"],
                               cfg["neuron"]["v_thresh"],
                               cfg["neuron"]["v_reset"],
                               ref_o, device)
        # Actor-critic
        self.ac = ActorCriticGPU(cfg, device)
        # Connectivity
        p = cfg["network"]["connection_prob"]
        self.W = ((torch.rand((M, O), device=device) < p).float()
                  * torch.normal(0.5,0.1,(M,O),device=device))

    def run_simulation(self, spikes, reward, verbose=False):
        dt = self.dt
        T = spikes.shape[1]
        out_counts = torch.zeros((self.output.N,), device=device)
        for step in range(T):
            t = step*dt
            rin = spikes[:,step].sum().item()
            I_mid = torch.full((self.middle.N,), rin*500.0, device=device)
            sp_mid = self.middle.update(t, dt, I_mid)
            I_out = (sp_mid.unsqueeze(0) @ self.W).squeeze(0)*10.0
            sp_out = self.output.update(t, dt, I_out)
            if t>=self.cfg["simulation"]["t_skip"]:
                out_counts += sp_out
            self.ac.simulate(t, dt)
            if sp_mid.sum()>0 and sp_out.sum()>0:
                delta = self.ac.update(reward, self.cfg["actor_critic"]["gamma"])
                upd = torch.ger(sp_mid, sp_out)*self.cfg["learning"]["learning_rate"]*delta
                self.W.add_(upd).clamp_(0,1.0)
            if verbose and step%200==0:
                print(f"t={t:.0f}ms | mid spikes={int(sp_mid.sum())}, out spikes={int(sp_out.sum())}")
        pred = int(out_counts.argmax().item())
        return pred, out_counts.cpu().numpy()

    def reset_state(self):
        self.middle.reset()
        self.output.reset()
        self.ac.reset()

# -------------------- Reward --------------------
def compute_reward(pred, label):
    return 1.0 if pred==label else -1.0

# -------------------- Training --------------------
def train_network():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0)*255)
    ])
    ds = torchvision.datasets.MNIST(root="./data", train=True,
                                    download=True, transform=transform)
    loader = DataLoader(ds, batch_size=CONFIG["training"]["batch_size"], shuffle=True)
    retina = Retina(CONFIG)
    net = PSACNetworkGPU(CONFIG, device)
    total, correct = 0, 0

    for ep in range(CONFIG["training"]["num_epochs"]):
        print(f"=== Epoch {ep+1}/{CONFIG['training']['num_epochs']} ===")
        for i, (img, lbl) in enumerate(loader):
            arr = img.squeeze(0).numpy().astype(np.uint8)
            rates = retina.process_image(arr)
            spikes = retina.generate_spike_train(rates,
                         CONFIG["simulation"]["t_total"], CONFIG["simulation"]["dt"])
            net.reset_state()
            pred, _ = net.run_simulation(spikes, 0.0, verbose=(i<2))
            r = compute_reward(pred, lbl.item())
            net.reset_state()
            pred, _ = net.run_simulation(spikes, r)
            total += 1
            correct += (pred==lbl.item())
            if (i+1)%100==0:
                print(f"Sample {i+1}: True={lbl.item()}, Pred={pred}, Acc={correct/total:.2f}")
        print(f"Epoch {ep+1} Acc: {correct/total:.2f}")
    print(f"Final train Acc: {correct/total:.2f}")
    return net

# -------------------- Fixture for pytest --------------------
@pytest.fixture(scope="session")
def network():
    return train_network()

# -------------------- Testing --------------------
def test_network(network):
    network.reset_state()
    evaluate_network(network)

# -------------------- Evaluation --------------------
def evaluate_network(network):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0)*255)
    ])
    ds = torchvision.datasets.MNIST(root="./data", train=False,
                                    download=True, transform=transform)
    loader = DataLoader(ds, batch_size=1, shuffle=False)
    retina = Retina(CONFIG)
    total, correct = 0,0
    print(f"▶️ Evaluating {len(ds)} samples...")
    for idx, (img, lbl) in enumerate(loader):
        try:
            arr = img.squeeze(0).numpy().astype(np.uint8)
            rates = retina.process_image(arr)
            spikes = retina.generate_spike_train(rates,
                         CONFIG["simulation"]["t_total"], CONFIG["simulation"]["dt"])
            network.reset_state()
            pred, _ = network.run_simulation(spikes, 0.0)
            total+=1; correct+=(pred==lbl.item())
            if total%100==0:
                print(f"[{total}/{len(ds)}] True={lbl.item()}, Pred={pred}, Acc={correct/total:.2f}")
        except Exception as e:
            print(f"Error at idx {idx}, label={lbl.item()}")
            raise
    print(f"✅ Eval Accuracy: {correct/total:.2f}")

# -------------------- Main --------------------
if __name__ == "__main__":
    trained = train_network()
    evaluate_network(trained)

    # Optionally, visualize an example’s output spike counts.
    # Here we take the first test sample.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0) * 255)
    ])
    test_dataset = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
    sample, label = test_dataset[0]
    sample_np = sample.numpy().astype(np.uint8)
    print("Visualizing output for test sample, True label:", label)
    retina = Retina(CONFIG)
    spike_rates = retina.process_image(sample_np)
    retina_spike_train = retina.generate_spike_train(spike_rates, CONFIG["simulation"]["t_total"],
                                                     CONFIG["simulation"]["dt"])
    train_network.reset_state()
    predicted_class, output_counts = train_network.run_simulation(retina_spike_train, reward=0.0, verbose=False)
    print("Predicted class:", predicted_class)
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(output_counts)), output_counts)
    plt.xlabel("Output Neuron (Class)")
    plt.ylabel("Spike Count")
    plt.title("Output Layer Spike Counts")
    plt.show()
