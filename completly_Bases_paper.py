#!/usr/bin/env python3
"""
Extended GPU-accelerated Spiking Neural Network Training with PSAC for MNIST

This script implements a simplified spiking neural network (SNN) using an Actor-Critic modulated
Power-STDP rule. It is extended to include a proper training loop over the MNIST dataset.
The simulation runs on Apple M2 GPUs (MPS device) using PyTorch.
"""

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import pytest


# Set device to MPS (Apple M2 GPU) if available, otherwise use CPU.
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Using device:", device)

# -------------------- Configuration --------------------
CONFIG = {
    "simulation": {
        "dt": 1.0,  # time-step (ms)
        "t_total": 1200,  # total simulation time per episode (ms)
        "t_skip": 200,  # initial period for stabilization (ms)
    },
    "neuron": {
        "v_rest": 0.0,  # resting potential (mV)
        "v_thresh": 18.0,  # spike threshold (mV)
        "v_reset": 0.0,  # reset potential (mV)
    },
    "middle_layer": {
        "num_neurons": 5000,
        "ratio_exc": 0.8,  # 80% excitatory neurons, 20% inhibitory
        "tau_m_exc": 20.0,  # membrane time constant for excitatory neurons (ms)
        "tau_m_inh": 10.0,  # membrane time constant for inhibitory neurons (ms)
        "refractory_exc": 2.0,  # refractory period (ms) for excitatory neurons
        "refractory_inh": 1.0,  # refractory period (ms) for inhibitory neurons
    },
    "output_layer": {
        "num_neurons": 10,  # MNIST: 10 classes
        "tau_m": 20.0,  # membrane time constant for output neurons (ms)
        "refractory": 2.0,  # refractory period for output neurons (ms)
    },
    "network": {
        "connection_prob": 0.2,  # probability for random connectivity from middle to output
    },
    "actor_critic": {
        "num_critic_neurons": 20,
        "gamma": 0.99,  # discount factor for future rewards
        "tau_r": 20.0,  # time constant for critic neurons (ms)
    },
    "learning": {
        "learning_rate": 0.001,  # learning rate for weight updates
    },
    "retina": {
        "input_size": (28, 28),  # MNIST images are 28x28
        "pool_size": 2,  # average pooling window size (2x2)
        "stride": 2,  # stride for pooling (results in 7x7 output)
        "spike_rate_scaling": 50.0  # multiplier to boost retina spike probability
    },
    "training": {
        "num_epochs": 2,  # Number of training epochs (set low for a prototype)
        "batch_size": 1,  # We'll process one image per episode for simplicity
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
        """
        Downsample the input image using average pooling.
        Returns a 2D array of activations (values normalized between 0 and 1).
        """
        h, w = self.input_size
        pool_h, pool_w = self.pool_size, self.pool_size
        out_h, out_w = self.output_dim
        pooled = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                window = image[i * self.stride: i * self.stride + pool_h,
                         j * self.stride: j * self.stride + pool_w]
                pooled[i, j] = np.mean(window)
        spike_rates = pooled / 255.0  # Normalize
        return spike_rates

    def generate_spike_train(self, spike_rates, simulation_time, dt):
        """
        Generate spike trains using a Poisson process.
        Returns a torch tensor of shape [num_units, num_time_steps] on device.
        """
        num_units = spike_rates.size
        num_steps = int(simulation_time / dt)
        spike_train = np.zeros((num_units, num_steps), dtype=np.float32)
        scaling = self.config["retina"].get("spike_rate_scaling", 1.0)
        # For each unit, determine spike times based on the scaled rate.
        for unit in range(num_units):
            rate = spike_rates.flatten()[unit]  # normalized [0,1]
            # Calculate probability per time step (dt in ms, converting to seconds)
            probs = np.random.rand(num_steps)
            spikes = (probs < (rate * scaling * dt / 1000.0)).astype(np.float32)
            spike_train[unit] = spikes
        return torch.tensor(spike_train, device=device, dtype=torch.float32)


# -------------------- GPU-Accelerated Layer Class --------------------

class LayerGPU:
    def __init__(self, num_neurons, tau_m, v_rest, v_thresh, v_reset, refractory, device):
        """
        Vectorized layer of LIF neurons.
        tau_m and refractory should be torch tensors of shape [num_neurons].
        """
        self.num_neurons = num_neurons
        self.device = device
        self.tau_m = tau_m.to(device)
        self.v = torch.full((num_neurons,), v_rest, device=device)
        self.v_rest = v_rest
        self.v_thresh = v_thresh
        self.v_reset = v_reset
        self.refractory = refractory.to(device)
        self.last_spike_time = torch.full((num_neurons,), -1e6, device=device)  # initialize far in past
        self.spike_counts = torch.zeros((num_neurons,), device=device)

    def update(self, t, dt, input_current):
        """
        Update membrane potentials in a vectorized manner.
        Returns a float tensor (0/1) indicating which neurons spiked.
        """
        not_refractory = (t - self.last_spike_time) >= self.refractory
        dv = torch.zeros_like(self.v)
        dv[not_refractory] = (
                    (-self.v[not_refractory] + input_current[not_refractory]) * dt / self.tau_m[not_refractory])
        self.v = self.v + dv
        spiked = self.v >= self.v_thresh
        if spiked.any():
            self.last_spike_time[spiked] = t
            self.v[spiked] = self.v_reset
            self.spike_counts[spiked] += 1
        return spiked.float()

    def reset(self):
        """
        Reset neuron states for new simulation episode.
        """
        self.v.fill_(self.v_rest)
        self.spike_counts.zero_()
        self.last_spike_time.fill_(-1e6)


# -------------------- GPU-Accelerated Actor-Critic Module --------------------

class ActorCriticGPU:
    def __init__(self, config, device):
        self.config = config
        num_critic = config["actor_critic"]["num_critic_neurons"]
        tau_m = torch.full((num_critic,), 20.0, device=device)
        refractory = torch.full((num_critic,), 2.0, device=device)
        self.critic_layer = LayerGPU(num_critic,
                                     tau_m,
                                     v_rest=config["neuron"]["v_rest"],
                                     v_thresh=config["neuron"]["v_thresh"],
                                     v_reset=config["neuron"]["v_reset"],
                                     refractory=refractory,
                                     device=device)
        self.last_value = 0.0

    def compute_value(self):
        value = self.critic_layer.spike_counts.mean().item()
        return value

    def update(self, reward, gamma):
        current_value = self.compute_value()
        delta = reward + gamma * current_value - self.last_value
        self.last_value = current_value
        return delta

    def simulate(self, t, dt):
        input_current = torch.zeros((self.critic_layer.num_neurons,), device=device)
        self.critic_layer.update(t, dt, input_current)

    def reset(self):
        self.critic_layer.reset()
        self.last_value = 0.0


# -------------------- GPU-Accelerated PSAC Network --------------------

class PSACNetworkGPU:
    def __init__(self, config, device):
        self.config = config
        self.device = device
        self.dt = config["simulation"]["dt"]
        t_total = config["simulation"]["t_total"]

        # Initialize Retina module (remains on CPU; output sent to GPU)
        self.retina = Retina(config=config)
        self.simulation_time = t_total

        # --- Middle (Liquid) Layer Initialization ---
        num_middle = config["middle_layer"]["num_neurons"]
        num_exc = int(num_middle * config["middle_layer"]["ratio_exc"])
        # Create per-neuron parameters for tau_m and refractory
        tau_m = torch.empty(num_middle, device=device)
        refractory = torch.empty(num_middle, device=device)
        # Set parameters for excitatory neurons
        tau_m[:num_exc] = config["middle_layer"]["tau_m_exc"]
        refractory[:num_exc] = config["middle_layer"]["refractory_exc"]
        # Set parameters for inhibitory neurons
        tau_m[num_exc:] = config["middle_layer"]["tau_m_inh"]
        refractory[num_exc:] = config["middle_layer"]["refractory_inh"]
        self.middle_layer = LayerGPU(num_middle,
                                     tau_m,
                                     v_rest=config["neuron"]["v_rest"],
                                     v_thresh=config["neuron"]["v_thresh"],
                                     v_reset=config["neuron"]["v_reset"],
                                     refractory=refractory,
                                     device=device)

        # --- Output Layer Initialization ---
        num_output = config["output_layer"]["num_neurons"]
        tau_m_output = torch.full((num_output,), config["output_layer"]["tau_m"], device=device)
        refractory_output = torch.full((num_output,), config["output_layer"]["refractory"], device=device)
        self.output_layer = LayerGPU(num_output,
                                     tau_m_output,
                                     v_rest=config["neuron"]["v_rest"],
                                     v_thresh=config["neuron"]["v_thresh"],
                                     v_reset=config["neuron"]["v_reset"],
                                     refractory=refractory_output,
                                     device=device)

        # --- Actor-Critic Module Initialization ---
        self.actor_critic = ActorCriticGPU(config, device)

        # --- Connectivity from Middle to Output Layer ---
        connection_prob = config["network"]["connection_prob"]
        m = self.middle_layer.num_neurons
        n = self.output_layer.num_neurons
        self.middle_to_output = (torch.rand((m, n), device=device) < connection_prob).float()
        weights = torch.normal(0.5, 0.1, size=(m, n), device=device)
        self.middle_to_output = self.middle_to_output * weights

    def run_simulation(self, retina_spike_train, reward, verbose=False):
        """
        Run one simulation episode over the full simulation time.
        Returns the predicted class and the output spike count vector.
        """
        dt = self.dt
        num_steps = retina_spike_train.shape[1]
        output_spike_counts = torch.zeros((self.output_layer.num_neurons,), device=device)

        for step in range(num_steps):
            t = step * dt
            # --- Retina Feedforward ---
            retina_input = retina_spike_train[:, step].sum().item()
            # Increase the feed-forward scaling factor if needed
            feed_forward = retina_input * 500.0
            middle_input = torch.full((self.middle_layer.num_neurons,), feed_forward, device=device)
            middle_spikes = self.middle_layer.update(t, dt, middle_input)

            # --- Projection to Output ---
            output_input = torch.matmul(middle_spikes.unsqueeze(0), self.middle_to_output).squeeze(0)
            output_input = output_input * 10.0  # tuning parameter
            output_spikes = self.output_layer.update(t, dt, output_input)

            # Record output spikes after stabilization period.
            if t >= self.config["simulation"]["t_skip"]:
                output_spike_counts += output_spikes

            # --- Actor-Critic Update ---
            self.actor_critic.simulate(t, dt)
            # Update weights if activity occurs in both layers.
            if middle_spikes.sum() > 0 and output_spikes.sum() > 0:
                delta = self.actor_critic.update(reward, self.config["actor_critic"]["gamma"])
                weight_update = self.config["learning"]["learning_rate"] * delta
                update_matrix = torch.ger(middle_spikes, output_spikes) * weight_update
                self.middle_to_output += update_matrix
                self.middle_to_output.clamp_(0, 1.0)

            if verbose and step % 100 == 0:
                print(
                    f"Time {t:.1f} ms: Middle spikes sum {middle_spikes.sum().item()}, Output spikes sum {output_spikes.sum().item()}")

        predicted_class = torch.argmax(output_spike_counts).item()
        return predicted_class, output_spike_counts.cpu().numpy()

    def reset_state(self):
        """
        Reset the state of all layers for a new simulation episode.
        """
        self.middle_layer.reset()
        self.output_layer.reset()
        self.actor_critic.reset()


# -------------------- Helper: Reward Function --------------------

def compute_reward(predicted_class, true_label):
    """
    Simple reward: +1 if correct, -1 otherwise.
    """
    return 1.0 if predicted_class == true_label else -1.0


# -------------------- Training Loop --------------------

def train_network():
    # Load MNIST training set.
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0) * 255)
    ])
    train_dataset = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=CONFIG["training"]["batch_size"], shuffle=True)

    retina = Retina(CONFIG)
    network = PSACNetworkGPU(CONFIG, device)

    num_epochs = CONFIG["training"]["num_epochs"]
    total_samples = 0
    correct_samples = 0

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch + 1}/{num_epochs} ---")
        for batch_idx, (image, label) in enumerate(train_loader):
            # For simplicity, use a single image per episode.
            image_np = image.squeeze(0).numpy().astype(np.uint8)
            true_label = label.item()
            # Process image through retina.
            spike_rates = retina.process_image(image_np)
            retina_spike_train = retina.generate_spike_train(spike_rates, CONFIG["simulation"]["t_total"],
                                                             CONFIG["simulation"]["dt"])
            # Reset network state.
            network.reset_state()
            # Compute reward (will be applied within simulation).
            # Run simulation (verbose for first few samples)
            predicted_class, _ = network.run_simulation(retina_spike_train, reward=0.0, verbose=(batch_idx < 2))
            # Compute reward based on prediction vs. ground truth.
            reward = compute_reward(predicted_class, true_label)
            # One more simulation pass using the computed reward to trigger weight updates.
            network.reset_state()
            predicted_class, _ = network.run_simulation(retina_spike_train, reward=reward, verbose=False)
            total_samples += 1
            if predicted_class == true_label:
                correct_samples += 1
            if (batch_idx + 1) % 100 == 0:
                print(
                    f"Sample {batch_idx + 1}: True {true_label}, Predicted {predicted_class}, Acc: {correct_samples / total_samples:.2f}")

        print(f"Epoch {epoch + 1} Accuracy: {correct_samples / total_samples:.2f}")

    print(f"Final Training Accuracy: {correct_samples / total_samples:.2f}")
    return network


# -------------------- Fixture for Training --------------------
@pytest.fixture(scope="session")
def network():
    """
    Session-scoped fixture: trains the network once per pytest session.
    Returns the trained PSACNetworkGPU instance.
    """
    # train_network() comes from your main script
    trained = train_network()
    return trained

# -------------------- Testing Loop --------------------
def test_network(network):
    """
    Pytest test function: evaluates the pre-trained 'network' fixture.
    """
    # Reset state before evaluation
    network.reset_state()
    evaluate_network(network)

# -------------------- Evaluation Loop --------------------
def evaluate_network(network):
    """
    Evaluate a pre-trained PSACNetworkGPU on the MNIST test set with detailed debugging.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.squeeze(0) * 255)
    ])
    test_dataset = torchvision.datasets.MNIST(
        root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    retina = Retina(CONFIG)

    total, correct = 0, 0
    print(f"▶️ Starting evaluation over {len(test_dataset)} samples.")

    for idx, (image, label) in enumerate(test_loader):
        try:
            img_np = image.squeeze(0).numpy().astype(np.uint8)
            true_label = label.item()

            # Retina preprocessing
            rates = retina.process_image(img_np)
            spikes = retina.generate_spike_train(
                rates, CONFIG["simulation"]["t_total"], CONFIG["simulation"]["dt"]
            )

            # Reset and run simulation
            network.reset_state()
            pred, _ = network.run_simulation(spikes, reward=0.0)

            total += 1
            correct += (pred == true_label)

            # Progress report every 100 samples
            if total % 100 == 0:
                acc = correct / total
                print(f"[{total}/{len(test_dataset)}] True={true_label}, Pred={pred}, Acc={acc:.2f}")

        except Exception as e:
            print(f"❌ Error at sample {idx}: True label={true_label}")
            print(f"   rates.shape={rates.shape}, spikes.shape={spikes.shape}")
            print("   Error:", e)
            raise

    print(f"✅ Final Evaluation Accuracy: {correct/total:.2f}")

# -------------------- Main Routine --------------------

if __name__ == "__main__":
    # Train the network on MNIST.
    trained_network = train_network()
    # Evaluate on the test set.
    evaluate_network(trained_network)

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
    trained_network.reset_state()
    predicted_class, output_counts = trained_network.run_simulation(retina_spike_train, reward=0.0, verbose=False)
    print("Predicted class:", predicted_class)
    plt.figure(figsize=(6, 4))
    plt.bar(range(len(output_counts)), output_counts)
    plt.xlabel("Output Neuron (Class)")
    plt.ylabel("Spike Count")
    plt.title("Output Layer Spike Counts")
    plt.show()
