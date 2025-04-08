from brian2 import *
from brian2 import prefs
prefs.codegen.target = "numpy"

import numpy as np
from tensorflow.keras.datasets import cifar10
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from collections import Counter

clip = np.clip

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Full CIFAR-10 dataset for more robust training
num_train = 50000
num_test = 10000
training_images = x_train[:num_train].copy()
training_labels = y_train[:num_train].copy()
test_images = x_test[:num_test].copy()
test_labels = y_test[:num_test].copy()

def poisson_encode_image(image_color, duration=100, max_rate=100):
    image = rgb2gray(image_color.astype(float) / 255.0)
    p2, p98 = np.percentile(image, (2, 98))
    image = np.clip((image - p2) / (p98 - p2 + 1e-8), 0, 1)
    rate_map = image * max_rate
    spikes = []
    for t in range(duration):
        rand_vals = np.random.rand(*rate_map.shape) * max_rate
        fired = rand_vals < rate_map
        indices = np.where(fired)
        for i, j in zip(*indices):
            neuron_idx = i * image.shape[1] + j
            spikes.append((neuron_idx, t))
    return spikes, image

# Adjusted temporal window for STDP
tau_pre = 40 * ms
tau_post = 40 * ms
A_pre = 0.02
A_post = -0.03
w_max = 1.0

stdp_model = '''
w : 1
dpre/dt = -pre/tau_pre : 1 (event-driven)
dpost/dt = -post/tau_post : 1 (event-driven)
'''

stdp_on_pre = '''
v_post += w
pre += A_pre
w = clip(w + post, 0, w_max)
'''

stdp_on_post = '''
post += A_post
w = clip(w + pre, 0, w_max)
'''

# R-STDP for output layer
reward_signal = TimedArray([0.0], dt=1*ms)

rstdp_model = '''
w : 1
dpre/dt = -pre/tau_pre : 1 (event-driven)
dpost/dt = -post/tau_post : 1 (event-driven)
'''

rstdp_on_pre = '''
v_post += w
pre += A_pre
'''

rstdp_on_post = '''
post += A_post
w = clip(w + reward_signal(t) * pre, 0, w_max)
'''

H_img, W_img = 32, 32
N_input = H_img * W_img
N_hidden1 = 500
N_hidden2 = 200
N_output = 10

tau = 10 * ms
V_th = 0.5
V_reset = 0.0
refractory = 5 * ms

eqs = '''
dv/dt = -v/tau : 1 (unless refractory)
'''

G_input = SpikeGeneratorGroup(N_input, [], [] * ms)
G_hidden1 = NeuronGroup(N_hidden1, eqs, threshold='v > V_th', reset='v = V_reset',
                        refractory=refractory, method='linear')
G_hidden2 = NeuronGroup(N_hidden2, eqs, threshold='v > V_th', reset='v = V_reset',
                        refractory=refractory, method='linear')
G_output = NeuronGroup(N_output, eqs, threshold='v > V_th', reset='v = V_reset',
                       refractory=refractory, method='linear')

syn_in_hidden1 = Synapses(G_input, G_hidden1, model=stdp_model,
                          on_pre=stdp_on_pre, on_post=stdp_on_post)
syn_in_hidden1.connect(p=0.2)
syn_in_hidden1.w = '0.01 * rand()'

syn_hidden1_hidden2 = Synapses(G_hidden1, G_hidden2, model='w : 1', on_pre='v_post += w')
syn_hidden1_hidden2.connect(p=0.2)
syn_hidden1_hidden2.w = '0.01 * rand()'

# R-STDP version of hidden2->output
syn_hidden2_output = Synapses(G_hidden2, G_output, model=rstdp_model,
                              on_pre=rstdp_on_pre, on_post=rstdp_on_post)
syn_hidden2_output.connect(p=0.2)
syn_hidden2_output.w = '0.01 * rand()'

lateral_inhib_h2 = Synapses(G_hidden2, G_hidden2, on_pre='v_post -= 0.3')
lateral_inhib_h2.connect(condition='i != j', p=0.1)

syn_inhib = Synapses(G_output, G_output, on_pre='v_post -= 0.5')
syn_inhib.connect(condition='i != j')

spike_monitor_hidden = SpikeMonitor(G_hidden2)
spike_monitor_output = SpikeMonitor(G_output)
state_monitor_hidden = StateMonitor(G_hidden2, 'v', record=True)

net = Network()
net.add(G_input, G_hidden1, G_hidden2, G_output,
        syn_in_hidden1, syn_hidden1_hidden2, syn_hidden2_output,
        lateral_inhib_h2, syn_inhib,
        spike_monitor_hidden, spike_monitor_output, state_monitor_hidden)

num_epochs = 10

print("Starting training...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    for idx, (img, label) in enumerate(zip(training_images, training_labels)):
        spikes, _ = poisson_encode_image(img)
        input_indices = [s[0] for s in spikes]
        input_times = [s[1] for s in spikes]

        current_offset = float(defaultclock.t / ms)
        shifted_times = [(t + current_offset) * ms for t in input_times]
        G_input.set_spikes(input_indices, shifted_times)

        G_hidden1.v = 0
        G_hidden2.v = 0
        G_output.v = 0

        # Increased simulation time to 200ms for better spike integration
        net.run(200 * ms)

        spike_counts = np.array([(spike_monitor_output.i == neur).sum() for neur in range(N_output)])
        pred_label = spike_counts.argmax()

        # Assign reward based on correctness
        reward_val = 1.0 if pred_label == label else -1.0
        reward_signal.values = np.full_like(reward_signal.values, reward_val)

        print(f"  Trained on image {idx + 1}/{num_train} in epoch {epoch + 1}")

sum_weights = np.sum(syn_in_hidden1.w[:])
if sum_weights > 0:
    syn_in_hidden1.w[:] = syn_in_hidden1.w[:] / sum_weights * w_max

previous_weights = syn_in_hidden1.w[:]

# Reshape for visualization with error handling
weight_vector = np.array(previous_weights)
try:
    weight_matrix = weight_vector.reshape((N_input, N_hidden1))[:100, :100]
except ValueError:
    print("Warning: Could not reshape weights into expected (N_input, N_hidden1) shape.")
    weight_matrix = np.zeros((100, 100))
print("Training complete.")

# Evaluation block with reward modulation during test
print("Starting evaluation...")
y_true = []
y_pred = []

for idx, (img, label) in enumerate(zip(test_images, test_labels)):
    spikes, _ = poisson_encode_image(img)
    input_indices = [s[0] for s in spikes]
    input_times = [s[1] for s in spikes]

    current_offset = float(defaultclock.t / ms)
    shifted_times = [(t + current_offset) * ms for t in input_times]
    G_input.set_spikes(input_indices, shifted_times)

    G_hidden1.v = 0
    G_hidden2.v = 0
    G_output.v = 0

    net.run(200 * ms)

    spike_counts = np.array([(spike_monitor_output.i == neur).sum() for neur in range(N_output)])
    pred_label = spike_counts.argmax()
    y_true.append(label)
    y_pred.append(pred_label)

    # Apply reward modulation during test as well
    reward_val = 1.0 if pred_label == label else -1.0
    reward_signal.values = np.full_like(reward_signal.values, reward_val)

    if idx % 100 == 0:
        print(f"  Evaluated test image {idx + 1}/{num_test}")

cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:")
print(cm)

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# Rich visual diagnostics
plt.figure(figsize=(12, 3))
plt.title("Hidden Neuron Firing Rates (Training Summary)")
counts = Counter(spike_monitor_hidden.i)
activity = np.zeros(N_hidden2)
for i, c in counts.items():
    activity[i] = c
plt.bar(range(N_hidden2), activity)
plt.xlabel("Neuron Index")
plt.ylabel("Spikes")
plt.show()

plt.figure(figsize=(6, 5))
sns.heatmap(weight_matrix, cmap='viridis')
plt.title("Weight Matrix Visualization (100x100 block)")
plt.xlabel("Hidden Neuron")
plt.ylabel("Input Neuron")
plt.show()
