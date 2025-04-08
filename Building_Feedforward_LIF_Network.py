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

num_train = 50
num_test = 20
training_images = x_train[:num_train]
training_labels = y_train[:num_train]
test_images = x_test[:num_test]
test_labels = y_test[:num_test]

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

tau_pre = 20 * ms
tau_post = 20 * ms
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

G_input = SpikeGeneratorGroup(N_input, [], [] * ms)  # correct: ensures time has units
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

syn_hidden2_output = Synapses(G_hidden2, G_output, model='w : 1', on_pre='v_post += w')
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

num_epochs = 2

print("Starting training...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    for idx, (img, label) in enumerate(zip(training_images, training_labels)):
        spikes, _ = poisson_encode_image(img)
        input_indices = [s[0] for s in spikes]
        input_times = [s[1] * ms for s in spikes]

        G_input.set_spikes(input_indices, input_times)

        G_hidden1.v = 0
        G_hidden2.v = 0
        G_output.v = 0

        net.run(100 * ms)
        print(f"  Trained on image {idx + 1}/{num_train} in epoch {epoch + 1}")

previous_weights = syn_in_hidden1.w[:]
print("Training complete.")

y_true = []
y_pred = []

print("Starting evaluation...")

for idx, (img, true_label) in enumerate(zip(test_images, test_labels)):
    spikes, _ = poisson_encode_image(img)
    input_indices = [s[0] for s in spikes]
    input_times = [s[1] * ms for s in spikes]

    G_input.set_spikes(input_indices, input_times)

    G_hidden1.v = 0
    G_hidden2.v = 0
    G_output.v = 0

    net.run(100 * ms)
    spike_counts = np.array([(spike_monitor_output.i == neur).sum() for neur in range(N_output)])
    pred_label = spike_counts.argmax()
    y_true.append(true_label)
    y_pred.append(pred_label)
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

plt.figure()
plt.title("Hidden Layer Spikes (Last Test Image)")
plt.plot(spike_monitor_hidden.t / ms, spike_monitor_hidden.i, 'k.')
plt.xlabel("Time (ms)")
plt.ylabel("Hidden Neuron Index")
plt.show()

neuron_idx = 0
plt.figure()
plt.title(f"Membrane Potential of Hidden Neuron {neuron_idx}")
plt.plot(state_monitor_hidden.t / ms, state_monitor_hidden.v[neuron_idx])
plt.xlabel("Time (ms)")
plt.ylabel("Membrane Potential (v)")
plt.show()

print("\nOutput neuron firing activity (last test image):")
for i in range(N_output):
    count = (spike_monitor_output.i == i).sum()
    print(f"Output neuron {i} fired {count} times")

firing_counts = Counter(spike_monitor_hidden.i)
heatmap_data = np.zeros(N_hidden2)
for neuron_idx, count in firing_counts.items():
    heatmap_data[neuron_idx] = count

plt.figure(figsize=(12, 3))
plt.title("Hidden Neuron Firing Rates")
plt.bar(range(N_hidden2), heatmap_data)
plt.xlabel("Neuron Index")
plt.ylabel("Spikes")
plt.show()

plt.figure()
plt.title("Output Neuron Spikes (Last Test Image)")
plt.plot(spike_monitor_output.t / ms, spike_monitor_output.i, 'r.')
plt.xlabel("Time (ms)")
plt.ylabel("Output Neuron Index")
plt.show()

print(f"\nLast Test Image: Output Spike Counts = {spike_counts}, Predicted = {pred_label}, True = {true_label}")

if previous_weights is not None:
    weight_matrix = np.array(previous_weights).reshape((N_input, N_hidden1))[:100, :100]
    plt.figure(figsize=(6, 5))
    sns.heatmap(weight_matrix, cmap='viridis')
    plt.title("Input-to-Hidden Synaptic Weights (STDP)")
    plt.xlabel("Hidden Neuron")
    plt.ylabel("Input Neuron")
    plt.show()
