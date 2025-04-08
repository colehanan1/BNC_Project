from brian2 import *
from brian2 import prefs
prefs.codegen.target = "numpy"

import numpy as np
from tensorflow.keras.datasets import cifar10
from skimage.filters import gabor
from skimage.color import rgb2gray
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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

def encode_image_to_spikes(image_color, T_max=100.0, threshold=0.2,
                           frequency=0.3, orientations=8):
    image = rgb2gray(image_color.astype(float) / 255.0)
    p2, p98 = np.percentile(image, (2, 98))
    image = np.clip((image - p2) / (p98 - p2 + 1e-8), 0, 1)
    gabor_sum = np.zeros_like(image)
    for theta in np.linspace(0, np.pi, orientations, endpoint=False):
        filt_real, _ = gabor(image, frequency=frequency, theta=theta)
        gabor_sum += filt_real
    gabor_norm = (gabor_sum - gabor_sum.min()) / (gabor_sum.max() - gabor_sum.min() + 1e-8)
    spike_times = T_max * (1.0 - gabor_norm)
    spike_times[gabor_norm < threshold] = np.inf
    return spike_times, image

def spikes_from_array(spike_times):
    input_indices = []
    input_times = []
    H, W = spike_times.shape
    for i in range(H):
        for j in range(W):
            t = spike_times[i, j]
            if np.isfinite(t):
                neuron_idx = i * W + j
                input_indices.append(neuron_idx)
                input_times.append(t)
    return input_indices, input_times

tau_pre = 20 * ms
tau_post = 20 * ms
A_pre = 0.01
A_post = -0.012
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

example_spike_times, gray_example = encode_image_to_spikes(training_images[0])
H_img, W_img = gray_example.shape
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

G_hidden1 = NeuronGroup(N_hidden1, eqs, threshold='v > V_th', reset='v = V_reset',
                        refractory=refractory, method='linear')
G_hidden2 = NeuronGroup(N_hidden2, eqs, threshold='v > V_th', reset='v = V_reset',
                        refractory=refractory, method='linear')
G_output = NeuronGroup(N_output, eqs, threshold='v > V_th', reset='v = V_reset',
                       refractory=refractory, method='linear')

syn_hidden1_hidden2 = Synapses(G_hidden1, G_hidden2, model='w : 1', on_pre='v_post += w')
syn_hidden1_hidden2.connect(p=1.0)
syn_hidden1_hidden2.w = '0.1 * rand()'

syn_hidden2_output = Synapses(G_hidden2, G_output, model='w : 1', on_pre='v_post += w')
syn_hidden2_output.connect(p=1.0)
syn_hidden2_output.w = '0.1 * rand()'

spike_monitor_hidden = SpikeMonitor(G_hidden2)
spike_monitor_output = SpikeMonitor(G_output)
state_monitor_hidden = StateMonitor(G_hidden2, 'v', record=True)

net = Network()
net.add(G_hidden1, G_hidden2, G_output,
        syn_hidden1_hidden2, syn_hidden2_output,
        spike_monitor_hidden, spike_monitor_output, state_monitor_hidden)

num_epochs = 10
previous_weights = None

print("Starting training...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    for idx, (img, label) in enumerate(zip(training_images, training_labels)):
        spike_times, _ = encode_image_to_spikes(img)
        input_indices, input_times = spikes_from_array(spike_times)
        current_offset = float(defaultclock.t / ms)
        shifted_input_times = [(t + current_offset) * ms for t in input_times]

        try: net.remove(G_input)
        except: pass
        G_input = SpikeGeneratorGroup(N_input, input_indices, shifted_input_times)
        net.add(G_input)

        try: net.remove(syn_in_hidden1)
        except: pass
        syn_in_hidden1 = Synapses(G_input, G_hidden1, model=stdp_model,
                                  on_pre=stdp_on_pre, on_post=stdp_on_post)
        syn_in_hidden1.connect(p=1.0)
        if previous_weights is not None:
            syn_in_hidden1.w = previous_weights
        else:
            syn_in_hidden1.w = '0.1 * rand()'
        net.add(syn_in_hidden1)

        net.run(100 * ms)
        previous_weights = syn_in_hidden1.w[:]
        print(f"  Trained on image {idx + 1}/{num_train} in epoch {epoch + 1}")

print("Training complete.")

print("Starting evaluation...")
y_true = []
y_pred = []

for idx, (img, true_label) in enumerate(zip(test_images, test_labels)):
    spike_times, _ = encode_image_to_spikes(img)
    input_indices, input_times = spikes_from_array(spike_times)
    current_offset = float(defaultclock.t / ms)
    shifted_input_times = [(t + current_offset) * ms for t in input_times]

    try: net.remove(G_input)
    except: pass
    G_input = SpikeGeneratorGroup(N_input, input_indices, shifted_input_times)
    net.add(G_input)

    try: net.remove(syn_in_hidden1)
    except: pass
    syn_in_hidden1 = Synapses(G_input, G_hidden1, model=stdp_model,
                              on_pre=stdp_on_pre, on_post=stdp_on_post)
    syn_in_hidden1.connect(p=1.0)
    syn_in_hidden1.w = previous_weights
    net.add(syn_in_hidden1)

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