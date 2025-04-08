from brian2 import *  # Brian2 for SNN simulation
from brian2 import prefs

prefs.codegen.target = "numpy"  # Use numpy code generation to avoid clang/Cython issues

import numpy as np
from tensorflow.keras.datasets import cifar10
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Make sure np.clip is available in the global namespace
clip = np.clip

# ---------------------------
# Load CIFAR-10 and define class names
# ---------------------------
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = y_train.flatten()
y_test = y_test.flatten()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# For demonstration, use a small subset:
num_train = 50
num_test = 20
training_images = x_train[:num_train]
training_labels = y_train[:num_train]
test_images = x_test[:num_test]
test_labels = y_test[:num_test]


# ---------------------------
# Helper Functions
# ---------------------------
def encode_image_to_spikes(image_color, T_max=100.0, threshold=0.1,
                           sigma_small=1.0, sigma_large=3.0):
    """
    Convert a CIFAR-10 color image into spike times using DoG filtering and latency coding.
    Returns:
      spike_times: 2D array of spike times (ms) with np.inf for no spike.
      gray_image: the grayscale version of the image.
    """
    # Convert color image to grayscale using luminance formula.
    image = np.dot(image_color[..., :3], [0.299, 0.587, 0.114])
    image = image.astype(float) / 255.0  # normalize to [0,1]

    # Apply Difference-of-Gaussians filtering.
    blur_small = gaussian_filter(image, sigma=sigma_small)
    blur_large = gaussian_filter(image, sigma=sigma_large)
    dog = blur_small - blur_large

    # Normalize DoG result to [0, 1]
    dog_norm = (dog - dog.min()) / (dog.max() - dog.min() + 1e-8)

    # Compute spike times: higher intensity => earlier spike.
    spike_times = T_max * (1.0 - dog_norm)
    spike_times[dog_norm < threshold] = np.inf  # no spike if below threshold
    return spike_times, image


def spikes_from_array(spike_times):
    """
    Convert a 2D spike_times array into lists of neuron indices and spike times.
    Returns:
      input_indices: list of neuron indices
      input_times: list of spike times (in ms) – these are relative times (starting at 0).
    """
    input_indices = []
    input_times = []
    H, W = spike_times.shape
    for i in range(H):
        for j in range(W):
            t = spike_times[i, j]
            if np.isfinite(t):
                neuron_idx = i * W + j
                input_indices.append(neuron_idx)
                input_times.append(t)  # leave as numeric (ms); we add units later.
    return input_indices, input_times


# ---------------------------
# Define STDP parameters and synapse model for input->hidden connection
# ---------------------------
tau_pre = 20 * ms
tau_post = 20 * ms
A_pre = 0.01
A_post = -0.012  # negative value for depression
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

# ---------------------------
# Build hidden and output layers (these remain fixed during training)
# ---------------------------
# Use one example image to get dimensions.
example_spike_times, gray_example = encode_image_to_spikes(training_images[0])
H_img, W_img = gray_example.shape
N_input = H_img * W_img  # one input neuron per pixel
N_hidden = 100  # adjustable number of hidden neurons
N_output = 10  # one neuron per class

# Define neuron model (LIF neurons)
tau = 10 * ms  # membrane time constant (global variable)
V_th = 0.5  # threshold
V_reset = 0.0  # reset potential
refractory = 5 * ms  # refractory period

eqs = '''
dv/dt = -v/tau : 1 (unless refractory)
'''

# Create hidden and output groups.
G_hidden = NeuronGroup(N_hidden, eqs, threshold='v > V_th', reset='v = V_reset',
                       refractory=refractory, method='linear')
G_output = NeuronGroup(N_output, eqs, threshold='v > V_th', reset='v = V_reset',
                       refractory=refractory, method='linear')

# Create synapses from hidden to output (non-STDP, fixed for now).
syn_hidden_out = Synapses(G_hidden, G_output, model='w : 1', on_pre='v_post += w')
syn_hidden_out.connect(p=1.0)
syn_hidden_out.w = '0.1 * rand()'

# Create monitors for hidden and output layers.
spike_monitor_hidden = SpikeMonitor(G_hidden)
spike_monitor_output = SpikeMonitor(G_output)
state_monitor_hidden = StateMonitor(G_hidden, 'v', record=True)

# ---------------------------
# Create the Network object with fixed components.
# ---------------------------
net = Network()
# We do NOT add the input group and its synapses yet (they will be added per training sample).
net.add(G_hidden, G_output, syn_hidden_out,
        spike_monitor_hidden, spike_monitor_output, state_monitor_hidden)

# ---------------------------
# Training loop
# ---------------------------
num_epochs = 10  # for demonstration; increase as needed
previous_weights = None  # to carry over synaptic weights between samples

print("Starting training...")

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")
    for idx, (img, label) in enumerate(zip(training_images, training_labels)):
        # Encode image into spike times.
        spike_times, _ = encode_image_to_spikes(img)
        input_indices, input_times = spikes_from_array(spike_times)

        # Get the current simulation time offset.
        current_offset = float(defaultclock.t / ms)  # in ms

        # Shift input spike times so that they occur after the current simulation time.
        shifted_input_times = [(t + current_offset) * ms for t in input_times]

        # Remove any existing input group.
        try:
            net.remove(G_input)
        except Exception:
            pass
        # Create a new input group with the current image's (shifted) spikes.
        G_input = SpikeGeneratorGroup(N_input, input_indices, shifted_input_times)
        net.add(G_input)

        # Remove and re-create the STDP synapses from input to hidden.
        try:
            net.remove(syn_in_hidden)
        except Exception:
            pass
        syn_in_hidden = Synapses(G_input, G_hidden, model=stdp_model,
                                 on_pre=stdp_on_pre, on_post=stdp_on_post)
        syn_in_hidden.connect(p=1.0)
        if previous_weights is not None:
            syn_in_hidden.w = previous_weights
        else:
            syn_in_hidden.w = '0.1 * rand()'
        net.add(syn_in_hidden)

        # Run simulation for this image.
        net.run(100 * ms)
        # (No need to reset the clock; we already shift spike times each run.)

        # Store updated weights for next iteration.
        previous_weights = syn_in_hidden.w[:]

        print(f"  Trained on image {idx + 1}/{num_train} in epoch {epoch + 1}")

print("Training complete.")

# ---------------------------
# Evaluation loop
# ---------------------------
print("Starting evaluation...")
y_true = []
y_pred = []

for idx, (img, true_label) in enumerate(zip(test_images, test_labels)):
    spike_times, _ = encode_image_to_spikes(img)
    input_indices, input_times = spikes_from_array(spike_times)
    current_offset = float(defaultclock.t / ms)  # current time in ms
    shifted_input_times = [(t + current_offset) * ms for t in input_times]

    # Remove old input group and create a new one.
    try:
        net.remove(G_input)
    except Exception:
        pass
    G_input = SpikeGeneratorGroup(N_input, input_indices, shifted_input_times)
    net.add(G_input)

    # Re-create synapses from input to hidden using the stored weights.
    try:
        net.remove(syn_in_hidden)
    except Exception:
        pass
    syn_in_hidden = Synapses(G_input, G_hidden, model=stdp_model,
                             on_pre=stdp_on_pre, on_post=stdp_on_post)
    syn_in_hidden.connect(p=1.0)
    syn_in_hidden.w = previous_weights  # use the trained weights
    net.add(syn_in_hidden)

    # Run simulation for this test image.
    net.run(100 * ms)
    # No need to reset the clock since we already shift spike times.

    # Count output spikes to decide the predicted label.
    spike_counts = np.array([(spike_monitor_output.i == neur).sum() for neur in range(N_output)])
    pred_label = spike_counts.argmax()
    y_true.append(true_label)
    y_pred.append(pred_label)

    print(f"  Evaluated test image {idx + 1}/{num_test}")

# Compute and display the confusion matrix.
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

# ---------------------------
# Additional Visualizations
# ---------------------------
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
