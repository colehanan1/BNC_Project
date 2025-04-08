import numpy as np
from scipy.ndimage import gaussian_filter
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt

# 1. Download and load the CIFAR-10 dataset.
# This automatically downloads the dataset if it isn’t already available locally.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. Select an example color image from the training set.
# CIFAR-10 images are 32x32 with 3 channels (RGB) and pixel values in [0, 255].
image_color = x_train[0]

# 3. Convert the color image to grayscale.
# We use the standard luminance formula: Y = 0.299 R + 0.587 G + 0.114 B.
image = np.dot(image_color[..., :3], [0.299, 0.587, 0.114])
image = image.astype(float) / 255.0  # Normalize the grayscale image to [0, 1].

# Optional: Visualize the grayscale image.
plt.imshow(image, cmap='gray')
plt.title("Grayscale CIFAR-10 Image")
plt.axis('off')
plt.show()

# 4. Apply Difference-of-Gaussians (DoG) filtering to extract edges/contrast.
sigma_small = 1.0  # small Gaussian blur (pixels)
sigma_large = 3.0  # larger Gaussian blur
blur_small = gaussian_filter(image, sigma=sigma_small)
blur_large = gaussian_filter(image, sigma=sigma_large)
dog = blur_small - blur_large  # The DoG result highlights edges.

# Optional: Visualize the DoG filtered image.
plt.imshow(dog, cmap='gray')
plt.title("Difference-of-Gaussians (DoG)")
plt.axis('off')
plt.show()

# 5. Normalize the DoG feature map to the [0, 1] range.
dog_norm = (dog - dog.min()) / (dog.max() - dog.min() + 1e-8)

# 6. Encode the normalized DoG image into spike times (latency coding).
T_max = 100.0  # Total simulation time window in milliseconds.
# In latency coding, higher intensity leads to an earlier spike.
spike_times = T_max * (1.0 - dog_norm)

# Optionally, set a threshold to ignore very low-contrast areas (no spike if intensity is low).
#spike_times[dog_norm < 0.1] = np.inf  # np.inf indicates no spike within the simulation window.

# Output some details about the resulting spike_times.
print("Spike times array shape:", spike_times.shape)
print("Spike times (ms):")
print(spike_times)
