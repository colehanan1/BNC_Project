import numpy as np
from scipy.ndimage import gaussian_filter

# 1. Load and preprocess an example image (e.g., CIFAR-10 image or MNIST digit).
# For CIFAR-10 (color), convert to grayscale for simplicity:
image = ...  # assume image is a 2D NumPy array of shape (H, W) with values 0-255
image = image.astype(float) / 255.0  # normalize to 0.0 - 1.0


sigma_small = 1.0  # small Gaussian blur (pixels)
sigma_large = 3.0  # larger Gaussian blur
blur_small = gaussian_filter(image, sigma=sigma_small)
blur_large = gaussian_filter(image, sigma=sigma_large)
dog = blur_small - blur_large  # DoG result: highlights edges

# 3. Normalize the DoG feature map to [0,1] range.
dog_norm = (dog - dog.min()) / (dog.max() - dog.min() + 1e-8)

# 4. Encode into spike times (latency coding).
T_max = 100.0  # total simulation time window in ms
# Invert intensity to get latency: higher value -> smaller spike time.
spike_times = T_max * (1.0 - dog_norm)  # array of same shape as image
# Optionally, set a threshold to avoid very low-contrast spikes:
spike_times[dog_norm < 0.1] = np.inf  # no spike if feature intensity < 0.1


# Poisson encoding (alternative approach)
R_max = 100.0  # max rate in Hz for a pixel with dog_norm=1
dt = 1.0       # time step in ms
time_steps = int(T_max / dt)
H, W = image.shape
# Initialize an array for spike train: dimensions (time_steps, H, W)
spike_train = np.zeros((time_steps, H, W), dtype=bool)
for t in range(time_steps):
    # Generate spikes with probability = rate*dt
    random_matrix = np.random.rand(H, W)
    spike_prob = dog_norm * (R_max * dt / 1000.0)  # (dt in ms, divide by 1000 to convert Hz to per ms probability)
    spike_train[t] = (random_matrix < spike_prob)
# Now spike_train[t,x,y] is True if a spike occurred at time t*dt for pixel (x,y).
