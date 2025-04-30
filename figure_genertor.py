import torch
import torchvision
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F

# 1. Load MNIST (will download the first time you run it)
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=ToTensor())
img_tensor, label = mnist[0]  # grab the very first sample
# img_tensor has shape (1, 28, 28)

# Convert to numpy for plotting
orig = img_tensor.squeeze().numpy()

# 2. Downsample via average pooling with kernel_size=4
#    (28 ÷ 4 = 7)
down = F.avg_pool2d(img_tensor.unsqueeze(0), kernel_size=4).squeeze().numpy()

# 3a. Plot the original 28×28 image
plt.figure(figsize=(4,4))
plt.imshow(orig, cmap='gray')
plt.title('28×28 MNIST Image')
plt.axis('off')  # hide axes
plt.show()

# 3b. Plot the downsampled 7×7 image
plt.figure(figsize=(4,4))
plt.imshow(down, cmap='gray', interpolation='nearest')
plt.title('Downsampled 7×7 Image\n(→ 49 inputs per time‑step)')
plt.axis('off')
plt.show()
