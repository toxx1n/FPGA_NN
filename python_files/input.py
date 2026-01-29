import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import MNIST

# Load MNIST test dataset
transform = transforms.Compose([transforms.ToTensor()])
mnist_test = MNIST(root="./", train=False, download=True, transform=transform)

# Get a single test image (first image)
image, _ = mnist_test[0]  # Change index if needed
image = image.squeeze().numpy()  # Convert to numpy array (28x28)

# Fixed-point scaling (assuming Q16.16 format)
scale_factor = 2**16
image_fixed = (image * scale_factor).astype(np.int32)

# Convert to 32-bit hexadecimal
hex_values = [format(val & 0xFFFFFFFF, '08X') for row in image_fixed for val in row]

padding = ["00000000"] * 20
final_data = padding + hex_values + padding

# Save to file
with open("data_in.mem", "w") as f:
    f.write("\n".join(final_data))

print("data_in.mem generated successfully!")
