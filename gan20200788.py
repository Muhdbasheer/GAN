import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10

# Load CIFAR-10 dataset as an example
(x_train, _), (x_test, _) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Define Gaussian noise parameters
mu = 0.0
sigma = 0.2  # Adjust the amount of noise

# Add Gaussian noise directly to the original images
x_train_noisy = x_train + np.random.normal(mu, sigma, size=x_train.shape)
x_test_noisy = x_test + np.random.normal(mu, sigma, size=x_test.shape)

# Clip the values to be in the valid range [0, 1]
x_train_noisy = np.clip(x_train_noisy, 0.0, 1.0)
x_test_noisy = np.clip(x_test_noisy, 0.0, 1.0)

# Define the autoencoder architecture
input_img = Input(shape=(32, 32, 3))

# Encoder
encoded = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)

# Decoder
decoded = Conv2D(3, (3, 3), activation='sigmoid', padding='same')(encoded)

# Build the autoencoder model
autoencoder = Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder on noisy images
autoencoder.fit(x_train_noisy, x_train, epochs=10, batch_size=128, shuffle=True)

# Use the trained autoencoder to denoise test images
decoded_imgs = autoencoder.predict(x_test_noisy)

# Display original, noisy, and denoised images for the first set
n = 5  # Number of images to display
plt.figure(figsize=(15, 6))
for i in range(n):
    # Original images
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(x_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Original")

    # Noisy input images
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(x_test_noisy[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Noisy")

    # Decoded (Denoised) images
    ax = plt.subplot(3, n, i + 1 + 2 * n)
    plt.imshow(decoded_imgs[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_title("Denoised")

plt.show()
