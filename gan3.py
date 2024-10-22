# import os
# import numpy as np
# import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# import matplotlib.pyplot as plt
# import cv2
# img = cv2.imread('C:\\Users\\hp\\Downloads\\parent\\class\\img1.jpeg ')
# # Set your dataset directory
# dataset_dir = 'C:\\Users\\hp\\Downloads\\parent'
#
# # Create two subdirectories for the two classes (objects)
# class1_dir = os.path.join(dataset_dir, 'class1')
# class2_dir = os.path.join(dataset_dir, 'class2')
#
# # Data Augmentation for each class
# datagen_class1 = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     brightness_range=[0.7, 1.3]
# )
#
# datagen_class2 = ImageDataGenerator(
#     rotation_range=20,
#     width_shift_range=0.2,
#     height_shift_range=0.2,
#     shear_range=0.2,
#     zoom_range=0.2,
#     horizontal_flip=True,
#     brightness_range=[0.7, 1.3]
# )
#
# # Flow from directory for each class
# image_width, image_height = 64, 64  # Adjust dimensions based on your requirements
# batch_size = 32
# num_epochs = 10
#
# train_generator_class1 = datagen_class1.flow_from_directory(
#     class1_dir,
#     target_size=(image_width, image_height),
#     batch_size=batch_size,
#     class_mode='binary'  # For binary classification
# )
#
# train_generator_class2 = datagen_class2.flow_from_directory(
#     class2_dir,
#     target_size=(image_width, image_height),
#     batch_size=batch_size,
#     class_mode='binary'  # For binary classification
# )
#
# # Print number of images found in each class
# print(f"Number of images in class 1: {len(train_generator_class1.filenames)}")
# print(f"Number of images in class 2: {len(train_generator_class2.filenames)}")
#
# # Combine the generators for both classes
# combined_generator = zip(train_generator_class1, train_generator_class2)
#
# # Continue with the rest of your code...
#
#
# # VAE Model
# latent_dim = 512
#
# encoder_inputs = keras.Input(shape=(image_width, image_height, 3))  # Assuming 3 channels for RGB images
# x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
# x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
# x = layers.Flatten()(x)
# x = layers.Dense(latent_dim + latent_dim)(x)
#
# z_mean, z_log_var = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1))(x)
# sampled_z = layers.Lambda(lambda x: x[0] + tf.exp(0.5 * x[1]) * tf.random.normal(tf.shape(x[0])))([z_mean, z_log_var])
#
# encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, sampled_z], name="encoder")
#
# decoder_inputs = keras.Input(shape=(latent_dim,))
# x = layers.Dense(units=8 * 8 * 256, activation="relu")(decoder_inputs)
# print(x.shape)
# x = layers.Reshape((16, 16, 64))(x)
# x = layers.Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")(x)
# x = layers.Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same")(x)
# decoder_outputs = layers.Reshape((image_width, image_height, 3))(x)
#
#
# decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")
#
# # Discriminator
# discriminator_inputs = layers.Input(shape=(image_width, image_height, 3))
# y = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(discriminator_inputs)
# y = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(y)
# y = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(y)
# y = layers.Flatten()(y)
# y = layers.Dense(1, activation="sigmoid")(y)
#
# discriminator = keras.Model(discriminator_inputs, y, name="discriminator")
# discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
#
# # VAE model
# vae_outputs = decoder(encoder(encoder_inputs)[2])
# vae = keras.Model(encoder_inputs, vae_outputs, name="vae")
#
# # Discriminator model in VAE
# discriminator.trainable = False
# vae_with_discriminator = keras.Model(encoder_inputs, [vae_outputs, discriminator(vae_outputs)])
#
# # Loss function for discriminator
# def custom_discriminator_loss(y_true, y_pred):
#     # Binary crossentropy for binary classification
#     return keras.losses.binary_crossentropy(y_true, y_pred)
#
# # Compile the discriminator in the VAE
# vae_with_discriminator.compile(optimizer="adam", loss=["binary_crossentropy", custom_discriminator_loss], loss_weights=[1.0, 1e-4])
#
# # Training the VAE with discriminator
#
# # Generate new images from the latent space
# def generate_images_from_latent_space(num_images, latent_dim, decoder_model):
#     # Generate random samples in the latent space
#     latent_samples = np.random.normal(size=(num_images, latent_dim))
#
#     # Use the decoder to generate images from the latent space
#     generated_images = decoder_model.predict(latent_samples)
#
#     return generated_images
#
# # Specify the number of images to generate
# num_generated_images = 10
#
# # Generate new images from the latent space
# generated_images = generate_images_from_latent_space(num_generated_images, latent_dim, decoder)
#
# # Visualize the generated images
# plt.figure(figsize=(10, 5))
# for i in range(num_generated_images):
#     plt.subplot(2, 5, i + 1)
#     plt.imshow(generated_images[i])
#     plt.axis("off")
# plt.show()
#
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set your dataset directory
dataset_dir = 'C:\\Users\\hp\\Downloads\\parent'

# Data Augmentation for each class
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

# Flow from directory for each class
image_width, image_height = 64, 64  # Adjust dimensions based on your requirements
batch_size = 32
num_epochs = 10

train_generator = datagen.flow_from_directory(
    dataset_dir,
    target_size=(image_width, image_height),
    batch_size=batch_size,
    class_mode='categorical'  # Use categorical for multi-class labeling
)

# VAE Model
latent_dim = 512

encoder_inputs = keras.Input(shape=(image_width, image_height, 3))  # Assuming 3 channels for RGB images
x = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(encoder_inputs)
x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(x)
x = layers.Flatten()(x)
x = layers.Dense(latent_dim + latent_dim)(x)

z_mean, z_log_var = layers.Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=1))(x)
sampled_z = layers.Lambda(lambda x: x[0] + tf.exp(0.5 * x[1]) * tf.random.normal(tf.shape(x[0])))([z_mean, z_log_var])

encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, sampled_z], name="encoder")

decoder_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(units=8 * 8 * 256, activation="relu")(decoder_inputs)
x = layers.Reshape((16, 16, 64))(x)
x = layers.Conv2DTranspose(filters=128, kernel_size=3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(filters=64, kernel_size=3, activation="relu", strides=2, padding="same")(x)
x = layers.Conv2DTranspose(filters=3, kernel_size=3, activation="sigmoid", padding="same")(x)
decoder_outputs = layers.Reshape((image_width, image_height, 3))(x)

decoder = keras.Model(decoder_inputs, decoder_outputs, name="decoder")

# Discriminator
discriminator_inputs = layers.Input(shape=(image_width, image_height, 3))
y = layers.Conv2D(64, 3, activation="relu", strides=2, padding="same")(discriminator_inputs)
y = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(y)
y = layers.Conv2D(256, 3, activation="relu", strides=2, padding="same")(y)
y = layers.Flatten()(y)
y = layers.Dense(1, activation="sigmoid")(y)

discriminator = keras.Model(discriminator_inputs, y, name="discriminator")
discriminator.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# VAE model
vae_outputs = decoder(encoder(encoder_inputs)[2])
vae = keras.Model(encoder_inputs, vae_outputs, name="vae")

# Discriminator model in VAE
discriminator.trainable = False
vae_with_discriminator = keras.Model(encoder_inputs, [vae_outputs, discriminator(vae_outputs)])

# Loss function for discriminator
def custom_discriminator_loss(y_true, y_pred):
    # Binary crossentropy for binary classification
    return keras.losses.binary_crossentropy(y_true, y_pred)

# Compile the discriminator in the VAE
vae_with_discriminator.compile(optimizer="adam", loss=["binary_crossentropy", custom_discriminator_loss], loss_weights=[1.0, 1e-4])

# Training the VAE with discriminator
# for epoch in range(num_epochs):
#     for batch in train_generator:
#         # Unpack the batch data
#         images, labels = batch
#
#         # Combine images and labels for discriminator training
#         discriminator_labels = np.zeros((batch_size * 2, 1))
#         discriminator_labels[:batch_size, 0] = 1  # Label for images from class1
#
#         # Train the discriminator
#         d_loss = discriminator.train_on_batch(images, discriminator_labels)
#
#         # Train the VAE with discriminator
#         vae_loss = vae_with_discriminator.train_on_batch(images, [images, np.zeros((batch_size, 1))])
#
#     # Print training progress
#     print(f"Epoch {epoch + 1}/{num_epochs}, VAE Loss: {vae_loss[0]}, Discriminator Loss: {d_loss[0]}")

# Generate new images from the latent space
def generate_images_from_latent_space(num_images, latent_dim, decoder_model):
    # Generate random samples in the latent space
    latent_samples = np.random.normal(size=(num_images, latent_dim))

    # Use the decoder to generate images from the latent space
    generated_images = decoder_model.predict(latent_samples)

    return generated_images

# Specify the number of images to generate
num_generated_images = 10

# Generate new images from the latent space
generated_images = generate_images_from_latent_space(num_generated_images, latent_dim, decoder)

# Visualize the generated images
plt.figure(figsize=(10, 5))
for i in range(num_generated_images):
    plt.subplot(2, 5, i + 1)
    plt.imshow(generated_images[i])
    plt.axis("off")
plt.show()
