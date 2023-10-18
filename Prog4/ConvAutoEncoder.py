# CS510 CV & DL. Summer 2023. Prog 4. Cera Oh
# Exercise 1: Convolutional Autoencoder
import tensorflow as tf
import numpy as np
from keras import models
from keras import layers
from keras import metrics
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import random

# Dataset
test_data, train_data = tf.keras.datasets.fashion_mnist.load_data()

# Separate out image array from label array
test_img, test_label = test_data  # tuple
train_img, train_label = train_data  # tuple

# Pre-process by dividing each images by 255
test_img = test_img * 1/255
test_img = np.reshape(test_img, (len(test_img), 28, 28, 1))
train_img = train_img * 1/255
train_img = np.reshape(train_img, (len(train_img), 28, 28, 1))

################################################
# Step 1
################################################
# Build Model
model = models.Sequential()

# Encoder
model.add(tf.keras.Input(shape=(28, 28, 1)))
model.add(layers.Conv2D(112, 3, activation='relu'))  # (None, 26, 26, 112)
model.add(layers.BatchNormalization())  # (None, 26, 26, 112)
model.add(layers.MaxPooling2D(2))  # (None, 13, 13, 112)
model.add(layers.Conv2D(112, 3, activation='relu'))  # (None, 26, 26, 112)
model.add(layers.BatchNormalization())  # (None, 11, 11, 64)
model.add(layers.MaxPooling2D(2))  # (None, 5, 5, 64)
# Bottleneck
model.add(layers.Flatten())  # Output shape (None, 1600)
model.add(layers.Dense(2, activation='relu'))  # (None, 2)
model.add(layers.BatchNormalization())  # (None, 2)
# Decoder
model.add(layers.Dense(1024, activation='relu'))  # (None, 1600)
model.add(layers.BatchNormalization())  # (None, 1600)
model.add(layers.Reshape((4, 4, 64), input_shape=(2, )))  # (None, 4, 4, 64)
model.add(layers.BatchNormalization())  # (None, 4, 4, 64)
model.add(layers.UpSampling2D(3))  # (None, 12, 12, 64)
model.add(layers.Conv2D(112, 3, activation='relu'))  # (None, 26, 26, 112)
model.add(layers.BatchNormalization())
model.add(layers.UpSampling2D(3))  # (None, 30, 30, 64)
model.add(layers.Conv2D(112, 3, activation='relu'))  # (None, 28, 28, 112)
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(1, 1, activation='sigmoid'))  # (None, 28, 28, 1)

model.summary()

################################################
# Step 2
################################################
model.compile(optimizer='adam', loss='binary_crossentropy')

# Train model
model.fit(train_img, train_img, epochs=100, batch_size=100,
          shuffle=True, validation_data=(test_img, test_img))

# Get Images
prediction = model.predict(test_img)

count = 0
while count < 10:
    img = prediction[count].reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.title(f"Model Output {count}")
    plt.show()
    plt.imshow(test_img[count], cmap='gray')
    plt.title(f"Test Image {count}")
    plt.show()
    count = count + 1

model.save("model.h5")

################################################
# Step 3
################################################
# Add noise to test and train data (from Lecture notes)
noise = 0.5
noisy_train_img = train_img + noise * \
    np.random.normal(loc=0.0, scale=1.0, size=train_img.shape)
noisy_test_img = test_img + noise * \
    np.random.normal(loc=0.0, scale=1.0, size=test_img.shape)
noisy_train_img = np.clip(noisy_train_img, 0., 1.)
noisy_test_img = np.clip(noisy_test_img, 0., 1.)

# Build Denoising CAE
d_model = models.Sequential()
d_model.add(model)

d_model.compile(optimizer='adam', loss='binary_crossentropy')

# Train model
d_model.fit(noisy_train_img, train_img, epochs=100, batch_size=100,
            shuffle=True, validation_data=(noisy_test_img, test_img))

# Get Images
prediction = d_model.predict(noisy_test_img)

count = 0
while count < 10:
    img = prediction[count].reshape((28, 28))
    plt.imshow(img, cmap='gray')
    plt.title(f"Model Output {count}")
    plt.show()
    plt.imshow(noisy_test_img[count], cmap='gray')
    plt.title(f"Noisy Test Image {count}")
    plt.show()
    count = count + 1

d_model.save("modeld.h5")
################################################
# Step 4
################################################
# Take out encoder part of the trained model

# If using saved model:
# new_model = tf.keras.models.load_model('model.h5')
# layers = new_model.layers

# Build only the encoder portion of the CAE
layers = model.layers  # Get all layers from CAE
encoder = models.Sequential(layers[:-12])  # Remove last 12 layers

# Pick 500 random images from the test set
indices = random.sample(range(10000), 500)

features = encoder.predict(test_img)

# Plot latent features
for i in indices:
    feature = features[i]
    # Get 2D latent feature
    if test_label[i] == 0:  # Top
        plt.scatter(feature[0], feature[1], c='hotpink')  # Pink
    elif test_label[i] == 1:  # Trouser
        plt.scatter(feature[0], feature[1], c='green')  # Green
    elif test_label[i] == 2:  # Pullover
        plt.scatter(feature[0], feature[1], c='maroon')  # Maroon
    elif test_label[i] == 3:  # Dress
        plt.scatter(feature[0], feature[1], c='purple')  # Purple
    elif test_label[i] == 4:  # Coat
        plt.scatter(feature[0], feature[1], c='darkorange')  # Orange
    elif test_label[i] == 5:  # Sandal
        plt.scatter(feature[0], feature[1], c='blue')  # Blue
    elif test_label[i] == 6:  # Shirt
        plt.scatter(feature[0], feature[1], c='red')  # Red
    elif test_label[i] == 7:  # Sneaker
        plt.scatter(feature[0], feature[1], c='darkblue')  # Dark Blue
    elif test_label[i] == 8:  # Bag
        plt.scatter(feature[0], feature[1], c='teal')  # Teal
    else:  # Ankle boot
        plt.scatter(feature[0], feature[1], c='cyan')  # Cyan

plt.show()

# PCA
pca = PCA(n_components=2)

# Plot PCA
for i in indices:
    img = np.reshape(test_img[i], (28, 28))
    principalComponents = pca.fit_transform(img)
    for feature in principalComponents:
        if test_label[i] == 0:  # Top
            plt.scatter(feature[0], feature[1], c='hotpink')  # Pink
        elif test_label[i] == 1:  # Trouser
            plt.scatter(feature[0], feature[1], c='green')  # Green
        elif test_label[i] == 2:  # Pullover
            plt.scatter(feature[0], feature[1], c='maroon')  # Maroon
        elif test_label[i] == 3:  # Dress
            plt.scatter(feature[0], feature[1], c='purple')  # Purple
        elif test_label[i] == 4:  # Coat
            plt.scatter(feature[0], feature[1], c='darkorange')  # Orange
        elif test_label[i] == 5:  # Sandal
            plt.scatter(feature[0], feature[1], c='blue')  # Blue
        elif test_label[i] == 6:  # Shirt
            plt.scatter(feature[0], feature[1], c='red')  # Red
        elif test_label[i] == 7:  # Sneaker
            plt.scatter(feature[0], feature[1], c='darkblue')  # Dark Blue
        elif test_label[i] == 8:  # Bag
            plt.scatter(feature[0], feature[1], c='teal')  # Teal
        else:  # Ankle boot
            plt.scatter(feature[0], feature[1], c='cyan')  # Cyan

plt.show()
