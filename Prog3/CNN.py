# CS 510 CV and DL. Summer 2023. Prog 3. Cera Oh
import tensorflow as tf
from keras import metrics
from keras import layers
from keras import models
import matplotlib.pyplot as plt
import keras_preprocessing
import numpy as np
import matplotlib as mpl
from sklearn.metrics import confusion_matrix

#############################################
# Step 1
#############################################
pre_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
    weights="imagenet", include_top=False, input_shape=(150, 150, 3))

# pre_model.summary() # Turned off for faster processing

# Visualize first layer filters:
# Code reference: https://machinelearningmastery.com/how-to-visualize-filters-and-feature-maps-in-convolutional-neural-networks/
first = pre_model.get_layer(index=1)  # get the first conv layer
filters = first.get_weights()  # get filter values
f_min = np.min(filters)
f_max = np.max(filters)
filters = (filters - f_min)/(f_max-f_min)  # normalize filter values

for index in range(32):
    f = filters[:, :, :, :, index]  # get filter at index
    for ch in range(3):
        plot = np.zeros((3, 3))
        channel = f[:, :, ch]  # get channel
        img = channel[0]
        row_count = 0
        for row in img:
            col_count = 0
            for col in row:
                plot[row_count][col_count] = col
                col_count = col_count + 1
            row_count = row_count + 1
        # plt.imshow(plot, cmap='gray')  # Turned off for faster processing. Plots 1 graph per filter and per channel
        # plt.title(f'Filter { index} Channel {ch}')
        # plt.show()

#########################################
# Step 2:
#########################################
# Load dataset and resize images to (150,150,3):
training_set = tf.keras.utils.image_dataset_from_directory(
    './cats_dogs_dataset/dataset/training_set',
    labels='inferred',
    label_mode='binary',
    class_names=['cats', 'dogs'],
    color_mode="rgb",
    batch_size=10,
    image_size=(150, 150),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=True,
)

test_set = tf.keras.utils.image_dataset_from_directory(
    './cats_dogs_dataset/dataset/test_set',
    labels='inferred',
    label_mode='binary',
    class_names=['cats', 'dogs'],
    color_mode="rgb",
    batch_size=10,
    image_size=(150, 150),
    shuffle=True,
    seed=None,
    validation_split=None,
    subset=None,
    interpolation="bilinear",
    follow_links=False,
    crop_to_aspect_ratio=True,
)

# Preprocess dataset:
normalization_layer = tf.keras.layers.Rescaling(1/255)
training_set_normalized = training_set.map(
    lambda x, y: (normalization_layer(x), y))
test_set_normalized = test_set.map(lambda x, y: (normalization_layer(x), y))

# Grabbing Labels
train_labels = np.zeros((800, 10, 1))
test_labels = np.zeros((200, 10, 1))

count = 0
for img, label in training_set_normalized.map(lambda x, y: (x, y)):
    train_labels[count] = label
    count = count + 1

count = 0
for img, label in test_set_normalized.map(lambda x, y: (x, y)):
    test_labels[count] = label
    count = count + 1

train_labels = train_labels.reshape(-1)
train_labels = train_labels.reshape((8000, 1))
test_labels = test_labels.reshape(-1)
test_labels = test_labels.reshape((2000, 1))

#################################################
# Step 3
#################################################
# Build model with transfer head
model = models.Sequential()
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))
# model.summary() # Turned off for faster processing

# Freeze pre-trained model weights:
pre_model.trainable = False

################################################
# Step 4
################################################
# (i) Evaluate transfer model on test dataset
prediction = model.predict(test_set_normalized)

# Test Accuracy and Confusion Matrix calculations:
count = 0
TN, TP, FN, FP = 0, 0, 0, 0
for value in prediction:
    if value < 0.5:  # Predicted cat
        if test_labels[count] == 0:  # True label is cat
            TN = TN + 1
        else:  # True label is dog
            FN = FN + 1
    else:  # Predicted dog
        if test_labels[count] == 0:  # True label is cat
            FP = FP + 1
        else:  # True label is dog
            TP = TP + 1
    count = count + 1

# Confusion matrix
confusion_matrix = np.array([[TP, FP], [FN, TN]])
print(f'Confusion Matrix: \n {confusion_matrix}')

# Overall accuracy
accuracy = (TP + TN)/(FN+TN+FP+TP)
print(f'Accuracy: {accuracy}')

# (ii) Train transfer model using binary cross entropy loss on data until approx convergence. Using RMSprop as optimizer.
model.compile(
    optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives()])

# Train model:
model.fit(training_set_normalized, epochs=5, batch_size=10,
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)])

# Testing the trained model:
loss, accuracy, TP, FP, TN, FN = model.evaluate(test_set_normalized)
# Per-epoch test loss for the transfer model:
print(f'Loss: {loss}')
# Accuracy
print(f'Accuracy: {accuracy}')
# Confusion matrix
confusion_matrix = np.array([[TP, FP], [FN, TN]])
print(f'Confusion Matrix: \n {confusion_matrix}')

# (iii) Perform the same as (i) and (ii) on sub-network of pre-train network.
# Unfreeze weights for parameter counting
pre_model.trainable = True

# Subnet pre_model:
# Code reference credit to https://www.reddit.com/r/learnmachinelearning/comments/zi3zqb/how_to_remove_layers_of_keras_functional_model/
subnet = models.Sequential()
start = pre_model.layers[0].input
end = pre_model.layers[19].output
cut = tf.keras.Model(
    inputs=start, outputs=end)
subnet.add(cut)
# subnet.summary() # Turned off for faster processing

# Attach transfer head:
new_model = models.Sequential()
new_model.add(subnet)
new_model.add(layers.Flatten())
new_model.add(layers.Dense(256, activation='relu'))
new_model.add(layers.Dense(1, activation='sigmoid'))
# new_model.summary() # Turned off for faster processing

# Freeze weights on the sub-network and attach a transfer head as before
subnet.trainable = False
pre_model.trainable = False

# Subnet prediction:
prediction = new_model.predict(test_set_normalized)

# Test Accuracy and Confusion Matrix calculations:
count = 0
TN, TP, FN, FP = 0, 0, 0, 0
for value in prediction:
    if value < 0.5:  # Predicted cat
        if test_labels[count] == 0:  # True label is cat
            TN = TN + 1
        else:  # True label is dog
            FN = FN + 1
    else:  # Predicted dog
        if test_labels[count] == 0:  # True label is cat
            FP = FP + 1
        else:  # True label is dog
            TP = TP + 1
    count = count + 1

# Confusion matrix
confusion_matrix = np.array([[TP, FP], [FN, TN]])
print(f'Confusion Matrix: \n {confusion_matrix}')

# Overall accuracy
accuracy = (TP + TN)/(FN+TN+FP+TP)
print(f'Accuracy: {accuracy}')

# Train subnet model like (ii):
new_model.compile(
    optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.TruePositives(), tf.keras.metrics.FalsePositives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.FalseNegatives()])

new_model.fit(training_set_normalized, epochs=15, batch_size=10,
              callbacks=[tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)])

# Testing the trained model:
loss, accuracy, TP, FP, TN, FN = new_model.evaluate(test_set_normalized)

# Per-epoch test loss for the transfer model:
print(f'Loss: {loss}')

# Accuracy
print(f'Accuracy: {accuracy}')

# Confusion matrix
confusion_matrix = np.array([[TP, FP], [FN, TN]])
print(f'Confusion Matrix: \n {confusion_matrix}')
