import os

import joblib
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

PATH = "cats_and_dogs"

train_dir = os.path.join(PATH, "train")
validation_dir = os.path.join(PATH, "validation")

# Get number of files in each directory. The train and validation directories
# each have the subdirectories "dogs" and "cats".
total_train = sum([len(files) for r, d, files in os.walk(train_dir)])
total_val = sum([len(files) for r, d, files in os.walk(validation_dir)])

# Variables for pre-processing and training.
batch_size = 128
epochs = 5
IMG_HEIGHT = 150
IMG_WIDTH = 150


# function to normalize pixel values
def rescale_image(image, label=None):
    # This already normalizes it between 0 and 1
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    return image, label


# load training data
train_data_gen = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels="inferred",
    label_mode="binary",
    image_size=[IMG_WIDTH, IMG_HEIGHT],
    interpolation="nearest",
    batch_size=batch_size,
    shuffle=True,
)


# load validation data
val_data_gen = tf.keras.utils.image_dataset_from_directory(
    validation_dir,
    labels="inferred",
    label_mode="binary",
    image_size=[IMG_WIDTH, IMG_HEIGHT],
    interpolation="nearest",
    batch_size=batch_size,
    shuffle=True,
)


# for image, label in train_data_gen.take(1):
#    print("Pixel values of the first image:")
#    print(image[0].numpy())

# normalize values
train_data_gen = train_data_gen.map(rescale_image)
val_data_gen = val_data_gen.map(rescale_image)

# data augmentation
train_image_generator = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    rescale=1.0 / 255.0,
)

train_data_gen = train_image_generator.flow_from_directory(
    batch_size=batch_size,
    directory=train_dir,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    class_mode="binary",
)

# create and compile model
model = Sequential(
    [
        Conv2D(16, (3, 3), activation="relu", input_shape=(150, 150, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation="relu"),
        Dense(1, activation="sigmoid"),
    ]
)

model.summary()
model.compile("adam", loss="binary_crossentropy", metrics=["accuracy"])

# fit the model
history = model.fit(train_data_gen, epochs=epochs, valid_data=val_data_gen)

# plot the training data
acc = history.history["accuracy"]
val_acc = history.history["val_accuracy"]

loss = history.history["loss"]
val_loss = history.history["val_loss"]

epochs_range = range(epochs)

# save the model to disk
joblib.dump(model, "model.pkl")

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label="Training Accuracy")
plt.plot(epochs_range, val_acc, label="Validation Accuracy")
plt.legend(loc="lower right")
plt.title("Training and Validation Accuracy")

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label="Training Loss")
plt.plot(epochs_range, val_loss, label="Validation Loss")
plt.legend(loc="upper right")
plt.title("Training and Validation Loss")
plt.show()
