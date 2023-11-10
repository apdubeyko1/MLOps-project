import os

import joblib
import numpy as np
import tensorflow as tf

PATH = "cats_and_dogs"

batch_size = 128
IMG_HEIGHT = 150
IMG_WIDTH = 150

test_dir = os.path.join(PATH, "test")

# Get number of files in each directory. The train and validation directories
# each have the subdirectories "dogs" and "cats".
total_test = len(os.listdir(test_dir))


# function to normalize test images (weird exception)
def rescale_image_test(image):
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = image / 255.0
    return image


# load test images
test_data_gen = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels=None,
    batch_size=batch_size,
    image_size=[IMG_WIDTH, IMG_HEIGHT],
    shuffle=False,
)

# normalize values
test_data_gen = test_data_gen.map(rescale_image_test)


# the test model
# load the model from disk
loaded_model = joblib.load("model.pkl")

answers = [
    1,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    1,
    1,
    0,
    0,
    1,
    1,
    1,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    1,
    1,
    0,
    1,
    1,
    1,
    1,
    0,
    1,
    0,
    1,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
]

correct = 0

probabilities_model = loaded_model.predict(test_data_gen)
probabilities_model = np.squeeze(probabilities_model)
probabilities = np.array(
    [1 if i >= 0.5 else 0 for i in probabilities_model], dtype=np.int8
)

for probability, answer in zip(probabilities, answers):
    if round(probability) == answer:
        correct += 1

percent_ident = (correct / len(answers)) * 100

passed_challenge = percent_ident >= 63

with open("output.csv", "w") as g:
    print(probabilities, end="\n", file=g)
    print(
        f"model correctly identified {round(percent_ident, 2)}% of the images",
        end="\n ",
        file=g,
    )
    if passed_challenge:
        print("You passed the challenge!", file=g)
    else:
        print(
            "You haven't passed yet",
            "Your model should identify at least 63% of the images",
            file=g,
        )
