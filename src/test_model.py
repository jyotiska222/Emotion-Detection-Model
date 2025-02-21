import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load Model
model = tf.keras.models.load_model("./saved_model/emotion_model.h5")

# Data Preprocessing for Testing
IMG_SIZE = 48
BATCH_SIZE = 64
DATASET_PATH = "./dataset"

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    os.path.join(DATASET_PATH, 'test'),
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    color_mode='grayscale',
    class_mode='categorical'
)

# Evaluate Model
loss, accuracy = model.evaluate(test_generator)
print(f"Test Accuracy: {accuracy*100:.2f}%")