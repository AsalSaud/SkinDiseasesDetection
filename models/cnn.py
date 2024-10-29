import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Function to preprocess an image
def preprocess_image(image):
    if isinstance(image, str):  # Check if the input is a file path
        image = cv2.imread(image)  # Load the image from file

    if image is None:
        print("Error: Unable to load the image.")
        return None

    if len(image.shape) < 3 or image.shape[2] < 3:
        print("Error: Input image should have at least 3 channels.")
        return None

    # Convert BGR to YUV color space
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, 0] = cv2.equalizeHist(image_yuv[:, :, 0])
    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)

    image = cv2.resize(image, (224, 224))

    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    image = image.astype(np.float32) / 255.0
    return image


# Data Augmentation
train_datagen = ImageDataGenerator(
    horizontal_flip=True,
    fill_mode='nearest',
    rotation_range=20,
    zoom_range=0.2,
    shear_range=0.2
)

# Load and preprocess dataset
train_dir = 'Dataset/Dataset/train'
test_dir = 'Dataset/Dataset/test'
classes = os.listdir(train_dir)
num_classes = len(classes)

# Load and preprocess dataset
def load_and_preprocess_dataset(directory):
    images = []
    labels = []
    for i, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = preprocess_image(image_path)
            if image is not None:
                images.append(image)
                labels.append(i)
    return np.array(images), np.array(labels)

train_images, train_labels = load_and_preprocess_dataset(train_dir)
test_images, test_labels = load_and_preprocess_dataset(test_dir)


# Model
model = Sequential([
    Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.2),  
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1)

# Train model with data augmentation and callbacks
history = model.fit(
    train_datagen.flow(train_images, train_labels, batch_size=32),
    epochs=30,
    validation_data=(test_images, test_labels),
    callbacks=[reduce_lr]
)

# Save the trained model
model_path = 'trained_cnn.h5'
model.save(model_path)
print("Model saved successfully.")
