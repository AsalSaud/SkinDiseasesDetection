import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

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


# MobileNet model
base_model = MobileNet(weights='imagenet', include_top=False)

# Find the index of the layer before the GlobalAveragePooling2D layer
for i, layer in enumerate(base_model.layers):
    if isinstance(layer, GlobalAveragePooling2D):
        break

# Freeze layers up to the layer before GlobalAveragePooling2D
for layer in base_model.layers[:i]:
    layer.trainable = False

# Add custom layers on top of MobileNetV1
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x) 
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the new model
model = Model(inputs=base_model.input, outputs=predictions)

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

# Evaluate model
loss, accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion Matrix
predictions = model.predict(test_images)
predicted_labels = np.argmax(predictions, axis=1)
cm = confusion_matrix(test_labels, predicted_labels)

# Display Confusion Matrix with red colors
plt.figure(figsize=(12, 10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(cmap='Reds')
plt.title('Confusion Matrix', fontsize=16)
plt.xlabel('Predicted Label', fontsize=14)
plt.ylabel('True Label', fontsize=14) 
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

# Save model
model.save("MobileNetV1.h5")

# Plotting accuracy and loss
plt.figure(figsize=(12, 5))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.show()