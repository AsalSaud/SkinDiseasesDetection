from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
import os
import cv2
import pickle

train_dir = 'Dataset/Dataset/train'
test_dir = 'Dataset/Dataset/test'
classes=[ 'Bullous','Eczema','Hair Loss','Light Diseases','Melanoma','Nail Fungus','Acne and Rosacea','Actinic Keratosis',
'Psoriasis','Seborrheic Keratoses','Tinea Ringworm','Warts Molluscum']


# Function to preprocess an image
def preprocess_image(image):
    if isinstance(image, str): # Check if the input is a file path
     image = cv2.imread(image) # Load
    if image is None:
      print("Error: Unable to load the image.")
      return None
    if len(image.shape) < 3 or image.shape[2] < 3:
     print("Error: Input image should have at least 3 channels.")
     return None
# Convert BGR to YUV color space
    image_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    image_yuv[:, :, -1] = cv2.equalizeHist(image_yuv[:, :, -1])
    image = cv2.cvtColor(image_yuv, cv2.COLOR_YUV2RGB)
    image = cv2.resize(image, (224, 224))
    if len(image.shape) == 2:
     image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    image = image.astype(np.float32) / 255.0
    return image

#Load and preprocess dataset
def load_and_preprocess_dataset(directory):
   images = []
   labels = []
   for i, class_name in enumerate (classes):
     class_dir = os.path.join(directory, class_name)
     for image_name in os.listdir(class_dir):
         image_path = os.path.join(class_dir, image_name)
         image = preprocess_image(image_path)
         if image is not None:
            images.append(image)
            labels.append(i)
   return np.array (images), np.array (labels)

train_images,train_labels = load_and_preprocess_dataset(train_dir)
test_images, test_labels = load_and_preprocess_dataset(test_dir)
train_images = train_images.reshape(train_images.shape[0], -1)
test_images = test_images.reshape(test_images.shape[0], -1)


# train the KNN classifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(train_images,train_labels)


# Predict labels for test set
test_predictions = knn.predict(test_images)
Train_predictions = knn.predict(train_images)

# Evaluate accuracy on Train set
Train_accuracy = accuracy_score(train_labels, Train_predictions)
print("Train Accuracy:", Train_accuracy*100)
# Evaluate accuracy on test set
test_accuracy = accuracy_score(test_labels, test_predictions)
print("Test Accuracy:", test_accuracy*100)
