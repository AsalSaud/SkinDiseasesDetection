import os
import cv2
import numpy as np
from collections import Counter
import shutil

# Function to resize
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.resize(image, target_size)
    return image

if __name__ == "__main__":
    # Data directories
    train_dir = 'Dataset/Dataset/train'
    output_dir = 'balanced_dataset'

    # Load dataset
    images = []
    labels = []
    classes = os.listdir(train_dir)
    class_counts = Counter()

    for i, class_name in enumerate(classes):
        class_dir = os.path.join(train_dir, class_name)
        num_images = len(os.listdir(class_dir))
        class_counts[class_name] = num_images

        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            image = cv2.imread(image_path)
            if image is not None:
                image = preprocess_image(image)  
                images.append(image)
                labels.append(class_name)

    max_class_count = 1000  # Maximum number of images per class

    # Balance the dataset by oversampling each class
    balanced_images = []
    balanced_labels = []

    for class_name, count in class_counts.items():
        class_indices = [idx for idx, label in enumerate(labels) if label == class_name]
        class_images = [images[idx] for idx in class_indices]
        class_labels = [labels[idx] for idx in class_indices]

        # Oversample to ensure exactly 1000 images per class
        if count < max_class_count:
            num_to_generate = max_class_count - count
            oversampled_indices = np.random.choice(class_indices, size=num_to_generate, replace=True)
            oversampled_images = [images[idx] for idx in oversampled_indices]
            oversampled_labels = [labels[idx] for idx in oversampled_indices]

            class_images.extend(oversampled_images)
            class_labels.extend(oversampled_labels)
        elif count > max_class_count:
            class_images = class_images[:max_class_count]
            class_labels = class_labels[:max_class_count]

        balanced_images.extend(class_images)
        balanced_labels.extend(class_labels)

    # Save balanced dataset
    for i, (image, label) in enumerate(zip(balanced_images, balanced_labels)):
        class_output_dir = os.path.join(output_dir, label)
        os.makedirs(class_output_dir, exist_ok=True)
        image_name = f"{i}.jpg"
        image_path = os.path.join(class_output_dir, image_name)
        cv2.imwrite(image_path, image)

    print("Balanced dataset saved successfully at:", output_dir)