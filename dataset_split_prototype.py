from sklearn.model_selection import train_test_split
import numpy as np

# Step 1: Create a mock dataset
# Suppose we have 100 sample images in each class (total 3 classes)
num_classes = 3
num_images_per_class = 100
data = []
labels = []

for i in range(num_classes):
    for j in range(num_images_per_class):
        # Each "image" is represented as a 64x64 array of random values (replace with actual image data)
        data.append(np.random.rand(64, 64, 3))
        labels.append(i)  # Label the image with the class index

data = np.array(data)
labels = np.array(labels)

# Step 2: Split into training, validation, and test sets
# Initial split: 85% for training + validation, 15% for testing
data_train_val, data_test, labels_train_val, labels_test = train_test_split(
    data, labels, test_size=0.15, stratify=labels, random_state=42
)

# Secondary split: 70% for training, 15% for validation (out of 85% of the original data)
data_train, data_val, labels_train, labels_val = train_test_split(
    data_train_val, labels_train_val, test_size=0.1765, stratify=labels_train_val, random_state=42
)  # 0.1765 * 85% ≈ 15%

# Output the dataset sizes
print("Training set size:", data_train.shape)
print("Validation set size:", data_val.shape)
print("Testing set size:", data_test.shape)

# Display the number of samples per class in each set to ensure proportions are maintained
unique, counts_train = np.unique(labels_train, return_counts=True)
print("Training set class distribution:", dict(zip(unique, counts_train)))

unique, counts_val = np.unique(labels_val, return_counts=True)
print("Validation set class distribution:", dict(zip(unique, counts_val)))

unique, counts_test = np.unique(labels_test, return_counts=True)
print("Testing set class distribution:", dict(zip(unique, counts_test)))
