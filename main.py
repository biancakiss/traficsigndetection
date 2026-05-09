from datasets import load_dataset
import numpy as np
import cv2
import tensorflow as tf
from keras import Sequential
from keras.src.callbacks import EarlyStopping
from keras.src.layers import Rescaling, Conv2D, MaxPooling2D, Dense, Flatten, Dropout

ds = load_dataset("keremberke/german-traffic-sign-detection", name="full")

print(ds)
# DatasetDict with train / validation / test splits

# Check splits and structure
print(ds)
print(ds["train"].features)

# Look at a single example
example = ds["train"][0]
print(example.keys())
# -> image, image_id, width, height, objects (with bboxes + labels)

# The bounding boxes
print(example["objects"])
# {'bbox': [[x, y, w, h], ...], 'category': [...], 'id': [...], ...}

import matplotlib.pyplot as plt
import matplotlib.patches as patches

def show_example(example):
    img = example["image"]
    fig, ax = plt.subplots(1, figsize=(10, 8))
    ax.imshow(img)

    for bbox, label in zip(example["objects"]["bbox"], example["objects"]["category"]):
        x, y, w, h = bbox
        rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor="red", facecolor="none")
        ax.add_patch(rect)
        ax.text(x, y - 5, str(label), color="red", fontsize=10, fontweight="bold")

    plt.axis("off")
    plt.show()

# show_example(ds["train"][0])

IMAGE_SIZE = (224, 224)

def extract_crops(split):
    images = []
    labels = []

    for example in split:
        img = np.array(example["image"].convert("RGB"))
        objects = example["objects"]

        for bbox, category in zip(objects["bbox"], objects["category"]):
            x, y, w, h = [int(v) for v in bbox]

            # Clamp to image bounds
            x2, y2 = min(x + w, img.shape[1]), min(y + h, img.shape[0])
            crop = img[y:y2, x:x2]

            if crop.size == 0:
                continue

            # Resize to standard size
            crop = cv2.resize(crop, IMAGE_SIZE)
            images.append(crop)
            labels.append(category)

    return np.array(images), np.array(labels)

print("Processing train split...")
X_train, y_train = extract_crops(ds["train"])

print("Processing validation split...")
X_val, y_val = extract_crops(ds["validation"])

print("Processing test split...")
X_test, y_test = extract_crops(ds["test"])

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# Get unique class names from the dataset features
class_names = ds["train"].features["objects"].feature["category"].names

plt.figure(figsize=(12, 10))
for i in range(25):
    ax = plt.subplot(5, 5, i + 1)
    plt.imshow(X_train[i])
    plt.title(class_names[y_train[i]], fontsize=8)
    plt.axis("off")
plt.tight_layout()
plt.show()

num_classes = len(class_names)
print(f"Number of classes: {num_classes}")

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal", input_shape=(224, 224, 3)),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.2),
])

model = Sequential([
    data_augmentation,
    Rescaling(1./255),                          # normalize pixels to [0,1]

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')    # output layer
])

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer='adam',
    metrics=['accuracy']
)

model.summary()

# Convert to tf.data.Dataset for efficiency
def make_dataset(X, y, batch_size=32, shuffle=False):
    ds = tf.data.Dataset.from_tensor_slices((X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(X))
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = make_dataset(X_train, y_train, shuffle=True)
val_ds   = make_dataset(X_val,   y_val)

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    callbacks=[early_stop]
)

# Plot Loss
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'],     label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.title('Loss')

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'],     label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()

# Final evaluation on test set
test_ds = make_dataset(X_test, y_test)
test_loss, test_acc = model.evaluate(test_ds)
print(f"Test Accuracy: {test_acc:.4f}")

model.save("traffic_sign_model.keras")
# Load later with:
# model = tf.keras.models.load_model("traffic_sign_model.keras")