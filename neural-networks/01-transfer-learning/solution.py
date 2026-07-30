import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models, optimizers
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Config
BATCH_SIZE = 32
IMG_SIZE = 224
EPOCHS = 15

print("Loading Oxford Flowers...")
# Load data
(train_ds, val_ds, test_ds), ds_info = tfds.load(
    'oxford_flowers102',
    split=['train', 'validation', 'test'],
    with_info=True,
    as_supervised=True
)

# Preprocessing
def preprocess(img, label):
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    return img, label

train_ds = train_ds.map(preprocess).shuffle(1000).batch(BATCH_SIZE)
val_ds = val_ds.map(preprocess).batch(BATCH_SIZE)
test_ds = test_ds.map(preprocess).batch(BATCH_SIZE)

# --- EXPERIMENT 1: Custom CNN (Results were bad, ~40% acc) ---
# model = models.Sequential([
#     layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
#     layers.MaxPooling2D((2, 2)),
#     layers.Conv2D(64, (3, 3), activation='relu'),
#     layers.MaxPooling2D((2, 2)),
#     layers.Flatten(),
#     layers.Dense(64, activation='relu'),
#     layers.Dense(102, activation='softmax')
# ])
# model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# model.fit(train_ds, epochs=10, validation_data=val_ds)

# --- EXPERIMENT 2: VGG16 Transfer Learning ---
print("Setting up VGG16...")

base = VGG16(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, 3))

# Freeze base model
base.trainable = False

model = models.Sequential([
    base,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5), # Added dropout to reduce overfitting
    layers.Dense(102, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

print("Training frozen model...")
history_frozen = model.fit(train_ds, epochs=10, validation_data=val_ds)

# --- Fine Tuning ---
print("Unfreezing last block for fine tuning...")
base.trainable = True

# Freeze everything except the last block (block5)
for layer in base.layers:
    if 'block5' not in layer.name:
        layer.trainable = False

# Recompile with low LR
model.compile(optimizer=optimizers.Adam(1e-5), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

history_ft = model.fit(train_ds, epochs=10, validation_data=val_ds)

# Plotting results
acc = history_frozen.history['val_accuracy'] + history_ft.history['val_accuracy']
loss = history_frozen.history['val_loss'] + history_ft.history['val_loss']

plt.figure(figsize=(10,5))
plt.plot(acc, label='Validation Accuracy')
plt.plot(loss, label='Validation Loss')
plt.axvline(x=10, color='r', linestyle='--', label='Fine Tuning Start')
plt.legend()
plt.title("Training Progress")
plt.show()

# Final Eval
print("Evaluating on Test Set...")
test_loss, test_acc = model.evaluate(test_ds)
print(f"Final Accuracy: {test_acc}")

# Confusion Matrix
y_pred = []
y_true = []

for img, label in test_ds:
    pred = model.predict(img, verbose=0)
    y_pred.extend(np.argmax(pred, axis=1))
    y_true.extend(label.numpy())

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(12,12))
sns.heatmap(cm, cmap='viridis', square=True)
plt.show()