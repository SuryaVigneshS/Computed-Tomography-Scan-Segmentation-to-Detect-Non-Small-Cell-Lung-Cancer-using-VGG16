import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Define image parameters
IMG_HEIGHT, IMG_WIDTH = 224, 224
BATCH_SIZE = 8

# Data augmentation and normalization as 80% Training and 20% validation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    'data/',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'
)

validation_generator = train_datagen.flow_from_directory(
    'data/',
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Load the VGG16 model without the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))
base_model.trainable = False  # Freeze the base model

# Add custom layers on top
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(train_generator.num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
EPOCHS = 200

history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator
)

# Evaluate model
val_loss, val_accuracy = model.evaluate(validation_generator)
print(f"\nValidation Accuracy: {val_accuracy * 100:.2f}%")

# Predictions
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)

# Confusion matrix
cm = confusion_matrix(validation_generator.classes, y_pred)

# Classification report
class_labels = list(validation_generator.class_indices.keys())
report = classification_report(validation_generator.classes, y_pred, target_names=class_labels)

# Save metrics to a text file
metrics_dir = 'metrics'
os.makedirs(metrics_dir, exist_ok=True)
metrics_path = os.path.join(metrics_dir, 'evaluation_metrics.txt')

with open(metrics_path, 'w') as f:
    f.write("Validation Accuracy: {:.2f}%\n\n".format(val_accuracy * 100))
    f.write("Classification Report:\n")
    f.write(report)
    f.write("\nConfusion Matrix:\n")
    for row in cm:
        f.write(' '.join(map(str, row)) + '\n')

print(f"Evaluation metrics saved to {metrics_path}")

# Plot and save confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=class_labels,
            yticklabels=class_labels, cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
confusion_matrix_path = os.path.join(metrics_dir, 'confusion_matrix.png')
plt.savefig(confusion_matrix_path)
plt.close()
print(f"Confusion matrix plot saved to {confusion_matrix_path}")

# Plot training & validation accuracy/loss
plt.figure(figsize=(12, 4))
print(history.history.keys())

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history.get('accuracy', []), label='Train Accuracy')
plt.plot(history.history.get('val_accuracy', []), label='Validation Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history.get('loss', []), label='Train Loss')
plt.plot(history.history.get('val_loss', []), label='Validation Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
training_plot_path = os.path.join(metrics_dir, 'training_metrics.png')
plt.savefig(training_plot_path)
plt.close()
print(f"Training metrics plot saved to {training_plot_path}")


# Save the trained model
model_dir = 'model'
os.makedirs(model_dir, exist_ok=True)
model_path = os.path.join(model_dir, 'vgg16_model.h5')
model.save(model_path)
print(f"Trained model saved to {model_path}")