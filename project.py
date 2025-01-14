import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import os
import warnings
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from glob import glob
from collections import Counter

# Ignore warnings
warnings.filterwarnings('ignore')

# Set dataset paths
train_path = r"D:/DATASET/TRAIN"  # Update with your dataset path
test_path = r"D:/DATASET/TEST"    # Update with your dataset path

# Data visualization
x_data = []
y_data = []

# Loading images and labels
for category in glob(train_path + '/*'):
    for file in tqdm(glob(category + '/*')):
        img_array = cv2.imread(file)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        x_data.append(img_array)
        y_data.append(category.split("\\")[-1])  # Updated for Windows path separator

# Create a DataFrame
data = pd.DataFrame({'image': x_data, 'label': y_data})
print("Dataset Shape:", data.shape)
print(Counter(y_data))

# Pie chart for visualization
colors = ['#a0d157', '#c48bb8']
plt.pie(data['label'].value_counts(), startangle=90, explode=[0.05, 0.05],
        autopct='%0.2f%%', labels=['Organic', 'Recyclable'], colors=colors, radius=2)
plt.show()

# Display random images
plt.figure(figsize=(20, 15))
for i in range(9):
    plt.subplot(3, 3, i + 1)
    index = np.random.randint(len(x_data))
    plt.title(f'This image is of {y_data[index]}', fontdict={'size': 12, 'weight': 'bold'})
    plt.imshow(x_data[index])
    plt.axis('off')
plt.tight_layout()
plt.show()

# Number of classes
class_names = glob(train_path + '/*')
number_of_classes = len(class_names)
print("Number Of Classes:", number_of_classes)

# CNN Model Definition
model = Sequential([
    Conv2D(32, (3, 3), input_shape=(224, 224, 3)),
    Activation("relu"),
    MaxPooling2D(),
    
    Conv2D(64, (3, 3)),
    Activation("relu"),
    MaxPooling2D(),
    
    Conv2D(128, (3, 3)),
    Activation("relu"),
    MaxPooling2D(),
    
    Flatten(),
    Dense(256),
    Activation("relu"),
    Dropout(0.5),
    
    Dense(64),
    Activation("relu"),
    Dropout(0.5),
    
    Dense(number_of_classes),  # Output layer
    Activation("softmax")
])

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Model summary
plot_model(model, to_file="model_plot.png", show_shapes=True, show_layer_names=True)
model.summary()

# Image Data Generator
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

test_generator = test_datagen.flow_from_directory(
    test_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical"
)

# Train the model
hist = model.fit(
    train_generator,
    epochs=20,
    validation_data=test_generator
)

# Plot training history
plt.figure(figsize=[10, 6])
plt.plot(hist.history["accuracy"], label="Train Accuracy")
plt.plot(hist.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.show()

plt.figure(figsize=[10, 6])
plt.plot(hist.history["loss"], label="Train Loss")
plt.plot(hist.history["val_loss"], label="Validation Loss")
plt.legend()
plt.show()

# Prediction Function
def predict_func(img_path):
    img = cv2.imread(img_path)
    plt.figure(figsize=(6, 4))
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.tight_layout()
    img = cv2.resize(img, (224, 224))
    img = np.reshape(img, [-1, 224, 224, 3]) / 255.0
    result = np.argmax(model.predict(img))
    class_labels = {v: k for k, v in train_generator.class_indices.items()}
    print(f"This image -> {class_labels[result]}")

# Test predictions
test_img_1 = r"D:/DATASET/TEST/R/o_12579.jpg"  # Update path
predict_func(test_img_1)

test_img_2 = r"D:/DATASET/TEST/R/R_10760.jpg"  # Update path
predict_func(test_img_2)
