# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 14:04:53 2024

@author: rajas
"""

import os
import pandas as pd
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# Load the CSV file
annotations = pd.read_csv('C:/Secondary/Prof.Farber/KIWI_Yolo2/KIWI_Yolo/Data/data/mini_data/sorted_annotations_2.csv')

# Remove '.csv' from scan_name to match image filenames
annotations['scan_name'] = annotations['scan_name'].str.replace('.csv', '')

# Directory containing the sorted images
image_dir = 'C:/Secondary/Prof.Farber/KIWI_Yolo2/KIWI_Yolo/Data/data/mini_data/sorted_images/'

# Function to load an image given its scan name
def load_image(scan_name):
    image_path = os.path.join(image_dir, scan_name + '.png')
    if os.path.exists(image_path):
        return cv2.imread(image_path)
    else:
        raise FileNotFoundError(f"Image {image_path} not found.")

# Prepare image and label data
image_data = []
label_data = []
image_names = []

for scan_name in annotations['scan_name'].unique():
    image = load_image(scan_name)
    labels = annotations[annotations['scan_name'] == scan_name][['lateral_distance_from_lidar [m]', 'height_above_lidar [m]']].values
    if len(labels) == 2:  # Ensure there are exactly 2 labels per image
        image_data.append(image)
        label_data.append(labels)
        image_names.append(scan_name)

image_data = np.array(image_data)
label_data = np.array(label_data).reshape(-1, 4)  # Reshape to have 4 values per image (2 centers)

# Split data into training and validation sets
X_train, X_val, y_train, y_val, train_names, val_names = train_test_split(
    image_data, label_data, image_names, test_size=0.2, random_state=42
)

# Define the CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_data.shape[1], image_data.shape[2], image_data.shape[3])),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(4)  # Predict 4 values (2 centers)
])

model.compile(optimizer=Adam(), loss='mean_squared_error', metrics=['mae'])

model.summary()

# Define a function to calculate IoU
def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x1_, y1_, x2_, y2_ = box2
    
    xi1 = max(x1, x1_)
    yi1 = max(y1, y1_)
    xi2 = min(x2, x2_)
    yi2 = min(y2, y2_)
    
    inter_area = max(xi2 - xi1, 0) * max(yi2 - yi1, 0)
    
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_ - x1_) * (y2_ - y1_)
    
    union_area = box1_area + box2_area - inter_area
    
    return inter_area / union_area

# Define a custom callback to calculate IoU after each epoch
class IoUCallback(tf.keras.callbacks.Callback):
    def __init__(self, X_val, y_val):
        self.X_val = X_val
        self.y_val = y_val
        self.ious = []

    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(self.X_val)
        ious = []
        for actual, predicted in zip(self.y_val, y_pred):
            actual_port_box = [actual[0] - 1, actual[1] - 1, actual[0] + 1, actual[1] + 1]
            actual_starboard_box = [actual[2] - 1, actual[3] - 1, actual[2] + 1, actual[3] + 1]
            predicted_port_box = [predicted[0] - 1, predicted[1] - 1, predicted[0] + 1, predicted[1] + 1]
            predicted_starboard_box = [predicted[2] - 1, predicted[3] - 1, predicted[2] + 1, predicted[3] + 1]
            
            iou_port = calculate_iou(actual_port_box, predicted_port_box)
            iou_starboard = calculate_iou(actual_starboard_box, predicted_starboard_box)
            
            ious.append((iou_port + iou_starboard) / 2)
        average_iou = np.mean(ious)
        self.ious.append(average_iou)
        print(f"Epoch {epoch + 1}: Average IoU: {average_iou}")

# Create an instance of the custom callback
iou_callback = IoUCallback(X_val, y_val)

# Train the model
history = model.fit(X_train, y_train, epochs=100, validation_data=(X_val, y_val), callbacks=[iou_callback])

# Save the trained model
model.save('vortex_center_model.h5')

# Save the IoU history
iou_history = iou_callback.ious

# Plot loss, MAE, and IoU curves
plt.figure(figsize=(18, 5))
plt.subplot(1, 3, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.ylim(0, 5000)
plt.title('Loss curve')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.ylim(0, 100)
plt.title('MAE curve')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(iou_history, label='Validation IoU')
plt.xlabel('Epoch')
plt.ylabel('IoU')
plt.title('Validation IoU curve')
plt.legend()

plt.show()

# Predict the centers on the validation set
y_pred = model.predict(X_val)

# Calculate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse = mean_squared_error(y_val, y_pred)
mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_val, y_pred)

print(f"Mean Absolute Error (MAE): {mae}")
print(f"Mean Squared Error (MSE): {mse}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R-squared (R²): {r2}")

# Create a DataFrame to save the image names and corresponding y_val, y_pred values, and IoU
results_df = pd.DataFrame({
    'Image Name': val_names,
    'Actual Port X': y_val[:, 0],
    'Actual Port Y': y_val[:, 1],
    'Actual Starboard X': y_val[:, 2],
    'Actual Starboard Y': y_val[:, 3],
    'Predicted Port X': y_pred[:, 0],
    'Predicted Port Y': y_pred[:, 1],
    'Predicted Starboard X': y_pred[:, 2],
    'Predicted Starboard Y': y_pred[:, 3]
})

# Save the results to a CSV file
results_df.to_csv('vortex_center_predictions.csv', index=False)

'''
# Plotting predictions vs actual values
plt.figure(figsize=(10, 5))
plt.scatter(y_val[:, 0], y_val[:, 1], color='blue', label='Actual Port')
plt.scatter(y_pred[:, 0], y_pred[:, 1], color='red', label='Predicted Port', marker='x')
plt.xlabel('Lateral Distance from Lidar [m]')
plt.ylabel('Height Above Lidar [m]')
plt.title('Actual vs Predicted Port Vortex Centers')
plt.legend()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(y_val[:, 2], y_val[:, 3], color='green', label='Actual Starboard')
plt.scatter(y_pred[:, 2], y_pred[:, 3], color='yellow', label='Predicted Starboard', marker='x')
plt.xlabel('Lateral Distance from Lidar [m]')
plt.ylabel('Height Above Lidar [m]')
plt.title('Actual vs Predicted Starboard Port Vortex Centers')
plt.legend()
plt.show()

# Function to plot the actual and predicted centers and bounding boxes on an image
def plot_vortex_centers(image, y_val, y_pred, title):
    actual_port = y_val[:2]
    actual_starboard = y_val[2:]
    predicted_port = y_pred[:2]
    predicted_starboard = y_pred[2:]

    fig, ax = plt.subplots(1)
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Plot actual centers
    ax.scatter(actual_port[0], actual_port[1], color='green', label='Actual Port', marker='x')
    ax.scatter(actual_starboard[0], actual_starboard[1], color='blue', label='Actual Starboard', marker='x')
    
    # Plot predicted centers
    ax.scatter(predicted_port[0], predicted_port[1], color='red', label='Predicted Port', marker='o')
    ax.scatter(predicted_starboard[0], predicted_starboard[1], color='yellow', label='Predicted Starboard', marker='o')
    
    # Plot actual bounding boxes
    actual_port_box = plt.Rectangle((actual_port[0] - 1, actual_port[1] - 1), 2, 2, linewidth=1, edgecolor='green', facecolor='none')
    actual_starboard_box = plt.Rectangle((actual_starboard[0] - 1, actual_starboard[1] - 1), 2, 2, linewidth=1, edgecolor='blue', facecolor='none')
    ax.add_patch(actual_port_box)
    ax.add_patch(actual_starboard_box)
    
    # Plot predicted bounding boxes
    predicted_port_box = plt.Rectangle((predicted_port[0] - 1, predicted_port[1] - 1), 2, 2, linewidth=1, edgecolor='red', facecolor='none')
    predicted_starboard_box = plt.Rectangle((predicted_starboard[0] - 1, predicted_starboard[1] - 1), 2, 2, linewidth=1, edgecolor='yellow', facecolor='none')
    ax.add_patch(predicted_port_box)
    ax.add_patch(predicted_starboard_box)
    
    plt.legend()
    plt.title(title)
    plt.show()

# Plot results for a few samples
for i in range(5):
    plot_vortex_centers(X_val[i], y_val[i], y_pred[i], f'Sample {i+1}')
    
'''
