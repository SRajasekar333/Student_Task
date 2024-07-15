# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 19:36:20 2024

@author: rajas
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import os

# Load the predictions CSV file
predictions_file_path = "vortex_center_predictions.csv"  # Replace with your file path
predictions_df = pd.read_csv(predictions_file_path)

# Select any 5 image names from the predictions
selected_images = predictions_df['Image Name'].sample(5).tolist()
print(selected_images)

# Directory containing the image data CSV files
data_dir = "C:/Secondary/Prof.Farber/KIWI_Yolo2/KIWI_Yolo/Data/data/data/"

# Function to extract data from the corresponding CSV file for a given image name
def get_image_data(image_name):
    file_path = os.path.join(data_dir, f"{image_name}.csv")
    if os.path.exists(file_path):
        data = pd.read_csv(file_path)
        # Convert timestamp to datetime for better handling
        data['Timestamp'] = pd.to_datetime(data['Timestamp'])
        # Convert Elevation from degrees to radians
        data['Elevation_rad'] = np.deg2rad(data['Elevation'])
        # Transform polar coordinates (Range Gate, Elevation) to Cartesian coordinates (x, y)
        data['x'] = data['Range Gate'] * np.cos(data['Elevation_rad'])
        data['y'] = data['Range Gate'] * np.sin(data['Elevation_rad'])
        return data
    else:
        raise FileNotFoundError(f"CSV file for image {image_name} not found.")
        
# Function to extract centers for a given image name
def get_centers(image_name):
    row = predictions_df[predictions_df['Image Name'] == image_name]
    if not row.empty:
        actual_port = (row['Actual Port X'].values[0], row['Actual Port Y'].values[0])
        predicted_port = (row['Predicted Port X'].values[0], row['Predicted Port Y'].values[0])
        actual_starboard = (row['Actual Starboard X'].values[0], row['Actual Starboard Y'].values[0])
        predicted_starboard = (row['Predicted Starboard X'].values[0], row['Predicted Starboard Y'].values[0])
        return actual_port, predicted_port, actual_starboard, predicted_starboard
    else:
        raise ValueError(f"No prediction data found for image {image_name}")

# Function to plot the centers and bounding boxes
def plot_vortex_centers(image_name, data, actual_port, predicted_port, actual_starboard, predicted_starboard):
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(data['x'], data['y'], c=data['Radial Velocity'], cmap='viridis', s=10)
    plt.colorbar(sc, label='Radial Velocity (m/s)')
    
    plt.scatter(*actual_starboard, color='black', label='Actual Starboard', marker='x', zorder=5)
    plt.scatter(*predicted_starboard, color='blue', label='Predicted Starboard', marker='o', zorder=5)
    plt.scatter(*actual_port, color='red', label='Actual Port', marker='x', zorder=5)
    plt.scatter(*predicted_port, color='yellow', label='Predicted Port', marker='o', zorder=5)

    def add_bounding_box(ax, point, color):
        size = 20  # Adjust the size of the bounding box
        alpha_gradient = np.linspace(0.1, 0.5, 10)
        for i, alpha in enumerate(alpha_gradient):
            rect = Rectangle((point[0] - size/2, point[1] - size/2), size, size, linewidth=1, edgecolor=color, facecolor='none', alpha=alpha, zorder=5)
            ax.add_patch(rect)
            size -= 0.02

    ax = plt.gca()
    add_bounding_box(ax, actual_starboard, 'black')
    add_bounding_box(ax, predicted_starboard, 'blue')
    add_bounding_box(ax, actual_port, 'red')
    add_bounding_box(ax, predicted_port, 'yellow')

    # Set the aspect of the plot to be equal
    plt.gca().set_aspect('equal', adjustable='box')

    plt.title(f'Vortex Center Prediction vs Actual for {image_name}')
    plt.xlabel('Lateral Distance from Lidar (m)')
    plt.ylabel('Height Above Lidar (m)')
    plt.legend()

    # Save the plot as an image file
    output_file = f'cartesian_lidar_data_{image_name}.png'
    plt.savefig(output_file)
    plt.show()

    print(f"Plot saved as {output_file}.")

# Process each selected image
for image_name in selected_images:
    # Extract centers from predictions CSV
    actual_port, predicted_port, actual_starboard, predicted_starboard = get_centers(image_name)
    # Load corresponding data
    data = get_image_data(image_name)
    # Plot the data
    plot_vortex_centers(image_name, data, actual_port, predicted_port, actual_starboard, predicted_starboard)
