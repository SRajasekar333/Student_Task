# -*- coding: utf-8 -*-
"""
Created on Wed Jul 10 23:59:58 2024

@author: rajas
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



# Function to convert polar coordinates to Cartesian and create a scatter plot
def process_csv_file(file_path, output_folder):
    # Load the CSV file
    data = pd.read_csv(file_path)
    
    # Convert timestamp to datetimamp'e for better handling
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    
    # Convert Elevation from degrees to radians
    data['Elevation_rad'] = np.deg2rad(data['Elevation'])
    
    # Transform polar coordinates (Range Gate, Elevation) to Cartesian coordinates (x, y)
    data['x'] = data['Range Gate'] * np.cos(data['Elevation_rad'])
    data['y'] = data['Range Gate'] * np.sin(data['Elevation_rad'])
    
    # Create a scatter plot for all data in Cartesian coordinates
    plt.figure(figsize=(10, 6))
    sc = plt.scatter(data['x'], data['y'], c=data['Radial Velocity'], cmap='viridis', s=10)
    plt.colorbar(sc, label='Radial Velocity (m/s)')
    plt.title('Lidar Data in Cartesian Coordinates')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    
    # Save the plot as an image file with the same name as the CSV file
    output_file = os.path.join(output_folder, os.path.basename(file_path).replace('.csv', '.png'))
    plt.savefig(output_file)
    plt.close()

    print(f"Plot saved as {output_file}.")

# Create an output folder to save the images
output_folder = os.path.join(mini_data_folder_path, 'output_images')
os.makedirs(output_folder, exist_ok=True)

# Iterate through all CSV files in the mini_data folder
for file_name in os.listdir(mini_data_folder_path):
    if file_name.endswith('.csv'):
        file_path = os.path.join(mini_data_folder_path, file_name)
        process_csv_file(file_path, output_folder)
