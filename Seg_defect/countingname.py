# -*- coding: utf-8 -*-
"""
Created on Mon May  6 15:05:10 2024

@author: susu
"""
import os
import json
from collections import defaultdict

def count_label_occurrences(directory, labels_to_count):
    # Initialize a dictionary to store label counts
    label_counts = defaultdict(int)
 
    # Iterate over all files in the specified directory
    for filename in os.listdir(directory):
        if filename.endswith('.json'):  # Process only .json files
            file_path = os.path.join(directory, filename)  # Construct the full file path

            # Read the content of the JSON file
            with open(file_path, 'r') as file:
                data = json.load(file)
                #print(f'data: {data}')

            # Extract labels from the JSON data
            if 'shapes' in data:
                shapes = data['shapes']
                #print(f'shapes: {shapes}')
                for shape in shapes:
                    label = shape.get('label')
                    
                    if label in labels_to_count:
                        label_counts[label] += 1
                        #print(f' label:{label}={label_counts[label]}')

    return label_counts

# Example usage:
directory_path = 'D:/MuStr811Total2172imgs/Val/json'
#'D:/NCU/Segmentation/lacey/6label_dataset' #'D:/NCU/Segmentation/YOLOv7/yolov7_org_dataset(70_20_10)/train/json'
#'D:/NCU/Segmentation/YOLOv7/yolov7_org_dataset/test/json'
#'D:/NCU/Segmentation/6label_dataset'  # Specify the directory containing the JSON files
#'D:/NCU/Segmentation/YOLOv7/TrainingModel/Wrinkle_MOD'
# List of labels (names) to count occurrences (adjust as needed)
labels_to_count = ['Crack', 'Wrinkle', 'Residue', 'Lacey Graphene', 'Nucleation', 'Multilayer of domain']

# Get the counts of label occurrences in the specified directory
label_counts = count_label_occurrences(directory_path, labels_to_count)

# Print the label counts
print("Label Occurrences Val dataset:")
for label, count in label_counts.items():
    print(f"{label}: {count}")


