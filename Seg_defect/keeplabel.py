# -*- coding: utf-8 -*-
"""
Created on Thu Jul 11 14:44:50 2024

@author: susu
"""
import json
import shutil
import os
#import glob

def filter_labels_in_files(input_dir, output_dir, label_to_keep, label1_to_keep, label2_to_keep):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all JSON files in the input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    for json_file in json_files:
        # Derive corresponding image file path
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        image_file = os.path.join(input_dir, f"{base_name}.jpg")
        
        
        if not os.path.exists(image_file):
            print(f"Image file for {json_file} not found, skipping.")
            image_file = os.path.join(input_dir, f"{base_name}.png")
            if not os.path.exists(image_file):
                print(f"Image file for {json_file} not found, skipping.")
                image_file = os.path.join(input_dir, f"{base_name}.bmp")
                if not os.path.exists(image_file):
                    print(f"Image file for {json_file} not found, skipping.")
                    continue
        
        # Define new file paths
        str_new_json_path = f"{label_to_keep}_{label1_to_keep}_{label2_to_keep}_{base_name}.json"
        str_new_image_path = f"{label_to_keep}_{label1_to_keep}_{label2_to_keep}_{base_name}.jpg"
        
        if label1_to_keep == 'I' and label2_to_keep == 'I':
            str_new_json_path = f"{label_to_keep}_{base_name}_1.json"
            str_new_image_path = f"{label_to_keep}_{base_name}_1.jpg"
            #print('II')
        
        elif label2_to_keep == 'I':
            str_new_json_path = f"{label_to_keep}_{label1_to_keep}_{base_name}_1.json"
            str_new_image_path = f"{label_to_keep}_{label1_to_keep}_{base_name}_1.jpg"
            #print('I')
            
        # Define new paths for the copied files
        new_json_path = os.path.join(output_dir, str_new_json_path)
        new_image_path = os.path.join(output_dir, str_new_image_path)
        
        # Step 1: Read the existing JSON file
        with open(os.path.join(input_dir, json_file), 'r') as file:
            data = json.load(file)
        
        # Check if the desired label is in the JSON file
        labels = [shape['label'] for shape in data['shapes']]
       
       # print(f"Labels found in {json_file}: {labels} , {len(labels)}")
        
        if label1_to_keep == 'I':
            if label_to_keep not in labels:
               # print(f"Label I '{label_to_keep}' not found in {json_file}, skipping.")
                continue
        else:
            if label_to_keep not in labels or label1_to_keep not in labels:
               # print(f"Label '{label_to_keep}' or '{label1_to_keep}' not found in {json_file}, skipping.")
                continue
        
        # Step 2: Copy the image and JSON file to new locations
        shutil.copyfile(image_file, new_image_path)
        shutil.copyfile(os.path.join(input_dir, json_file), new_json_path)
        #print(f"Copied files to {new_json_path} and {new_image_path}")
        
        # Step 3: Modify the copied JSON file to keep only the desired labels       
        if label_to_keep != 'I' and label1_to_keep != 'I' and label2_to_keep != 'I':
            new_shapes = [shape for shape in data['shapes'] if shape['label'] in [label_to_keep, label1_to_keep, label2_to_keep]]
           # print('Keeping labelsII:', [label_to_keep, label1_to_keep, label2_to_keep])
        elif label_to_keep != 'I' and label1_to_keep != 'I':
            new_shapes = [shape for shape in data['shapes'] if shape['label'] in [label_to_keep, label1_to_keep]]
            print('Keeping labelsI:', [label_to_keep, label1_to_keep])
        else:
            new_shapes = [shape for shape in data['shapes'] if shape['label'] == label_to_keep]
            print(f'Keeping label: {label_to_keep}, {new_shapes}')
        
        # Debug: Print the number of shapes before and after filtering
        #print(f"Number of shapes before filtering: {len(data['shapes'])}")
        #print(f"Number of shapes after filtering: {len(new_shapes)}")
        
        data['shapes'] = new_shapes
        
        # Write the modified JSON data to the new JSON file
        with open(new_json_path, 'w') as file:
            json.dump(data, file, indent=4)
        #print(f"Filtered JSON written to {new_json_path}")


# Example usage:
input_dir = 'D:\\MuStr811Total171img\\duplicate'
output_dir = 'D:\\MuStr811Total171img\\duplicate'
label_to_keep = 'Crack'
label1_to_keep = 'Wrinkle'
label2_to_keep = 'I'
filter_labels_in_files(input_dir, output_dir, label_to_keep,label1_to_keep,label2_to_keep)
