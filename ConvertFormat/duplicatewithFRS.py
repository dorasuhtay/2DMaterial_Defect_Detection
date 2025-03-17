# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 12:57:55 2025

@author: susu
"""
import json
import shutil
import os
import cv2
import numpy as np
import base64

def apply_transformations(image, shapes, transformation):
    h, w = image.shape[:2]
    new_shapes = json.loads(json.dumps(shapes))  # Deep copy to avoid modifying original shapes

    if transformation == "flip_horizontal":
        image = cv2.flip(image, 1)  # Flip left-right
        for shape in new_shapes:
            for point in shape['points']:
                point[0] = w - point[0]  # Adjust X-coordinates

    elif transformation == "flip_vertical":
        image = cv2.flip(image, 0)  # Flip up-down
        for shape in new_shapes:
            for point in shape['points']:
                point[1] = h - point[1]  # Adjust Y-coordinates

    elif transformation == "rotate_90":
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        for shape in new_shapes:
            for point in shape['points']:
                x, y = point
                point[0] = h - y  # Swap & adjust X
                point[1] = x  # Swap Y

    elif transformation == "rotate_180":
        image = cv2.rotate(image, cv2.ROTATE_180)
        for shape in new_shapes:
            for point in shape['points']:
                point[0] = w - point[0]  # Adjust X
                point[1] = h - point[1]  # Adjust Y

    elif transformation == "rotate_270":
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
        for shape in new_shapes:
            for point in shape['points']:
                x, y = point
                point[0] = y  # Swap X
                point[1] = w - x  # Swap & adjust Y

    elif transformation == "shear":
        M = np.float32([[1, 0.2, 0], [0.2, 1, 0]])  # Shear transformation matrix
        image = cv2.warpAffine(image, M, (w, h))
        for shape in new_shapes:
            for point in shape['points']:
                x, y = point
                point[0] = x + 0.2 * y  # Apply shear
                point[1] = 0.2 * x + y  # Apply shear

    return image, new_shapes

def augment_and_save(image, json_data, base_name, label_suffix, output_dir):
    transformations = ["flip_horizontal", "flip_vertical", "rotate_90", "rotate_180",  "shear"]
    
    for transform in transformations:
        new_image, new_shapes = apply_transformations(image.copy(), json_data['shapes'], transform)
        '''
        # Debugging: Show transformed image
        plt.imshow(f"Transformed - {transform}", new_image)
        plt.waitKey(500)  # Display for 500ms
        plt.destroyAllWindows()
        '''
        # Construct new filenames
        new_json_name = f"{label_suffix}_{base_name}_{transform}.json"
        new_image_name = f"{label_suffix}_{base_name}_{transform}.jpg"

        # Define paths
        new_json_path = os.path.join(output_dir, new_json_name)
        new_image_path = os.path.join(output_dir, new_image_name)

        os.makedirs(output_dir, exist_ok=True)  # Ensure directory exists

        # Save new image
        success = cv2.imwrite(new_image_path, new_image)
        

        # Save updated JSON with new shapes
        json_data_copy = json_data.copy()
        json_data_copy['shapes'] = new_shapes
        json_data_copy['imagePath'] = new_image_name
        
       # Read transformed image as binary and encode to base64
        with open(new_image_path, "rb") as img_file:
            encoded_string = base64.b64encode(img_file.read()).decode('utf-8')
        
        json_data_copy["imageData"] = encoded_string
        
        with open(new_json_path, 'w') as file:
            json.dump(json_data_copy, file, indent=4)

        #print(f"Saved: {new_image_name}, {new_json_name}")

def filter_labels_in_files(input_dir, output_dir, label_to_keep, label1_to_keep, label2_to_keep, label3_to_keep):
    os.makedirs(output_dir, exist_ok=True)  # Ensure output directory exists
    json_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]

    for json_file in json_files:
        base_name = os.path.splitext(json_file)[0]

        # Find corresponding image file
        image_file = None
        for ext in [".jpg", ".png", ".bmp"]:
            temp_path = os.path.join(input_dir, f"{base_name}{ext}")
            if os.path.exists(temp_path):
                image_file = temp_path
                break

        if not image_file:
            #print(f"Image file for {json_file} not found, skipping.")
            continue  # Skip if no image is found

        # Read JSON file
        json_path = os.path.join(input_dir, json_file)
        with open(json_path, 'r') as file:
            data = json.load(file)

        labels_in_json = {shape['label'] for shape in data['shapes']}

        # Determine which labels exist in this file
        selected_labels = [label for label in [label_to_keep, label1_to_keep, label2_to_keep, label3_to_keep] if label != 'I' and label in labels_in_json]

        if not selected_labels:
            #print(f"No matching labels found in {json_file}, skipping.")
            continue  # Skip if none of the target labels are found

        # Construct new filename based on found labels
        label_suffix = "_".join(selected_labels)
        new_json_name = f"{label_suffix}_{base_name}.json"
        new_image_name = f"{label_suffix}_{base_name}.jpg"

        # Define new file paths
        new_json_path = os.path.join(output_dir, new_json_name)
        new_image_path = os.path.join(output_dir, new_image_name)

        # Copy image and JSON file to new location
        shutil.copyfile(image_file, new_image_path)
        shutil.copyfile(json_path, new_json_path)

        # Filter JSON to only include selected labels
        data['shapes'] = [shape for shape in data['shapes'] if shape['label'] in selected_labels]

        # Save the modified JSON
        with open(new_json_path, 'w') as file:
            json.dump(data, file, indent=4)

        #print(f"Saved: {new_json_name}, {new_image_name}")

        # Load image for augmentation
        image = cv2.imread(new_image_path)
        augment_and_save(image, data, base_name, label_suffix, output_dir)

# Example usage
input_dir = "D:/NCU/Segmentation/Lacey/171images/Res_lacy"
output_dir = "D:/NCU/Segmentation/Lacey/171images/Res_lacy_dupli"
filter_labels_in_files(input_dir, output_dir, "Residue", "Lacey Graphene", "I", "I")
