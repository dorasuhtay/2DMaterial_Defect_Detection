# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 14:44:25 2024

@author: susu
"""
import numpy as np
from skimage import io, color, measure, filters,morphology,exposure
import json
import os
import matplotlib.pyplot as plt
from skimage.color import rgb2hsv
#from skimage import morphology

def segment_image_1(image_path):
    # Read the image
    image = io.imread(image_path)
    plt.imshow(image)
    plt.title('Uploaded Image')
    plt.axis('off')
    plt.show()
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    # Convert the image to grayscale
    gray_image = color.rgb2gray(image)
    plt.figure()
    plt.title("Grayscale Image")
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.show()

    # Apply Otsu's threshold
    thresh = filters.threshold_otsu(gray_image)
    #binary_mask = gray_image > thresh
    binary_mask = gray_image > thresh


    # Display binary mask for debugging
    plt.figure()
    plt.title("Binary Mask")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    plt.show()

    # Label the regions
    labeled_mask, num_labels = measure.label(binary_mask, return_num=True, connectivity=2)

    return labeled_mask, num_labels

def segment_image(image_path):
    # Read the image
    image = io.imread(image_path)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    
   # Convert the image to HSV
    hsv_image = rgb2hsv(image)
    h_channel, s_channel, v_channel = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
    
    # Define thresholds for yellow, blue, and purple
    yellow_mask = (h_channel > 0.1) & (h_channel < 0.2) & (s_channel > 0.5) & (v_channel > 0.5)
    blue_mask = (h_channel > 0.55) & (h_channel < 0.7) & (s_channel > 0.5) & (v_channel > 0.5)
    purple_mask = (h_channel > 0.7) & (h_channel < 0.8) & (s_channel > 0.3) & (v_channel > 0.4)
    
    # Combine masks (if needed)
    combined_mask = yellow_mask | blue_mask | purple_mask
    
    # Debugging plots
    plt.figure()
    plt.title("Yellow Mask")
    plt.imshow(yellow_mask, cmap='gray')
    plt.axis('off')
    plt.show()
    
    plt.figure()
    plt.title("Blue Mask")
    plt.imshow(blue_mask, cmap='gray')
    plt.axis('off')
    plt.show()
    
    plt.figure()
    plt.title("Purple Mask")
    plt.imshow(purple_mask , cmap='gray')
    plt.axis('off')
    plt.show()
   
       
    # Label the regions
    labeled_mask, num_labels = measure.label(combined_mask, return_num=True, connectivity=2)
    
    return labeled_mask, num_labels


def save_to_json(labeled_mask, image_path, output_path):
    # Convert labels to a format suitable for Labelme
    labelme_format = {
        "version": "4.5.6",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,
        "imageHeight": labeled_mask.shape[0],
        "imageWidth": labeled_mask.shape[1]
    }

    unique_labels = np.unique(labeled_mask)
   # print(f"unique_labels:{unique_labels}")
    for label in unique_labels:
       # print(f"label:{label}")
        if label == 0 :
            continue  # Skip background
        
        mask = labeled_mask == label
        contours = measure.find_contours(mask, 0.5)
        for contour in contours:
            if len(contour) > 0 and len(contour) < 1500:  # Check if the contour has more than 10 points
                shape = {
                    "label": "Nucleation", #"Crack", #"Nucleation"
                    "points": contour[:, ::-1].tolist(),
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {}
                }
                labelme_format["shapes"].append(shape)

    # Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(labelme_format, f, indent=4)

def process_image(image_path, output_json_folder):
    if not os.path.exists(output_json_folder):
        os.makedirs(output_json_folder)
    
    output_path = os.path.join(output_json_folder, os.path.splitext(os.path.basename(image_path))[0] + '.json')
    labeled_mask, num_labels = segment_image(image_path)
    save_to_json(labeled_mask, image_path, output_path)
    print(f"Processed {image_path}")
    # Plot the original and labeled images
    plot_images(image_path, labeled_mask)
    
def plot_images(image_path, labeled_mask):
    image = io.imread(image_path)
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    plt.figure(figsize=(12, 6))
    
    # Plot original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title('Original Image')
    plt.axis('off')
    
    # Plot labeled image
    plt.subplot(1, 2, 2)
    plt.imshow(labeled_mask, cmap='nipy_spectral')
    plt.title('Labeled Image')
    plt.axis('off')
    
    plt.show()

def main(input_image_path, output_json_folder):
    directory, directory_name = os.path.split(input_image_path)
    cnt_file,file_name =count_files(input_image_path)
    if directory_name == "" :
        for dirpath, _, filenames in os.walk(input_image_path): 
            for filename in filenames:
                if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp"):
                    image_path = os.path.join(dirpath, filename)
                    process_image(image_path, output_json_folder)
                    
    else:
        process_image(input_image_path, output_json_folder)
    
    
def count_files(root_dir):
    file_count = 0
    file_names = ""
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(".png") or filename.endswith(".jpg") or filename.endswith(".bmp"):
                file_count += 1
                file_names = filename
                #print(f"file_count{file_count}, {filenames}")
    return file_count, file_names

if __name__ == "__main__":
    input_image_path  = input("Please type file location and name : ")# Update this path to your image file path
    output_json_folder = os.path.dirname(input_image_path)
    main(input_image_path, output_json_folder)
