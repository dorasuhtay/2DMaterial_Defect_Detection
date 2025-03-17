# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 11:21:04 2024

@author: susu
"""

import numpy as np
from skimage import io, color, measure, filters,morphology,exposure
import json
import os
import matplotlib.pyplot as plt
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
    
    # Convert to grayscale
    gray_image = color.rgb2gray(image)
    
    # Preprocess image: denoise and enhance contrast
    blurred_image = filters.gaussian(gray_image, sigma=1)
    #p2, p98 = np.percentile(blurred_image, (2, 98))
    p2, p98 = np.percentile(blurred_image, (2, 98))
    contrast_stretched = exposure.rescale_intensity(blurred_image, in_range=(p2, p98))
    
    # Apply Otsu's threshold
    thresh = filters.threshold_otsu(contrast_stretched)
    #thresh = filters.threshold_local(contrast_stretched, block_size=35, offset=10)
    print(f"thresh:{thresh}")
    binary_mask = contrast_stretched > thresh
    
    # Remove small objects
    #min_size = 500  # X20
    min_size =500 # Adjust based on your needs
    cleaned_binary_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)
    
    # Label the regions
    labeled_mask, num_labels = measure.label(cleaned_binary_mask, return_num=True, connectivity=2)
    
    # Debugging plots
    plt.figure()
    plt.title("Original Image")
    plt.imshow(image)
    plt.axis('off')
    plt.show()
    
    plt.figure()
    plt.title("Grayscale Image")
    plt.imshow(gray_image, cmap='gray')
    plt.axis('off')
    plt.show()
    
    plt.figure()
    plt.title("Binary Mask")
    plt.imshow(binary_mask, cmap='gray')
    plt.axis('off')
    plt.show()
   
    plt.figure()
    plt.title("Cleaned Binary Mask")
    plt.imshow(cleaned_binary_mask, cmap='gray')
    plt.axis('off')
    plt.show()
    
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
            if len(contour) >50 and len(contour) < 3000:  # Check if the contour has more than 10 points
                shape = {
                    "label": "Nucleation",#"Seed", #"Crack", #"Nucleation"
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
