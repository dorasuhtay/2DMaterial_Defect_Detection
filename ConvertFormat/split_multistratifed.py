# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 09:39:11 2024

@author: susu
"""

import json
import os
import shutil
import argparse
import sys
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import numpy as np

class_list = ['Crack', 'Wrinkle', 'Residue', 'Lacey Graphene', 'Nucleation']

def extract_labels(json_files, labelme_dataset_dir):
    labels = []
    i=0
    for json_file in json_files:
        i+=1
        json_path = os.path.join(labelme_dataset_dir, json_file)
        with open(json_path, 'r') as f:
            data = json.load(f)
            shapes = data.get('shapes', [])
            label_vector = [0] * len(class_list)
            for shape in shapes:
                label = shape.get('label')
                if label in class_list:
                    label_vector[class_list.index(label)] = 1
            labels.append(label_vector)
        #print(f"labels:{i}")
    return np.array(labels)

def labelme_json_to_yolov7_seg(labelme_dataset_dir):
    json_files = [pos_json for pos_json in os.listdir(labelme_dataset_dir) if pos_json.endswith('.json')]
    for i in range(len(json_files)):
        with open(labelme_dataset_dir + "/" + json_files[i]) as f:
            data = json.load(f)
        width = data["imageWidth"]
        height = data["imageHeight"]
        shapes = data["shapes"]
        text_file_name = labelme_dataset_dir + "/" + json_files[i]
        text_file_name = text_file_name.replace("json", "txt")
        text_file = open(text_file_name, 'w')
        for shape in shapes:
            class_name = shape["label"]
            class_id = class_list.index(str(class_name))
            points = shape["points"]
            normalize_point_list = [class_id]
            for point in points:
                normalize_x = point[0] / width
                normalize_y = point[1] / height
                normalize_point_list.append(normalize_x)
                normalize_point_list.append(normalize_y)
            text_file.write(" ".join(map(str, normalize_point_list)) + "\n")
            
def stratified_train_val_test_split(labelme_dataset_dir, output_dataset_dir, image_name, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
#def stratified_train_val_test_split(labelme_dataset_dir, output_dataset_dir, image_name, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    print(f'labelme_dataset_dir:{labelme_dataset_dir}')
    json_files = [pos_json for pos_json in os.listdir(labelme_dataset_dir) if pos_json.endswith('.json')]
    
    # Extract labels for stratified splitting
    labels = extract_labels(json_files, labelme_dataset_dir)
    # Debug prints to check the ratio values
    #print(f"Train Ratio: {train_ratio}, Validation Ratio: {val_ratio}, Test Ratio: {test_ratio}")

    # Ensure that train_ratio + val_ratio + test_ratio == 1.0
    assert train_ratio + val_ratio + test_ratio == 1.0, "The sum of the ratios must be 1.0"

    # Initialize stratified splitters
    stratified_split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=(1 - train_ratio))
    stratified_val_test_split = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_ratio / (test_ratio + val_ratio))

    # Split the data
    for train_index, val_test_index in stratified_split.split(json_files, labels):
        for val_index, test_index in stratified_val_test_split.split([json_files[i] for i in val_test_index], 
                                                                     [labels[i] for i in val_test_index]):
            train_idxs, val_idxs, test_idxs = train_index, val_test_index[val_index], val_test_index[test_index]
            
    print(f"train:{train_idxs} , val:{val_idxs}, test:{val_idxs}")

    # Output dataset dir setting
    train_folder = os.path.join(output_dataset_dir, 'train/')
    val_folder = os.path.join(output_dataset_dir, 'val/')
    test_folder = os.path.join(output_dataset_dir, 'test/')
    
    def create_folders(base_folder):
        image_folder = os.path.join(base_folder, 'images/')
        label_folder = os.path.join(base_folder, 'labels/')
        json_folder = os.path.join(base_folder, 'json/')
        os.makedirs(image_folder, exist_ok=True)
        os.makedirs(label_folder, exist_ok=True)
        os.makedirs(json_folder, exist_ok=True)
        return image_folder, label_folder, json_folder

    train_image_folder, train_label_folder, train_json_folder = create_folders(train_folder)
    val_image_folder, val_label_folder, val_json_folder = create_folders(val_folder)
    test_image_folder, test_label_folder, test_json_folder = create_folders(test_folder)

    def copy_files(idxs, dest_image_folder, dest_label_folder, dest_json_folder):
        for idx in idxs:
            basename = os.path.splitext(json_files[idx])[0]
            jpg_path = os.path.join(labelme_dataset_dir, f"{basename}.jpg")
            png_path = os.path.join(labelme_dataset_dir, f"{basename}.png")
            PNG_path = os.path.join(labelme_dataset_dir, f"{basename}.PNG")
            txt_path = os.path.join(labelme_dataset_dir, f"{basename}.txt")
            json_path = os.path.join(labelme_dataset_dir, f"{basename}.json")
            
            if os.path.exists(jpg_path):
                shutil.copy2(jpg_path, dest_image_folder)
            elif os.path.exists(png_path):
                shutil.copy2(png_path, dest_image_folder)
            elif os.path.exists(PNG_path):
                shutil.copy2(PNG_path, dest_image_folder)
            else:
                print(f"Image file not found for {basename}")
            if os.path.exists(txt_path):
                shutil.copy2(txt_path, dest_label_folder)
            else:
                print(f"Image file not found for {basename}")
            if os.path.exists(json_path):
                shutil.copy2(json_path, dest_json_folder)
            else:
                print(f"Image file not found for {basename}")

    copy_files(train_idxs, train_image_folder, train_label_folder, train_json_folder)
    copy_files(val_idxs, val_image_folder, val_label_folder, val_json_folder)
    copy_files(test_idxs, test_image_folder, test_label_folder, test_json_folder)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--labelme_dataset_dir', type=str, default=None,
                        help='Please input the path of the labelme files (jpg and json)')
    parser.add_argument('--train_ratio', type=float, default=0.7,
                        help='Please input the training dataset size, for example 0.7')
    parser.add_argument('--val_ratio', type=float, default=0.2,
                        help='Please input the validation dataset size, for example 0.2')
    parser.add_argument('--test_ratio', type=float, default=0.1,
                        help='Please input the test dataset size, for example 0.1')
    parser.add_argument('--output_dataset_dir', type=str, default=None,
                        help='Please input desired processed data directory.')
    parser.add_argument('--image_name', type=str, default=None,
                        help='Please input image name without ids.')
    args = parser.parse_args(sys.argv[1:])
    labelme_json_to_yolov7_seg(args.labelme_dataset_dir)
    stratified_train_val_test_split(args.labelme_dataset_dir, args.output_dataset_dir, args.image_name, 
                         train_ratio=args.train_ratio, val_ratio=args.val_ratio, test_ratio=args.test_ratio)


#python D:\NCU\Segmentation\yolov7\split_multistratifed.py --labelme_dataset_dir D:\IndSeparate_5label_dataset --train_ratio 0.7 --val_ratio 0.15 --test_ratio 0.15 --output_dataset_dir D:/MuStr71515Total631img --image_name MuStr71515Total631img
#python split_multistratifed.py --labelme_dataset_dir D:\5label_datasetwithNewImage --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1 --output_dataset_dir D:\Mustr811Total180img --image_name Mustr811Total180img
#pip install scikit-learn
#pip install iterstrat
