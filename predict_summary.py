# -*- coding: utf-8 -*-
"""
Created on Sun Feb 16 13:10:39 2025

@author: susu
"""

import argparse
import os
import platform
import sys
import numpy as np
from pathlib import Path
import tkinter as tk
from tkinter import filedialog

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_coords, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.segment.general import process_mask, scale_masks
from utils.segment.plots import plot_masks
from utils.torch_utils import select_device, smart_inference_mode
#from pycocotools import mask as mask_util
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties

'''
graph_color = {'Nucleation': 'lightgreen',
             #'Multilayer of domain': 'blue',
             'Crack': 'purple',
             'Residue': 'red',
             'Lacey Graphene': 'orange',
             'Wrinkle': 'pink'}
'''
graph_color = {'Nucleation': '#CFD231',
             #'Multilayer of domain': 'blue',
             'Crack': 'red',
             'Residue': '#FF701F',
             'Lacey Graphene': '#FFB21D',
             'Wrinkle': 'pink'}

#set KMP_DUPLICATE_LIB_OK=TRUE
#weights_only=True

def_imgs = 832
db_header = ['No','Filename','Nucleation','Crack','Residue','Lacey Graphene','Wrinkle',
             'Area_Nucleation','Area_Crack','Area_Residue','Area_Lacey Graphene','Area_Wrinkle']
categories = ['Nucleation', 'Crack', 'Residue', 'Lacey Graphene', 'Wrinkle']
'''
db_header = ['No','Filename','Nucleation','Multilayer of domain','Crack','Residue','Lacey Graphene','Wrinkle',
             'Area_Nucleation','Area_Multilayer of domain','Area_Crack','Area_Residue','Area_Lacey Graphene','Area_Wrinkle']
# Define the categories
categories = ['Nucleation', 'Multilayer of domain', 'Crack', 'Residue', 'Lacey Graphene', 'Wrinkle']
'''
savefilename = 'y7seg_MuSt5M701515_X20'

def draw_legend(image, class_colors, font_scale=0.5, font_thickness=1, padding=10):
    """
    Draw a legend on the right side of the image.
    :param image: The image on which to draw the legend.
    :param class_colors: A dictionary with class names as keys and colors as values.
    :param font_scale: The scale of the font.
    :param font_thickness: The thickness of the font.
    :param padding: The padding between the text lines.
    """
    y_offset = padding
    for class_name, color in class_colors.items():
        # Draw the colored rectangle
        cv2.rectangle(image, (image.shape[1] - 100, y_offset), (image.shape[1] - 80, y_offset + 20), color, -1)
        # Draw the class name text
        cv2.putText(image, class_name, (image.shape[1] - 75, y_offset + 15), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, (0, 0, 0), font_thickness, lineType=cv2.LINE_AA)
        y_offset += 30  # Move to the next line

def count_mask_area1(masks,im0):
    total_pixels = 0
    total_masked_pixels = 0
    #Crate a binary image repersenting the union of all masks
    union_mask= np.zeros_like(im0[:,:,0], dtype=np.uint8)
    Total_union_mask=0

    for mask in masks:
        # Resize the mask to match the shape of the original image
        mask_resized = cv2.resize(mask.cpu().numpy(), (im0.shape[1], im0.shape[0]))

        # Perform logical OR with the resized mask
        union_mask = np.logical_or(union_mask, mask_resized)
        total_pixels += np.prod(mask.shape)
        #total_masked_pixels += np.sum(mask)
        mask_cpu = mask.cpu().numpy()

        # Perform NumPy operations on the CPU-resident tensor
        total_masked_pixels += np.sum(np.float64(mask_cpu))
        #total_masked_pixels += np.sum(np.float64(mask))

    #Count non-zero pixels in the union mask
    Total_union_mask = np.sum(union_mask)

    percentage_area = (total_masked_pixels / total_pixels) * 100

    return total_pixels, total_masked_pixels, percentage_area, Total_union_mask


def count_mask_area(masks, im0):
    # Assuming 'masks' is a binary mask (0 and 1 values)
    # Convert the binary mask to grayscale (0 and 255 values)
    mask_gray = (masks * 255).to(torch.uint8)
    total_pixels_image = np.prod(im0.shape[:2])
    union_mask = np.zeros_like(im0[:,:,0], dtype=np.uint8)
    total_masked_pixels = 0
    Total_union_mask = 0
    Total_seg_pixel =0
    for mask in masks:
        # Resize the binary mask to the image size
        mask_resized = cv2.resize(mask.cpu().numpy(), (im0.shape[1], im0.shape[0]))
        # Perform logical OR with the resized mask
        union_mask = np.logical_or(union_mask, mask_resized)
        mask_cpu = mask.cpu().numpy()
        # Perform NumPy operations on the CPU-resident tensor
        total_masked_pixels += np.sum(np.float64(mask_cpu))

    # Get the total number of pixels in the union mask
    Total_union_mask = np.sum(union_mask)
    Total_seg_pixel =Total_union_mask+total_masked_pixels
    # Calculate percentage area
    percentage_area = (Total_seg_pixel / total_pixels_image) * 100
    return total_pixels_image, total_masked_pixels, percentage_area, Total_union_mask

def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
def addtwolabels(x, y, z):
    for i in range(len(x)):
        fontdict={'family': 'Arial', 'weight': 'bold', 'size': 14}
        plt.text(i, y[i], f"{y[i]} ({z[i]:.2f}%)" , ha= 'center' , color="blue", fontdict = fontdict)
        '''
        plt.text(i, y[i]-5, y[i] , ha= 'center' , color="blue", fontsize =14)
        # Convert z[i] to a float for comparison
        z_value = float(z[i])
        color = 'black' if z_value > 0 else 'red'
        plt.text(i, y[i], f"{z[i]}%" , ha= 'center', color=color)
        '''

def fount_class_count(found_classes, im0,img_dir,img_name,per_Total_masks_in_image, area_class):
    #strname = f"{img_name} \n Total area percentage of occupied defect:{per_Total_masks_in_image:.2f}%"
    strname = f" Total area percentage of occupied defect:{per_Total_masks_in_image:.2f}%"
    '''
    aligns = im0.shape
    align_btm = aligns[0]
    align_rgt = (aligns[1]/1.7)

    for i,(k,v) in enumerate(found_classes.items()):
        a=f"{k} = {v}"
        align_btm=align_btm-35
        cv2.putText(im0,str(a),(int(align_rgt),align_btm),cv2.FONT_HERSHEY_PLAIN,1,(45,255,255),1,cv2.LINE_AA)
    '''
    keys = list(found_classes.keys())
    area_values=[]
    area_values_pie=[]
    for key in keys:
        if key == 'Nucleation' or key == 'Multilayer of domain':
            key = 'Area_Nucleation'
        if key == 'Crack':
            key = 'Area_Crack'
        if key == 'Residue':
            key = 'Area_Residue'
        if key == 'Lacey Graphene':
            key = 'Area_Lacey Graphene'
        if key == 'Wrinkle':
            key = 'Area_Wrinkle'
            
        area_value = area_class.get(key)
        #area_values_pie = area_class.get(key)
        if area_value is not None:
            area_values.append(area_value)
            area_values_pie.append(area_value)
    # Calculate the total and normalize
    not_defectarea = 100 - sum(area_values_pie)
    area_values_pie.append(not_defectarea)
    #normalized_values = [value / total_value * 100 for value in area_values]
    #print(f"area_values:{area_values} , total_value:{total_value}, normalized_values:{normalized_values}")
    # Prepare labels for the pie chart
    labels = list(found_classes.keys()) + ['Defect-Free Area']  # Append the label for 'Not Defect Area'

    fontdict={'family': 'Arial', 'weight': 'bold', 'size': 14}
    font = FontProperties(family='Arial', weight='bold', size=14)
    #Plotting the pie chart
    #overlap problem
    
    #original
    plt.figure(figsize=(10,5))
    plt.pie(area_values_pie, labels=labels, autopct='%1.1f%%', startangle=90,
            colors=[graph_color.get(name, 'gray') for name in labels],
            textprops={'fontsize': 12, 'fontweight': 'bold', 'family': 'Arial'}, pctdistance=0.85, labeldistance=1.1)
    plt.title(strname,fontdict = fontdict)
    #addtwolabels(keys, list(found_classes.values()),area_values)
    #plt.legend(loc='lower left', bbox_to_anchor=(1, 0))
    save_pie_path =  str(img_name) + "_pie.jpg"
    plt.savefig(img_dir/save_pie_path)
    '''
    plt.figure(figsize=(12, 8))
    plt.pie(area_values_pie, 
            labels=labels, 
            autopct='%1.1f%%', 
            startangle=90,
            colors=[graph_color.get(name, 'gray') for name in labels],
            textprops={'fontsize': 12, 'fontweight': 'bold', 'family': 'Arial'}, 
            pctdistance=0.8, 
            labeldistance=1.4)  # Adjust distances
    
    plt.title(strname, fontdict=fontdict)
    plt.legend(labels, loc='best', bbox_to_anchor=(1.2, 0.5), prop=font)  # Add a legend
    
    save_pie_path =  str(img_name) + "_pie.jpg"
    plt.savefig(img_dir/save_pie_path,bbox_inches="tight")
    '''
     # creating the bar plot
    plt.figure(figsize = (10, 5))
    plt.bar(found_classes.keys(), found_classes.values(), width = 0.4, color=[graph_color.get(name, 'gray') for name in found_classes.keys()])
    plt.xlim(-0.4, len(found_classes) - 0.4)  # Adjust based on the number of categories
    plt.ylim(0, max(found_classes.values()) * 1.2)  # Add extra space at the top
    #addlabels(list(found_classes.keys()), list(found_classes.values()))
    #print(f"area_values:{area_values}")
    addtwolabels(keys, list(found_classes.values()),area_values)
    #plt.xlabel('Type', fontdict = fontdict)
    plt.ylabel('Defect Count', fontdict = fontdict)
    plt.title(strname, fontdict = fontdict)
    # Set x and y tick labels
    plt.xticks(rotation=45,fontproperties=font)  # Rotate x tick labels for better visibility
    plt.yticks(fontproperties=font)
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    save_pie_path =  str(img_name)+"_graph.jpg"
    plt.savefig(img_dir/save_pie_path)
    

def summary_graph(output_results_path,save_dir):
    #found_classes={}
    get_lastno = 0
    fontdict={'family': 'Arial', 'weight': 'bold', 'size': 14}
    font = FontProperties(family='Arial', weight='bold', size=14)
    count_class = {'Nucleation': 0,  'Crack': 0, 'Residue': 0, 'Lacey Graphene': 0, 'Wrinkle': 0 }
    area_class = {'Nucleation': 0,  'Crack': 0, 'Residue': 0, 'Lacey Graphene': 0, 'Wrinkle': 0, 'Defect-free' :0}
    
    first_line_flag = True
    defect_freearea = 0.0
    with open(output_results_path, "r") as file:
        content = file.read()
        for i, line in enumerate(content.split('\n')):
            parts = line.split(',')
            #if line.strip():
            #print(f"len(content):{len(content)}")
            if i > 0 and line.strip():                    
                if first_line_flag:     
                    count_class['Nucleation'] = parts[2]
                    #count_class['Multilayer of domain'] = parts[3]
                    count_class['Crack'] = parts[3]
                    count_class['Residue'] = parts[4]
                    count_class['Lacey Graphene'] = parts[5]
                    count_class['Wrinkle'] = parts[6]          

                    area_class['Nucleation'] = parts[7]
                     #count_class['Multilayer of domain'] = int(count_class['Multilayer of domain']) + int(parts[3])
                    area_class['Crack'] = parts[8]
                    area_class['Residue'] = parts[9]
                    area_class['Lacey Graphene'] = parts[10]
                    area_class['Wrinkle'] = parts[11]     
                    area_values = 100 - (float(parts[7])+float(parts[8])+float(parts[9])+float(parts[10])+float(parts[11]))                     
                    area_class['Defect-free'] = float(area_values)
                    #defect_freearea = float(area_values)
                    first_line_flag = False
                else:
                    count_class['Nucleation'] = int(count_class['Nucleation']) + int(parts[2])
                    #count_class['Multilayer of domain'] = int(count_class['Multilayer of domain']) + int(parts[3])
                    count_class['Crack'] = int(count_class['Crack']) + int(parts[3])
                    count_class['Residue'] = int(count_class['Residue']) + int(parts[4])
                    count_class['Lacey Graphene'] = int(count_class['Lacey Graphene']) + int(parts[5])
                    count_class['Wrinkle'] = int(count_class['Wrinkle']) + int(parts[6])
                    
                    area_class['Nucleation'] = float(area_class['Nucleation']) + float(parts[7])
                    #count_class['Multilayer of domain'] = int(count_class['Multilayer of domain']) + int(parts[3])
                    area_class['Crack'] = float(area_class['Crack']) + float(parts[8])
                    area_class['Residue'] = float(area_class['Residue']) + float(parts[9])
                    area_class['Lacey Graphene'] = float(area_class['Lacey Graphene']) + float(parts[10])
                    area_class['Wrinkle'] = float(area_class['Wrinkle']) + float(parts[11])
                    area_values = 100 - (float(parts[7])+float(parts[8])+float(parts[9])+float(parts[10])+float(parts[11])) 
                    area_class['Defect-free'] += float(area_values)
                    #defect_freearea += float(area_values)
                    get_lastno = int(parts[0])
                    
    print("Defect Counts:", count_class)
    print("Defect Areas:", area_class)
    # creating the bar plot
    plt.figure(figsize = (10, 5))
    plt.bar(count_class.keys(), count_class.values(), width = 0.4, color=[graph_color.get(name, 'gray') for name in count_class.keys()])
    plt.xlim(-0.4, len(count_class) - 0.4)  # Adjust based on the number of categories
    plt.ylim(0, max(count_class.values()) * 1.2)  # Add extra space at the top
    addtwolabels(count_class.keys(), list(count_class.values()),list(area_class.values()))
    plt.ylabel('Defect Count', fontdict = fontdict)
    plt.title('Summary Graph', fontdict = fontdict)
    # Set x and y tick labels
    plt.xticks(rotation=45,fontproperties=font)  # Rotate x tick labels for better visibility
    plt.yticks(fontproperties=font)
    plt.tight_layout()
    plt.savefig(save_dir/'summary_graph.jpg')
    print(f"get_lastno: {get_lastno}")
    
    def custom_autopct(pct):
        return ('%1.2f%%' % pct) if pct > 2 else ''  # Hide small labels
    def custom_label(label, pct):
        if isinstance(pct, list):
            pct = pct[0]
        return label if pct > 2 else ''
    #creat pie chart
    #labels = list(area_class.keys())  # Append the label for 'Not Defect Area'
    pie_chart = {'Nucleation': [],  'Crack': [], 'Residue': [], 'Lacey Graphene': [], 'Wrinkle': [], 'Defect-free': []}
    plt.figure(figsize=(10,5))
    #explode = [0.1 if v < 50 else 0 for v in area_class.values()]
    print(f"piechart:{area_class.values(), area_class.keys()}")
    for category in pie_chart:
        pie_chartarea = round(float(area_class[category]) / get_lastno, 2)
        pie_chart[category].append(pie_chartarea) 
       # print(f"pie_chartarea:{pie_chartarea}, pie_chartcategory: {pie_chart}")
    '''
    plt.pie(area_class.values(), labels=area_class.keys(), colors=[graph_color.get(name, 'gray') for name in area_class.keys()],
            autopct='%1.1f%%', startangle=90,textprops={'fontsize': 12, 'fontweight': 'bold', 'family': 'Arial'}, pctdistance=0.85, labeldistance=1.1)
   
    plt.pie(area_class.values(), labels=area_class.keys(), autopct='%1.1f%%', startangle=90,
            colors=[graph_color.get(name, 'gray') for name in area_class.keys()],explode = explode,
            textprops={'fontsize': 12, 'fontweight': 'bold', 'family': 'Arial'}, pctdistance=0.85, labeldistance=1.1)
    '''
    wedges, texts, autotexts = plt.pie([v[0] if isinstance(v, list) and v else 0 for v in pie_chart.values()],
                            labels=[custom_label(label, pct) for label, pct in zip(pie_chart.keys(), pie_chart.values())], 
                            autopct=custom_autopct, startangle=90,colors=[graph_color.get(name, 'gray') for name in area_class.keys()], 
                            pctdistance=0.85, labeldistance=1.1, textprops={'fontsize': 12, 'fontweight': 'bold', 'family': 'Arial'})
    # Add legend to the right side
    # Generate legend labels with values
    legend_labels =[f"{key}: {value[0]:.1f}%" if isinstance(value, list) and value else f"{key}: 0.0%" for key, value in pie_chart.items()]
    
    # Create legend with colored patches
    plt.legend(wedges, legend_labels, loc='upper left', bbox_to_anchor=(1, 1), prop=font)
   
    plt.title("Summary Pie-chart",fontdict = fontdict)
    plt.savefig(save_dir/'summary_pie.jpg')
    
    
def summary_graph_org(output_results_path,save_dir):
    found_classes={}
    with open(output_results_path, "r") as file:
        content = file.read()
        for line in content.split('\n'):
            parts = line.split('=')
            if len(parts) == 2:
                name, count = parts
                name = name.strip()  # Remove any leading or trailing whitespaces
                count = int(count.strip())  # Convert count to integer
                found_classes[name] = count
         # creating the bar plot
    fig = plt.figure(figsize = (10, 5))
    plt.bar(found_classes.keys(), found_classes.values(), width = 0.4, color=[graph_color.get(name, 'gray') for name in found_classes.keys()])
    addlabels(list(found_classes.keys()), list(found_classes.values()))
    plt.xlabel('Type')
    plt.ylabel('defect Count')
    plt.title("Summary Graph")
    plt.xticks(rotation=45)  # Rotate x tick labels for better visibility
    plt.yticks()
    plt.tight_layout()
    #save_pie_path =  output_results_path+"/summary_graph.jpg"
    #print(f"save_pie_path:{save_pie_path}")
    plt.savefig(save_dir/'summary_graph.jpg')

def open_new_file(output_results_path,found_classes):
    # Save the output string to a text file
    with open(output_results_path, 'a') as f:
        #print(f'output_results_path:{output_results_path}')
    #with open(f'{save_dir}/output_results.txt', 'a') as f:
        combined_results = {}
        for found_name, found_count in found_classes.items():
            combined_results[found_name] = found_count

        for name, count in combined_results.items():
            f.write(f"{name}={count}\n")

def save_db(no_, filename, db_path, found_classes, area_class):
    str_filestatus = 'a'  # Always open the file in append mode

    with open(db_path, str_filestatus) as f:
        #print(f'db_results_path: {db_path}')

        if os.path.getsize(db_path) == 0:
            #print(f"{db_header}")
            # If the file is empty, write the header
            f.write(f"{db_header}\n")
        '''
        combined_results = {'No': '0', 'Filename': '0', 'Nucleation': '0', 'Multilayer of domain': '0',
                        'Crack': '0', 'Residue': '0', 'Lacey Graphene': '0', 'Wrinkle': '0',
                        'Area_Nucleation': '0', 'Area_Multilayer of domain' : '0', 'Area_Crack' : '0',
                        'Area_Residue': '0', 'Area_Lacey Graphene': '0', 'Area_Wrinkle': '0'}
        '''
        combined_results = {'No': '0', 'Filename': '0', 'Nucleation': '0', 'Crack': '0', 'Residue': '0', 'Lacey Graphene': '0', 'Wrinkle': '0',
                        'Area_Nucleation': '0',  'Area_Crack' : '0', 'Area_Residue': '0', 'Area_Lacey Graphene': '0', 'Area_Wrinkle': '0'}
        #combined_results['No'] = no_
        combined_results['No'] = no_
        combined_results['Filename'] = filename
        for found_name, found_count in found_classes.items():
            if found_name == 'Nucleation' or found_name == 'Multilayer of domain':
                combined_results['Nucleation'] = found_count
            #elif found_name == 'Multilayer of domain':
            #    combined_results['Multilayer of domain'] = found_count
            elif found_name == 'Crack':
                combined_results['Crack'] = found_count
            elif found_name == 'Residue':
                combined_results['Residue'] = found_count
            elif found_name == 'Lacey Graphene':
                combined_results['Lacey Graphene'] = found_count
            elif found_name == 'Wrinkle':
                combined_results['Wrinkle'] = found_count
        #area_class      
        for area_name, area_count in area_class.items():
            if area_name == 'Area_Nucleation' or area_name == 'Area_Multilayer of domain':
                combined_results['Area_Nucleation'] = area_count
            #elif area_name == 'Area_Multilayer of domain':
            #    combined_results['Area_Multilayer of domain'] = area_count
            elif area_name == 'Area_Crack':
                combined_results['Area_Crack'] = area_count
            elif area_name == 'Area_Residue':
                combined_results['Area_Residue'] = area_count
            elif area_name == 'Area_Lacey Graphene':
                combined_results['Area_Lacey Graphene'] = area_count
            elif area_name == 'Area_Wrinkle':
                combined_results['Area_Wrinkle'] = area_count
                

        for i, entry in enumerate(combined_results.values()):
            if i != len(combined_results) - 1:
                f.write(f"{entry},")
            else:
                f.write(f"{entry}\n")

def sum_category_values(input_file_path):
    # Initialize a dictionary to store the summed values for each category
    summed_values = {}

    # Open the input file for reading
    with open(input_file_path, 'r') as file:
        # Skip the header line
        next(file)

        # Process each line in the file
        for line in file:
            # Split the line into its components
            parts = line.strip().split(',')
            filename = parts[1]
            # Iterate over the categories and update the summed values
            for category, value in zip(categories, parts[2:]):
                summed_values[category] = summed_values.get(category, 0) + int(value)

    return summed_values

def save_summed_values(summed_values, output_file_path):
    # Open the output file for writing
    with open(output_file_path, 'w') as file:
        # Write the summed values to the file
        for category, value in summed_values.items():
            if value != 0:
                file.write(f"{category}={value}\n")

def open_existing_file(output_results_path,found_classes):
    with open(output_results_path, 'r+') as f:
    #with open(f'{save_dir}/output_results.txt', 'r+') as f:
        # Read the existing content of the file
        existing_content = f.read()
        combine_flag = False
        # Create a dictionary to store occurrences of each string
        combined_results = {}

        for found_name, found_count in found_classes.items():
            # Iterate over each line in the existing content
            #print(f'combine_result start:{combined_results}')
            for index, line in enumerate(existing_content.split('\n')):
                # Split the line by '=' to separate the name and count
                parts = line.split('=')
                if len(parts) == 2:
                    name, count = parts
                    name = name.strip()  # Remove any leading or trailing whitespaces
                    count = int(count.strip())  # Convert count to integer
                    if name == found_name:
                        combined_results[name] = found_count+count
                        print(f'Same, name:{name}{count}, found_name:{found_name}{found_count}, combine_count:{found_count+count}')
                    else:
                        if not combine_flag:
                            combined_results[name] = count
                            print(f'not Same, combined_results:{name}, found_name:{found_name}, count:{count}, {found_count}')
                        else:
                            #print(f'else')
                            for combine_name, combine_count  in combined_results.items():
                                if combine_name == name and count < combine_count:
                                    combined_results[name] = combine_count
                                    print(f'not Same, name:{name}, combine_name:{combine_name}, count:{count}, {combine_count}')
                                elif combine_name == name and count < combine_count:
                                    combined_results[name] = combine_count
                                    print(f'not Same, name:{name}, found_name:{found_name}, count:{count}, {found_count}')

            if index == len(existing_content.split('\n')) - 1:
                combine_flag = True
        # Move the file pointer to the beginning of the file to overwrite its content
        f.seek(0)

        # Write the updated content back to the file
        for name, count in combined_results.items():
            f.write(f"{name}={count}\n")

def calculate_mask_area(mask):
    """
    Calculate the area of a binary mask using OpenCV.
    """
    mask_np = mask.cpu().numpy().astype(np.uint8)  # Convert mask to numpy array and uint8 type
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = sum(cv2.contourArea(contour) for contour in contours)
    
    return area

@smart_inference_mode()
def run(
        weights=ROOT / 'yolov5s-seg.pt',  # model.pt path(s) #runs/train-seg/yolov7-seg4_832/weights/best.pt
        source=ROOT / 'data/images',  # file/dir/URL/glob, 0 for webcam
        data=ROOT / 'data/coco128.yaml',  # dataset.yaml path
        imgsz=(def_imgs, def_imgs),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        #iou_thres=0.5,  # NMS IOU threshold
        max_det=1500,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/predict-seg',  # save results to project/name
        name=savefilename,#'exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=2,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
):
    source = str(source)
    save_img = not nosave and not source.endswith('.txt')  # save inference images
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    #print(f'source:{source}')
    chk_ind_image = False
    if source.lower().endswith('.jpg') or source.lower().endswith('.jpeg') or source.lower().endswith('.png') or source.lower().endswith('.bmp'):
        chk_ind_image = True
    directory, directory_name = os.path.split(source)
    output_results_path_exp = f'{save_dir}/output_results.txt'
    path_result = directory_name+'_result.txt'
    output_results_path = directory+'/'+path_result
    #Create database structure
    dbpath_result = directory_name+'_dbresult.txt'
    dboutput_results_path = directory+'/'+dbpath_result

    if os.path.exists(output_results_path):
        os.remove(output_results_path)

    if os.path.exists(output_results_path_exp):
        os.remove(output_results_path_exp)

    if os.path.exists(dboutput_results_path):
        os.remove(dboutput_results_path)
    dbrecod_NOfilename =0
    # Load model
    device = select_device(device)
   # print(f'starting')
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size
   # print(f'START')
    # Dataloader
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt)
        bs = len(dataset)  # batch_size
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
        bs = 1  # batch_size
    vid_path, vid_writer = [None] * bs, [None] * bs
    
    # Run inference
    model.warmup(imgsz=(1 if pt else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
   

    for path, im, im0s, vid_cap, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred, out = model(im, augment=augment, visualize=visualize)
            proto = out[1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det, nm=32)
            #print(f'non_max_suppression')
        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)
        dbrecod_NOfilename +=1
        # Process predictions
        for i, det in enumerate(pred):  # per image

            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
            sp=""
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # im.jpg
            #save_path = str(save_dir / p.name)
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            found_classes={}
            area_class = {}
            if len(det):
                
                masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                #print(f"processmasks:{masks}")

                #calculate the total number of pixels in the original image
                Total_pixels_in_image = np.prod(im0.shape[:2])
                custom_name = "Nucleation"
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], im0.shape).round()
                combined_count=0
                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    class_name = names[int(c)]
                    if(class_name == "Multilayer of domain"):
                        class_name = "Nucleation"
                        value_nucleation = found_classes.get(class_name,0)
                        found_classes[class_name] = value_nucleation + int(n)
                        #print(f"Mulit: {class_name}: {found_classes[class_name]}:{int(n)}")
                    else:
                        found_classes[class_name] = int(n)
                        #print(f"a: {class_name}: {found_classes[class_name]}:{int(n)}")
                    s += f" {class_name}{'s' * (n > 1)} "  # add to string
                    sp += f"{class_name}={n}\n"
                '''
                # Mask plotting ----------------------------------------------------------------------------------------
                mcolors = [colors(int(cls), True) for cls in det[:, 5]]
                im_masks = plot_masks(im[i], masks, mcolors)  # image with masks shape(imh,imw,3)
                annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                # Mask plotting ----------------------------------------------------------------------------------------
                ''' 
               #class_name = 'Residue'
                          
                # Filter the detections to only customize
                cust_indices = [i for i, cls in enumerate(det[:, 5])  
                              if names[int(cls)] == custom_name and masks[i].sum() > 0]
                
                if cust_indices:
                    cust_masks = masks[cust_indices]  # Select masks for "Residue"
                    #print(f"index30:{cust_indices}")
                    cust_boxes = det[cust_indices, :4]  # Select corresponding bounding boxes
                    cust_classes = det[cust_indices, 5]  # Select corresponding classes
                    mcolors = [colors(int(cls), True) for cls in cust_classes]
                    im_masks = plot_masks(im[i], cust_masks, mcolors)  # image with masks shape(imh, imw, 3)
                    annotator.im = scale_masks(im.shape[2:], im_masks, im0.shape)  # scale to original h, w
                else:
                    print("No valid mask at index 10 or class does not match.")
                 
                total_percentage_area=0
                #kk=0
                
                for j, (*xyxy, conf, cls) in enumerate(reversed(det[:, :6]), start=0):
                    x1,y1,x2,y2 = xyxy
                    b_width = x2-x1
                    b_height = y2-y1
                    bbox_size = b_width * b_height
                    #print(f"b_width:{b_width}, b_height:{b_height}, bbox_size:{bbox_size} , Total_pixels_in_image:{Total_pixels_in_image}")
                   
                    c = int(cls)
                     # Assuming masks is a list of binary masks corresponding to each detected object
                    mask = masks[j] #Assuming class indices start from 1
                    #mask_area = calculate_mask_area(mask)
                    
                    
                    #print(f"mask.shape:{mask.shape}")
                    #total_pixels = mask.numel()  # Total number of pixels in the mask (assuming 2D)
                    
                    total_pixels = mask.numel()  # Total number of pixels in the mask (assuming 2D)
                    #print(f"{total_pixels}")
                    # Calculate the area of the mask
                    # Calculate the area of the mask using morphological dilation
                    mask_dilated = F.pad(mask, (1, 1, 1, 1), value=0)  # Pad mask to avoid border artifacts
                    mask_dilated = F.max_pool2d(mask_dilated.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze(0).squeeze(0)
                    #mask_area = np.sum(mask)
                    mask_area = torch.sum(mask_dilated).item()
                    
                    # Calculate the percentage area occupied by the mask
                    percentage_area = round((mask_area / total_pixels ) * 100, 2)
                    total_percentage_area+=round(percentage_area,2)
                    '''
                    mask = (mask > 0).float()
                    mask_area = torch.sum(mask>0).item()
                    #extra_mask_area = Total_pixels_in_image/640
                    #print(f'extra_mask_area:{extra_mask_area}, mask_area:{mask_area}')
                    percentage_area = round((mask_area / Total_pixels_in_image ) * 100, 2)
                    total_percentage_area+=round(percentage_area,2)
                    each_item = (bbox_size/Total_pixels_in_image)*100
                    
                    # Print tensor values (first converting to numpy array if needed)
                   # mask_values = mask.cpu().numpy() if mask.is_cuda else mask.numpy()
                    #print(f"{j}mask values:\n{mask_values}")
                    #print(f"total_pixels:{Total_pixels_in_image}, mask_area:{mask_area}, Percentage Area: {percentage_area}%, bbox_size:{bbox_size}, each_item: {each_item}")
                    '''
                    #print(f"{j+1}.mask_area:{mask_area},bb:{bbox_size:.2f}, total_pixels:{total_pixels}, percentage_area:{percentage_area}")
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if save_conf else (cls, *xywh)  # label format
                        with open(f'{txt_path}.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')
                    if save_img or save_crop or view_img:  # Add bbox to image
                        
                        #area_class[names[c]] = percentage_area
                        # Add the percentage area to the corresponding class in the dictionary
                        class_name = names[c]
                        #if(names[c] == "Multilayer of domain"):
                         #   names[c] = "Nucleation"
                        if(names[c] == "Lacey Graphene"):
                            class_name = "Area_Lacey Graphene"
                        #if(names[c] == "Multilayer of domain"):
                        #    class_name = "Area_Multilayer of domain"
                        if(names[c] == "Nucleation") :
                            class_name = "Area_Nucleation"
                        if(names[c] == "Crack"):
                            class_name = "Area_Crack"
                        if(names[c] == "Residue"):
                            class_name = "Area_Residue"
                        if(names[c] == "Wrinkle"):
                            class_name = "Area_Wrinkle"
                        if class_name=="Multilayer of domain":
                            class_name = "Area_Nucleation"
                            
                        if class_name in area_class:
                            area_class[class_name] =round(area_class[class_name] + percentage_area,2)
                        else:
                            area_class[class_name] = round(percentage_area,2)
                        
                        #if names[c] == custom_name:
                            #kk+=1
                        #label = None if hide_labels else (names[c] if hide_conf else f'{j+1}.{names[c]} {conf:.2f}, area:{percentage_area:.2f}%')
                       # Ensure the same color for "Area_Nucleation" and "Multilayer of domain"
                        #if names[c] == "Multilayer of domain":
                         #   color = colors(names.index("Nucleation"), True)  # Use the color for "Nucleation"
                       # else:
                        color = colors(c, True)
                        # Determine the label to display
                        if hide_labels:
                            label = None
                        elif hide_conf:
                            label = class_name
                        else:
                            if(names[c] == "Multilayer of domain"):
                                names[c] = "Nucleation"
                                
                            if(custom_name == names[c]):
                            #label = f'{j+1}.{names[c]} {conf:.2f}, area:{percentage_area:.2f}%'
                           # label = f'{j+1}.{names[c]}'
                                label = f'{names[c]}'
                            #if(names[c] == "Nucleation" and j==30):
                                annotator.box_label(xyxy, label, color=color)
                    if save_crop:
                        save_one_box(xyxy, imc, file=save_dir / 'crops' / names[c] / f'{p.stem}.jpg', BGR=True)
                #fount_class_count(found_classes=found_classes,im0=im0, img_dir=save_dir, img_name=p.stem,per_Total_masks_in_image=total_percentage_area, area_class=area_class)
                fount_class_count(found_classes=found_classes,im0=im0, img_dir=save_dir, img_name=p.stem,
                                  per_Total_masks_in_image=total_percentage_area, area_class=area_class)
            print(f'{sp} \n Total % of seg/mask area: {total_percentage_area:.2f}, {area_class}')
            
            if chk_ind_image == False:
                save_db(str(dbrecod_NOfilename), str(p.name), dboutput_results_path, found_classes, area_class)
                save_db(str(dbrecod_NOfilename), str(p.name), output_results_path_exp, found_classes, area_class)
            # Stream results
            im0 = annotator.result()
            if view_img:
                if platform.system() == 'Linux' and p not in windows:
                    windows.append(p)
                    cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
               
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    #plt.savefig(save_path)
                else:  # 'video' or 'stream'
                    if vid_path[i] != save_path:  # new video
                        vid_path[i] = save_path
                        if isinstance(vid_writer[i], cv2.VideoWriter):
                            vid_writer[i].release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer[i].write(im0)

                # Print time (inference-only)

        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    #print(f'END')
    if chk_ind_image == False:
        # Sum the category values
        summed_values = sum_category_values(dboutput_results_path)
        # Save the summed values to a new file
        #save_summed_values(summed_values, output_results_path)
        #save_summed_values(summed_values, output_results_path_exp) #need to change summary
        summary_graph(output_results_path_exp,save_dir) #need to change summary
    # Print results
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)
    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
    if update:
        strip_optimizer(weights[0])  # update model (to fix SourceChangeWarning)

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'yolov5s-seg.pt', help='model path(s)')
    parser.add_argument('--source', type=str, default=ROOT / 'data/images', help='file/dir/URL/glob, 0 for webcam')
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='(optional) dataset.yaml path')
    parser.add_argument('--imgsz', '--img', '--img-size', nargs='+', type=int, default=[def_imgs], help='inference size h,w')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='NMS IoU threshold')
    #parser.add_argument('--iou-thres', type=float, default=0.5, help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1500, help='maximum detections per image')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='show results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--save-crop', action='store_true', help='save cropped prediction boxes')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--visualize', action='store_true', help='visualize features')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default=ROOT / 'runs/predict-seg', help='save results to project/name')
    #parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--name', default= savefilename, help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=2, type=int, help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', help='hide confidences')
    parser.add_argument('--half', action='store_true', help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', help='use OpenCV DNN for ONNX inference')
    opt = parser.parse_args()
    opt.imgsz *= 2 if len(opt.imgsz) == 1 else 1  # expand
    print_args(vars(opt))
    return opt


def main(opt):
    check_requirements(exclude=('tensorboard', 'thop'))
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)
