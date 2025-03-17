# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 10:30:35 2024

@author: user
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from matplotlib.font_manager import FontProperties
#import screeninfo


root_dir = input("Please type file location and name : ")
#root_dir = "D:\\NCU\\Segmentation\\YOLOv7\\detection\\test"  # Replace "example.txt" with the path to your file
file_data = []  # List to store file name and content tuples
error_data = []
#file_names =[]
#contents = []

# record start time (in datetime format)
start = datetime.now()
try:
    # Print the contents of the root directory
    #print("Contents of root directory:")
    #print(os.listdir(root_dir))

    # Traverse subdirectories using os.walk()
    subdirs = [dirpath for dirpath, _, _ in os.walk(root_dir)]
    #print("Subdirectories:", subdirs)
    existing_file=[]
    # Execute commands for each subdirectory
    for subdir in subdirs:
        if subdir != root_dir:
            #command = f"python segment/predict_pie.py --weights runs/train-seg/TrainingModel/Hypermeter/DataAug5M_Model5NEW_/weights/best.pt --device cpu --source {subdir}"
            command = f"python segment/predict_summary.py --weights runs/train-seg/TrainingModel/MuStr811Total171img/weights/best.pt --device cpu --source {subdir} "
            #command = f"python segment/predict_summary.py --weights runs/train-seg/TrainingModel/y7seg_MuSt5M701515/weights/best.pt --device cpu --source {subdir} "
         #   os.chdir(subdir)  # Change to the directory
            os.system(command)
        else:
            for per_root_dir in os.listdir(root_dir):
                file_path=subdir+'/'+per_root_dir+'_result.txt'
                file_path_graph=subdir+'/summarygraph.jpg'
               # print(f'else:{file_path}')
                if os.path.exists(file_path):
                    existing_file.append(file_path)
                    #print(f'existing_file:{existing_file}')
                    # Delete the file
                    os.remove(file_path)

                if os.path.exists(file_path_graph):
                   # existing_file.append(file_path)
                    #print(f'existing_file:{existing_file}')
                    # Delete the file
                    os.remove(file_path_graph)

except FileNotFoundError:
    print(f"Directory '{root_dir}' not found.")
except Exception as e:
    print(f"An error occurred: {e}")


try:
   # print("Contents of root directory:")
    #print(os.listdir(root_dir))

    # Traverse subdirectories using os.walk()
    for dirpath, _, filenames in os.walk(root_dir):
        # Filter filenames to include only those ending with "_result.txt"
        #result_files = [os.path.join(dirpath, filename) for filename in filenames if filename.endswith("_result.txt")]
        result_dbfiles = [os.path.join(dirpath, filename) for filename in filenames if filename.endswith("_dbresult.txt")]
        
       # print(f"goerror_data:{result_dbfiles}")
        for result_dbfile in result_dbfiles:
            #print(f"error_data1:{error_data}")
            # Read the contents of the result file
            with open(result_dbfile, "r") as file:
                content = file.read()
               # Get the filename without the directory path
                filename = os.path.basename(result_dbfile)
                parts = filename.split('_dbr')
                name, endname = parts
                # Store the file name and content in a tuple
                error_data.append((name, content))
                #print(f"error_data:{error_data}")
        break;
except FileNotFoundError:
    print(f"File  not found.")
except Exception as e:
    print(f"An error occurred: {e}")

# Initialize lists to store data for each category
categories = {'Nucleation': [], 'Crack': [], 'Residue': [], 'Lacey Graphene': [], 'Wrinkle': []}
area_categories = {'Nucleation': [], 'Crack': [], 'Residue': [], 'Lacey Graphene': [], 'Wrinkle': []}
avg_categories = {'Nucleation': [],  'Crack': [], 'Residue': [], 'Lacey Graphene': [], 'Wrinkle': []}
error_categories = {'Nucleation': [], 'Crack': [], 'Residue': [], 'Lacey Graphene': [], 'Wrinkle': []}
errorarea_categories = {'Nucleation': [], 'Crack': [], 'Residue': [], 'Lacey Graphene': [], 'Wrinkle': []}

#error_categories = {'Nucleation': [], 'Multilayer of domain': [], 'Crack': [], 'Residue': [], 'Lacey Graphene': [], 'Wrinkle': []}
for filename, content in error_data:
    #area_category_counts = {category: 0 for category in Area_categories}
    min_store_value = {category: 0 for category in error_categories} #stor min value
    max_store_value = {category: 0 for category in error_categories} #store max value
    minarea_store_value = {category: 0 for category in errorarea_categories} #stor min value
    maxarea_store_value = {category: 0 for category in errorarea_categories} #store max value
    first_line_flag = True
    get_lastno=0
    count_class = {'Nucleation': 0,  'Crack': 0, 'Residue': 0, 'Lacey Graphene': 0, 'Wrinkle': 0 }
    area_class = {'Nucleation': 0,  'Crack': 0, 'Residue': 0, 'Lacey Graphene': 0, 'Wrinkle': 0 }
    for i, line in enumerate(content.split('\n')):
        parts = line.split(',')
        #if line.strip():
        #print(f"len(content):{len(content)}")
        if i > 0 and line.strip():
            #print(f"parts:{parts}")
                
            if first_line_flag:             
                #*******min value*****
                min_store_value['Nucleation'] = int(parts[2])
               # min_store_value['Multilayer of domain'] = int(parts[3])
                min_store_value['Crack'] = int(parts[3])
                min_store_value['Residue'] = int(parts[4])
                min_store_value['Lacey Graphene'] = int(parts[5])
                min_store_value['Wrinkle'] = int(parts[6])
                #*******max value*****
                max_store_value['Nucleation'] = int(parts[2])
                #max_store_value['Multilayer of domain'] = int(parts[3])
                max_store_value['Crack'] = int(parts[3])
                max_store_value['Residue'] = int(parts[4])
                max_store_value['Lacey Graphene'] = int(parts[5])
                max_store_value['Wrinkle'] = int(parts[6])
                
                #*******min value*****
                minarea_store_value['Nucleation'] = float(parts[7])
               # minarea_store_value['Multilayer of domain'] = float(parts[9])
                minarea_store_value['Crack'] = float(parts[8])
                minarea_store_value['Residue'] = float(parts[9])
                minarea_store_value['Lacey Graphene'] = float(parts[10])
                minarea_store_value['Wrinkle'] = float(parts[11])
                #*******max value*****
                maxarea_store_value['Nucleation'] = float(parts[7])
                #maxarea_store_value['Multilayer of domain'] = float(parts[9])
                maxarea_store_value['Crack'] = float(parts[8])
                maxarea_store_value['Residue'] = float(parts[9])
                maxarea_store_value['Lacey Graphene'] = float(parts[10])
                maxarea_store_value['Wrinkle'] = float(parts[11])   

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
                
                first_line_flag = False
            else:
                # Update min and max values
                min_store_value['Nucleation'] = min(min_store_value['Nucleation'], int(parts[2]))
                max_store_value['Nucleation'] = max(max_store_value['Nucleation'], int(parts[2]))

               # min_store_value['Multilayer of domain'] = min(min_store_value['Multilayer of domain'], int(parts[3]))
                #max_store_value['Multilayer of domain'] = max(max_store_value['Multilayer of domain'], int(parts[3]))

                min_store_value['Crack'] = min(min_store_value['Crack'], int(parts[3]))
                max_store_value['Crack'] = max(max_store_value['Crack'], int(parts[3]))

                min_store_value['Residue'] = min(min_store_value['Residue'], int(parts[4]))
                max_store_value['Residue'] = max(max_store_value['Residue'], int(parts[4]))

                min_store_value['Lacey Graphene'] = min(min_store_value['Lacey Graphene'], int(parts[5]))
                max_store_value['Lacey Graphene'] = max(max_store_value['Lacey Graphene'], int(parts[5]))

                min_store_value['Wrinkle'] = min(min_store_value['Wrinkle'], int(parts[6]))
                max_store_value['Wrinkle'] = max(max_store_value['Wrinkle'], int(parts[6]))
                
                minarea_store_value['Nucleation'] = min(minarea_store_value['Nucleation'], float(parts[7]))
                maxarea_store_value['Nucleation'] = max(maxarea_store_value['Nucleation'], float(parts[7]))

                #minarea_store_value['Multilayer of domain'] = min(minarea_store_value['Multilayer of domain'], float(parts[9]))
                #maxarea_store_value['Multilayer of domain'] = max(maxarea_store_value['Multilayer of domain'], float(parts[9]))

                minarea_store_value['Crack'] = min(minarea_store_value['Crack'], float(parts[8]))
                maxarea_store_value['Crack'] = max(maxarea_store_value['Crack'], float(parts[8]))

                minarea_store_value['Residue'] = min(minarea_store_value['Residue'], float(parts[9]))
                maxarea_store_value['Residue'] = max(maxarea_store_value['Residue'], float(parts[9]))

                minarea_store_value['Lacey Graphene'] = min(minarea_store_value['Lacey Graphene'], float(parts[10]))
                maxarea_store_value['Lacey Graphene'] = max(maxarea_store_value['Lacey Graphene'], float(parts[10]))

                minarea_store_value['Wrinkle'] = min(minarea_store_value['Wrinkle'], float(parts[11]))
                maxarea_store_value['Wrinkle'] = max(maxarea_store_value['Wrinkle'], float(parts[11]))
                
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

                get_lastno = int(parts[0])
                
    #print(f"get_lastno:{get_lastno}")

     # Calculate the difference for each category and append to error_categories
    for category in avg_categories:
        #diff = (max_store_value[category]- min_store_value[category])
        #diff = round(diff / get_lastno, 2)
        avg_count = round(float(count_class[category]) / get_lastno, 2)
        avg_categories[category].append(avg_count)
        
    for category in categories:
        #print(f"Before appending: {category} = {count_class[category]} (Type: {type(count_class[category])})")
        categories[category].append(int(count_class[category]))
        #print(f"After appending: {categories}")
        
    for category in errorarea_categories: 
        diff = (maxarea_store_value[category] - minarea_store_value[category])
        #print(f"diff:{diff}")
        #diff = round(diff / 2, 2) #need to divide by 2 because it need value of centre
        diff = round(diff / get_lastno, 2)
        errorarea_categories[category].append(diff)
    #print(f"errorarea:{errorarea_categories}")
        
    for category in area_categories:
        area_categories[category].append(round(float(area_class[category]),2))
       # print(f"area_categories: {area_categories}")
        #count_class['Wrinkle'] = int(count_class['Wrinkle']) + int(parts[6])
        
#print(f"categories{categories}\narea_categories{area_categories}\n errorarea_categories:{errorarea_categories}")

# Get the list of categories and their colors
category_names = list(categories.keys())
num_categories = len(category_names)
num_files = len(error_data)
#xlabelname = 'Transfer method'

# Calculate the width for each group of bars
bar_width = 0.3
index = np.arange(num_files)

# Define colors for each category

color = {'Nucleation': 'lightgreen',
         #'Multilayer of domain': 'blue',
         'Crack': 'purple',
         'Residue': 'red',
         'Lacey Graphene': 'orange',
         'Wrinkle': 'pink'}
'''
color = {'Nucleation': '#CFD231',
             #'Multilayer of domain': 'blue',
             'Crack': 'red',
             'Residue': '#FF701F',
             'Lacey Graphene': '#FFB21D',
             'Wrinkle': 'pink'}
'''
# Create subplots
#plt.subplots(2, 3, figsize=(12, 12))
#Get the screen resolution
#screen = screeninfo.get_monitors()[0]
screen_width,screen_height = 1800, 900
plt.subplots(2, 4, figsize=(screen_width/100, screen_height/100))
filenames = [entry[0] for entry in error_data]
ylabel_names = index + (num_categories - 3) * bar_width / 2
ylabel_names_g = index + (num_categories - 4) * bar_width / 2
fontdict={'family': 'Arial', 'weight': 'bold', 'size': 14}
font = FontProperties(family='Arial', weight='bold', size=14)
#plt.subplots_adjust(left=0.1, right=0.9, bottom=0.5, top=0.9, wspace=0.2, hspace=0.4)
# Plotting the grouped bar graph for the first subplot (Nucleation, Multilayer of domain, Wrinkle)
#plt.subplot(2, 3, 1)
plt.subplot(2, 4, 1)
#plt.title('Growth Graphene')
for i, category in enumerate(['Nucleation', 'Wrinkle']):
#for i, category in enumerate(['Nucleation', 'Wrinkle', 'Multilayer of domain']):
    bar_data = categories[category] + [0] * (num_files - len(categories[category]))
    #print(f"defectcount_bar_data:{bar_data}")
    #error_data = error_categories[category] + [0] * (num_files - len(error_categories[category]))
    plt.bar(index + i * bar_width, bar_data, bar_width, label=category, color=color[category])
   # plt.errorbar(index + i * bar_width, bar_data, yerr=error_data, fmt='none', ecolor='black', capsize=5)
    for j, value in enumerate(bar_data):
        if value != 0:
            plt.text(index[j] + i * bar_width, value, str(value), ha='center', fontdict = fontdict)
#plt.legend()
#plt.xlabel(xlabelname)
plt.ylabel('Defect  Count',  fontdict = fontdict)
plt.xticks(ylabel_names_g, filenames, fontproperties=font )

#plt.subplot(2, 3, 2)
plt.subplot(2, 4, 2)
#plt.title('Growth Graphene')
for i, category in enumerate(['Nucleation', 'Wrinkle']):
    bar_data = avg_categories[category] + [0] * (num_files - len(avg_categories[category]))
    error_data = errorarea_categories[category] + [0] * (num_files - len(errorarea_categories[category]))
    #print(f"Average cnt bar_data:{bar_data} , error_data:{error_data} ")
    # Manually define the lower and upper errors
    lower_error = [min(val, bar_data[j]) for j, val in enumerate(error_data)]  # Clip lower error at zero
    upper_error = error_data  # Keep the upper error as is
    
    plt.bar(index + i * bar_width, bar_data, bar_width, label=category, color=color[category])
    plt.errorbar(index + i * bar_width, bar_data, yerr=[lower_error, upper_error], fmt='none', ecolor='black', capsize=5)
    '''for j, value in enumerate(error_data):
        if value != 0:
            #plt.text(index[j] + i * bar_width, value, str(value),  ha='center', va='center', fontdict = fontdict)
            plt.text(index[j] + i * bar_width, bar_data[j] + upper_error[j] + 0.05, 
                    str(round(value, 2)), ha='center', fontdict=fontdict)
    '''
#plt.legend()
#plt.xlabel(xlabelname)
plt.ylabel('Average Defect Count', fontdict = fontdict)
plt.xticks(ylabel_names_g, filenames, fontproperties=font)

#plt.subplot(2, 3, 3)
plt.subplot(2, 4, 3)
#plt.title('Growth Graphene')
for i, category in enumerate(['Nucleation', 'Wrinkle']):
#for i, category in enumerate(['Nucleation']):
    bar_data = area_categories[category] + [0] * (num_files - len(categories[category]))
   #error_data = error_categories[category] + [0] * (num_files - len(error_categories[category]))
    plt.bar(index + i * bar_width, bar_data, bar_width, label=category, color=color[category])
    #plt.errorbar(index + i * bar_width, bar_data, yerr=error_data, fmt='none', ecolor='black', capsize=5)
    for j, value in enumerate(bar_data):
        if value != 0:
            plt.text(index[j] + i * bar_width, value, f"{value:.2f}", ha='center', fontdict = fontdict)
#plt.legend()
#plt.xlabel(xlabelname)
plt.ylabel('Defect Area Percentage(%)', fontdict = fontdict)
plt.xticks(ylabel_names_g, filenames, fontproperties=font)

plt.subplot(2, 4, 5)
for i, category in enumerate(['Crack', 'Residue', 'Lacey Graphene']):
    bar_data = categories[category] + [0] * (num_files - len(categories[category]))
    plt.bar(index + i * bar_width, bar_data, bar_width, label=category, color=color[category])
    for j, value in enumerate(bar_data):
        if value != 0:
            plt.text(index[j] + i * bar_width, value, str(value), ha='center', fontdict = fontdict)
plt.ylabel('Defect  Count', fontdict = fontdict)
plt.xticks(ylabel_names, filenames, fontproperties=font)            

plt.subplot(2, 4, 6)
for i, category in enumerate(['Crack', 'Residue', 'Lacey Graphene']):
    bar_data = avg_categories[category] + [0] * (num_files - len(avg_categories[category]))
    error_data = errorarea_categories[category] + [0] * (num_files - len(errorarea_categories[category]))
    # Manually define the lower and upper errors
    lower_error = [min(val, bar_data[j]) for j, val in enumerate(error_data)]  # Clip lower error at zero
    upper_error = error_data  # Keep the upper error as is
    plt.bar(index + i * bar_width, bar_data, bar_width, label=category, color=color[category])
    plt.errorbar(index + i * bar_width, bar_data, yerr=[lower_error, upper_error], fmt='none', ecolor='black', capsize=5) #fmt='o' centre point
plt.ylabel('Average Defect Count', fontdict = fontdict)
plt.xticks(ylabel_names, filenames, fontproperties=font)

plt.subplot(2, 4, 7)
#plt.title('Destruct Graphene')
for i, category in enumerate(['Crack', 'Residue', 'Lacey Graphene']):
    bar_data = area_categories[category] + [0] * (num_files - len(categories[category]))
    plt.bar(index + i * bar_width, bar_data, bar_width, label=category, color=color[category])
    for j, value in enumerate(bar_data):
        if value != 0:
            plt.text(index[j] + i * bar_width, value, f"{value:.2f}", ha='center', fontdict = fontdict) 
plt.ylabel('Defect Area Percentage(%)', fontdict = fontdict)
plt.xticks(ylabel_names, filenames, fontproperties=font)

plt.subplot(2, 4, 4)  # Create a dummy subplot to place the legend
plt.axis('off')  # Turn off axis
# Create a single legend for all subplots
plt.legend(handles=[plt.Line2D([0], [0], color=color[label], linewidth=3, linestyle='-') for label in color.keys()],
           labels=color.keys(),
           loc='upper right',
           bbox_to_anchor=(1.1, 1.1),
           prop={'family': 'Arial', 'weight': 'bold', 'size': 14})

plt.subplot(2, 4, 4)  # Create a dummy subplot to place the legend
plt.axis('off')  # Turn off axis
#plt.title('Defect Due to Grow')
plt.text(0.5, 0.5, "Defect Due to Grow", ha='center', va='center', fontdict = fontdict)

plt.subplot(2, 4, 8)  # Create a dummy subplot to place the legend
plt.axis('off')  # Turn off axis
#plt.legend(["Testing", "This is a summary note about the graph."], loc="lower right")
plt.text(0.5, 0.5, "Defect Due to Transfer", ha='center', va='center', fontdict = fontdict)
#plt.text(0.5, 0.4, "conclusion", ha='bottom', va='bottom')

save_path = root_dir + '/summarygraph.jpg'
plt.tight_layout()
plt.savefig(save_path)
plt.show()
#os.system("python segment/Bareachimg.py")