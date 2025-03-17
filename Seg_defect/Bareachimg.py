# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 09:22:26 2024

@author: user
"""
import os
import matplotlib.pyplot as plt
import re

x_label ='Type'
y_label ='Defect Area %'
bar_title ='Nucleation Area Percentage -'
split_name = "_dbresult.txt"
save_file = "MuStr5M811Total180img317X20_Gph"
def addlabels(ax, x, y, z):
    fontdict={'family': 'Arial', 'weight': 'bold', 'size': 14}
    for i in range(len(x)):
        y_label = f"{y[i]:.2f}%"  # Concatenate y value with %
        ax.text(i, y[i], y_label , ha= 'center', fontdict = fontdict)
        # Convert z[i] to a float for comparison
        '''
        z_value = float(z[i])
        color = 'blue' if z_value > 0 else 'red'
        ax.text(i, y[i]-5, z[i] , ha= 'center', fontsize =14, color=color)
        '''
        

# Function to read data from file
def read_file(file_path):
    with open(file_path, 'r') as file:
        # Read the header line
        next(file).strip().split(',')
        
        # Initialize list to store data
        file_data = []
        
        # Process each line in the file
        for line in file:
            parts = line.strip().split(',')
            file_data.append(parts)
            
    return file_data

def count_files(root_dir):
    file_count = 0
    file_names = ""
    for dirpath, _, filenames in os.walk(root_dir):
        #print(f"file_count{file_count}, {filenames}")
        for filename in filenames:
            if filename.endswith(split_name):
                file_count += 1
                file_names = filename
                #print(f"file_count{file_count}, {filenames}")
    return file_count, file_names

def sort_data(column_1_data, column_9_data, column_2_data):
    '''# Extract numeric part for sorting
    def extract_number(filename):
        try:
            return int(filename.split('_')[0])
        except ValueError:
            #print(f"valueError")
            return int(filename.split('.')[0])#float('inf')  # Handle non-numeric filenames gracefully
    ''' 
    # Extract numeric part for sorting
    def extract_number(filename):
        # Find all numeric parts in the filename
        numbers = re.findall(r'\d+', filename)
        if numbers:
            return int(numbers[0])  # Return the first found number
        return float('inf')  # Return a large number if no numbers are found

    # Zip the data, sort by numeric part, and unzip
    sorted_data = sorted(zip(column_1_data, column_9_data, column_2_data), key=lambda x: extract_number(x[0]))
    sorted_column_1_data, sorted_column_9_data, sorted_column_2_data = zip(*sorted_data)
    
    return list(sorted_column_1_data), list(sorted_column_9_data), list(sorted_column_2_data)

def plot_data(root_dir):
  
    # Initialize lists to store all data
    all_column_1_data = []
    all_column_2_data = [] #addcount
    all_column_9_data = []
    all_filenames = []
    # Walk through the directory
    for dirpath, _, filenames in os.walk(root_dir):    
        
        for filename in filenames:
            if filename.endswith(split_name):
                file_path = os.path.join(dirpath, filename)
                file_data = read_file(file_path)
                #print(f"file_data:{file_data}")
                # Initialize arrays to store data for current file
                column_1_data = []
                column_2_data = []#addcount
                column_9_data = []
                
                # Process each line in file_data
                for row in file_data:
                    col_1_value = row[1]  # Assuming Filename is in column 1 (index 1)
                    col_2_value = row[2] #addcount
                    #col_9_value = round(float(row[8])+ float(row[9]),2)# Assuming Area_Nucleation is in column 9 (index 8)
                    col_9_value = round(float(row[7]),2)
                    #print(f"col_9_value:{col_9_value}")
                    
                    # Append to current file's data lists
                    column_1_data.append(col_1_value)
                    column_2_data.append(col_2_value) #addcount
                    column_9_data.append(col_9_value)
                
                # Append current file's data to all data lists
                all_column_1_data.append(column_1_data)
                all_column_2_data.append(column_2_data)#addcount
                all_column_9_data.append(column_9_data)
                all_filenames.append(filename)
    
    fontdict={'family': 'Arial', 'weight': 'bold', 'size': 14}
    # Plotting all data in a single figure
    fig, axs = plt.subplots(len(all_column_1_data), figsize=(10, 6 * len(all_column_1_data)))
    for i, (column_1_data, column_9_data, column_2_data, filename) in enumerate(zip(all_column_1_data, all_column_9_data, all_column_2_data, all_filenames)):
        # Sort data
        sorted_column_1_data, sorted_column_9_data, sorted_column_2_data = sort_data(column_1_data, column_9_data, column_2_data)
        #print(f"sorted_column:{sorted_column_1_data}")
        axs[i].bar(sorted_column_1_data, sorted_column_9_data, color='lightgreen')
        axs[i].set_xlabel(x_label, fontsize =18)
        axs[i].set_ylabel(y_label,fontsize =18)
        split_dirname = filename.split('_', 1)[0]
        axs[i].set_title(f' {bar_title} {split_dirname}',fontdict =fontdict)
        axs[i].tick_params(axis='x', rotation=45)
        addlabels(axs[i], sorted_column_1_data, sorted_column_9_data, sorted_column_2_data)
        #addlabels(axs[i],sorted_column_1_data, sorted_column_2_data, 'bottom')
        axs[i].grid(False)
    
    plt.ylim(0, max(sorted_column_9_data) * 1.2)  # Add extra space at the top #NEW
    plt.tight_layout(pad=1.0)
    
    # Save all plots into one image file
    save_path = os.path.join(root_dir, save_file)
    plt.savefig(save_path)
    
    plt.show()
    
def plot_datanew(root_dir):
    # Initialize lists to store all data
    all_column_1_data = []
    all_column_2_data = []  # addcount
    all_column_9_data = []
    all_filenames = []

    # Walk through the directory
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith(split_name):
                file_path = os.path.join(dirpath, filename)
                file_data = read_file(file_path)
                
                # Initialize arrays to store data for current file
                column_1_data = []
                column_2_data = []  # addcount
                column_9_data = []

                # Process each line in file_data
                for row in file_data:
                    col_1_value = row[1]  # Assuming Filename is in column 1 (index 1)
                    col_2_value = row[2]  # addcount
                    col_9_value = round(float(row[7]), 2)  # Assuming Area_Nucleation is in column 9 (index 8)

                    # Append to current file's data lists
                    column_1_data.append(col_1_value)
                    column_2_data.append(col_2_value)  # addcount
                    column_9_data.append(col_9_value)

                # Append current file's data to all data lists
                all_column_1_data.append(column_1_data)
                all_column_2_data.append(column_2_data)  # addcount
                all_column_9_data.append(column_9_data)
                all_filenames.append(filename)
    
    # Plotting all data in separate figures for each group of 15
    group_size = 15
    
    for j in range(len(all_column_1_data)):
        num_plots = (len(all_column_1_data[j]) + group_size - 1) // group_size  # Calculate the number of plots needed
        
        for group_idx in range(num_plots):
            fig, axs = plt.subplots(figsize=(10, 8))  # Adjust the figure size as needed
            
            start_idx = group_idx * group_size
            end_idx = min(start_idx + group_size, len(all_column_1_data[j]))
            
            # Slicing the data for this group
            current_column_1_data = all_column_1_data[j][start_idx:end_idx]
            current_column_2_data = all_column_2_data[j][start_idx:end_idx]
            current_column_9_data = all_column_9_data[j][start_idx:end_idx]
            current_filename = all_filenames[j]
            
            # Sort data
            sorted_column_1_data, sorted_column_9_data, sorted_column_2_data = sort_data(current_column_1_data, current_column_9_data, current_column_2_data)
            
            axs.bar(sorted_column_1_data, sorted_column_9_data, color='#CFD231')
            axs.set_xlabel(x_label, fontsize=18)
            axs.set_ylabel(y_label, fontsize=18)
            split_dirname = filename.split('_', 1)[0]
            axs.set_title(f'{bar_title} {split_dirname} (Group {group_idx + 1})', fontsize=16)
            axs.tick_params(axis='x', rotation=45)
            addlabels(axs, sorted_column_1_data, sorted_column_9_data, sorted_column_2_data)
            axs.grid(True)
            
            # Save the figure
            save_path = os.path.join(root_dir, f"{save_file}_{current_filename}_group_{group_idx + 1}.png")
            plt.tight_layout()
            plt.savefig(save_path)
            plt.show()
        
def addlabels_nor(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center' ,fontsize =14)
        
        
def read_onefile(root_dir):   
    # Read the data from the file
    file_data = []
    directory, directory_name = os.path.split(root_dir)
    split_dirname,_ = directory_name.split('_')
    with open(root_dir, 'r') as file:
        # Read the header line
        next(file).strip().split(',')
    
        # Process each line in the file
        for line in file:
            parts = line.strip().split(',')
            file_data.append(parts)
           
    
    # Initialize arrays to store data
    column_1_data = []
    column_9_data = []
    
    # Iterate through each row (skipping the header row)
    for row in file_data:
        # Extract data from column 1 and column 9
        col_1_value = row[1]  # Column 1 corresponds to index 1 (Filename)
        #col_9_value = round(float(row[8])+ float(row[9]),2)  # Column 9 corresponds to index 8 (Area_Nucleation)
        col_9_value = round(float(row[7]),2)
        
        # Append data to respective arrays
        column_1_data.append(col_1_value)
        column_9_data.append(col_9_value)
    #print(f"column_1_data:{column_1_data}\n column_9_data:{column_9_data}")
    
    # Sort data
    sorted_column_1_data, sorted_column_9_data = sort_data(column_1_data, column_9_data)
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(sorted_column_1_data, sorted_column_9_data, color='lightgreen')
    addlabels_nor(sorted_column_1_data, sorted_column_9_data)
    plt.xlabel(x_label, fontsize =18)
    plt.ylabel(y_label,fontsize =18)
    plt.title(f' {bar_title} {split_dirname}',fontsize =16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    save_path = directory + '/Bar'+directory_name+'.jpg'
    plt.savefig(save_path)
    plt.show()
    
# Example usage
root_dir = input("Please type file location and name: ")
directory, directory_name = os.path.split(root_dir)
#print(f"directory:{directory}, directory_name:{directory_name}")

cnt_file,file_name =count_files(root_dir)
#print(f"cnt_file{cnt_file}")

if directory_name == "" :
    if cnt_file < 2:
        root_dirpath = root_dir+'/'+file_name
        read_onefile(root_dirpath)
    else:
        plot_data(root_dir)
else:
    read_onefile(root_dir)


'''
import matplotlib.pyplot as plt
import numpy as np



    # Read the file location and name
    root_dir = input("Please type file location and name: ")
def addlabels(x,y):
    for i in range(len(x)):
        plt.text(i, y[i], y[i], ha = 'center')
def read_onefile(root_dir):   
    # Read the data from the file
    file_data = []
    
    with open(root_dir, 'r') as file:
        # Read the header line
        header = next(file).strip().split(',')
    
        # Process each line in the file
        for line in file:
            parts = line.strip().split(',')
            file_data.append(parts)
           
    
    # Initialize arrays to store data
    column_1_data = []
    column_9_data = []
    
    # Iterate through each row (skipping the header row)
    for row in file_data:
        # Extract data from column 1 and column 9
        col_1_value = row[1]  # Column 1 corresponds to index 1 (Filename)
        col_9_value = float(row[8])  # Column 9 corresponds to index 8 (Area_Nucleation)
        
        # Append data to respective arrays
        column_1_data.append(col_1_value)
        column_9_data.append(col_9_value)
    print(f"column_1_data:{column_1_data}\n column_9_data:{column_9_data}")
    # Plotting
    plt.figure(figsize=(10, 6))
    plt.bar(column_1_data, column_9_data, color='lightgreen')
    addlabels(column_1_data, column_9_data)
    plt.xlabel('Filenames')
    plt.ylabel('Area %')
    plt.title('Nucleation Area Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
'''

