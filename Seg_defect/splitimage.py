# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 21:45:44 2025

@author: susu
"""

from PIL import Image
import os

# Open the image
folder_path = input("Please type folder path : ")
file_name = input("File name:  ")
file_dir = folder_path+"\\"+file_name
image = Image.open(file_dir)

# Get the size of the image
width, height = image.size
folder_name = folder_path+"\\"+"split"
if not os.path.exists(folder_name):
    os.makedirs(folder_name)
else:
    #increment folder name if it already exists
    counter=0
    while os.path.exists(folder_name):
        counter += 1
        folder_name = os.path.join(folder_path, f"split{counter}")
    os.makedirs(folder_name)

# Define the number of rows and columns for the grid
rows = 5
cols = 5

# Calculate the size of each piece
piece_width = width // cols
piece_height = height // rows

print(f"piece_width:{piece_width} , piece_height{piece_height}")

# Create a list to store the pieces
pieces = []

# Split the image into 100 pieces
for row in range(rows):
    for col in range(cols):
        # Define the box to extract the piece
        left = col * piece_width
        upper = row * piece_height
        right = left + piece_width
        lower = upper + piece_height
        
        # Crop the image to get the piece
        piece = image.crop((left, upper, right, lower))
        
        # Resize the piece (for example, doubling the size)
        resized_piece = piece.resize((piece_width*4, piece_height*4))
        print(f"left:{left}, upper:{upper}, right:{right}, lower:{lower}\n piece:{piece}, resized_piece:{resized_piece}")
        # Append the resized piece to the list
        pieces.append(resized_piece)

# Optional: To save or display the pieces
for i, piece in enumerate(pieces):
    save_img = folder_name+"\\"+f"piece_{i + 1}.jpg"
    piece = piece.convert("RGB")
    piece.save(save_img)  # Saving each piece
   # piece.show()  # Display each piece
