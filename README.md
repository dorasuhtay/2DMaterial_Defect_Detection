# 🧠 Thesis Title
High-Precision and Rapid Detection of Complex Defects in Transferred 2D Materials Enabled by Machine Learning Algorithms

## 📄 Overview
This repository contains the code, models, and datasets used in my Master's thesis at National Central University. The research focuses on using AI (specifically YOLOv7 and image processing techniques) for high-precision and real-time detection of defects in 2D materials like graphene.
![Image](https://github.com/user-attachments/assets/f14159bf-1504-4656-8046-e83b09b8ad63)

## 🎯 Objectives
Develop a robust AI-based detection system for irregularly shaped defects in 2D materials.

Automate annotation and segmentation for efficient dataset preparation.

![Image](https://github.com/user-attachments/assets/2c691b18-eee3-49ea-9e87-2e8ed845922a)
![Image](https://github.com/user-attachments/assets/c0302862-92dd-4802-845f-dab1f8baed6f)
![Image](https://github.com/user-attachments/assets/2ab6ece1-0c36-4a15-a920-7e35a5d71e82)
![Image](https://github.com/user-attachments/assets/7677233b-e324-419d-aae3-ead0c2bc31de)

Optimize YOLOv7 with custom loss functions and HSV tuning for better accuracy.

Evaluate performance based on precision, recall, and accuracy for each defect type.

![Image](https://github.com/user-attachments/assets/afe7388f-1e5d-4bfc-9ab6-b9e78e2ce357)
![Image](https://github.com/user-attachments/assets/44f26e87-9cae-4ff2-a207-e961ef981bce)

## 🧪 Methodology
Model: YOLOv7 for object detection

Loss Function Optimization: Custom loss terms for irregular shapes

Preprocessing: HSV tuning, segmentation, data augmentation

Defect Types: Wrinkle, crack, nucleation, residue, lacey graphene

Training: Batch size 2, 300 epochs, Google Colab Free

## 🤖 Technologies Used
- Python
- YOLOv7
- OpenCV
- Google Colab
- PyTorch

## 📊 Results

Trained on optical microscopy images at 100µm and 10µm.

Real-time detection speed suitable for lab and industrial use.
100µm:

![Image](https://github.com/user-attachments/assets/be950d3f-1763-4c7e-bc52-1731281721f7)

10µm:

![Image](https://github.com/user-attachments/assets/feff16a7-91f9-48a2-9b32-a70240014c12)

## 📸 Sample Output
Add here some sample images with bounding boxes showing defect detection.
![Image](https://github.com/user-attachments/assets/b0237ce4-79e5-4d66-b8bd-14f26a7136a9)
![Image](https://github.com/user-attachments/assets/179b683e-1303-4928-b42a-dd22705fc7cb)
