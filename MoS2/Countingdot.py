import cv2
import math

string_char = input("Please type file location and name : ")
string_name = input("Please type file name : ")
st_fullname = string_char+'\\'+string_name
#img = cv2.imread('D:/NCU/Lab/OM images/MOS2/1/3OM5_0.png')
img = cv2.imread(st_fullname)
#print(st_fullname)
blur = cv2.medianBlur(img, 3)
h,w, _ = img.shape
print('width: ', w)
print('height: ', h)

gray = cv2.cvtColor(blur, cv2.COLOR_BGR2GRAY)
#cv2.imshow('Gray image', gray)
cv2.imwrite(string_char+'\\gray'+string_name,gray)
thresh = cv2.threshold(gray,93,255, cv2.THRESH_BINARY_INV)[1]

#cv2.imshow('thresh image', thresh)
cv2.imwrite(string_char+'\\thresh'+string_name,thresh)
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#print(cnts)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]

min_area = -1
black_dots = []
for c in cnts:
   area = cv2.contourArea(c)
   #print('area :' + str(area) )
  # print('c :' + str(c) +', count: ' + str(len(black_dots)))
   if area > min_area:
      cv2.drawContours(img, [c], -1, (36, 255, 12), 2)
      black_dots.append(c)

print("Black Dots Count is:",len(black_dots))
#cv2.imshow('Output image', img)
cv2.imwrite(string_char+'\\output'+string_name,img)
#cv2.waitKey()
