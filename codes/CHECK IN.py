from skimage.io import imread
from skimage.filters import threshold_otsu
import matplotlib.pyplot as plt
import time
import pandas as pd
import numpy as np
from nltk.tokenize import RegexpTokenizer

filename = './video/1 (5).mp4'

#A B C D E F G H I J K L M N O P Q R S T U V W X Y Z
#0 1 2 3 4 5 6 7 8 9


def correct_text(plate_no):
    plate_no = plate_no.upper()
    tokenizer = RegexpTokenizer("[A-Z0-9]+")
    plate_no = tokenizer.tokenize(plate_no)

    plate_no = "".join(plate_no)
    ans = str("")

    plate_no = plate_no[-10:]
    
    count = 1
    for i,ch in enumerate(plate_no):
        #print("iterator"+str(i))
        if(i==2 or i==6 or i==7 or i==8 or i==9):
            ch=str(Mapping(ch))
          #  print("Mapping"+ch)
        if(i==0 or i==1 or i==4 or i==5):
            ch = str(REVERSE_Mapping(str(ch)))
         #   print("REVERSE_MAPPING"+ch)
            
            
        if (ch>='A' and ch<='Z') or (ch>='0'and ch<='9'):
            ans = ans + str(ch)
            count = count+1

    return ans


def REVERSE_Mapping(ch):
    if(ch == '0'):
        return 'O'
    elif (ch == '1'):
        return 'I'
    elif (ch == '7'):
        return 'L'
    elif (ch =='2'):
        return 'Z'
    elif (ch == '3' or ch=='9'):
        return 'B'
    elif (ch == '8'):
        return B
    elif (ch == '4'):
        return 'A'
    elif (ch =='5'):
        return 'S'
    elif (ch=='6'):
        return 'G'
    else:
        return ch


def Mapping(ch):
    if(ch == 'D' or ch =='O' or ch=='Q' ):
        return 0
    elif (ch == 'I' or ch == 'J'):
        return 1
    elif (ch == 'L' or ch == 'T'):
        return 7
    elif (ch =='Z'):
        return 2
    elif (ch == 'B' or ch=='H' or ch == 'R'):
        return 8
    elif (ch == 'A'):
        return 4
    elif (ch =='S'):
        return 5
    elif (ch=='G'or ch=='C'):
        return 6
    else:
        return ch
            
import cv2
cap = cv2.VideoCapture(filename)
# cap = cv2.VideoCapture(0)
count = 0
while cap.isOpened():
    ret,frame = cap.read()
    if ret == True:
        cv2.imshow('window-name',frame)
        cv2.imwrite("./output/frame%d.jpg" % count, frame)
        count = count + 1
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

# car image -> grayscale image -> binary imageg
import imutils
car_image = imread("./output/frame%d.jpg"%(count-1), as_gray=True)
car_image = imutils.rotate(car_image, 0)



gray_car_image = car_image * 255
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(gray_car_image, cmap="gray")
threshold_value = threshold_otsu(gray_car_image)
binary_car_image = gray_car_image > threshold_value
# print(binary_car_image)
ax2.imshow(binary_car_image, cmap="gray")
# ax2.imshow(gray_car_image, cmap="gray")
plt.show()

# CCA (finding connected regions) of binary image


from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# this gets all the connected regions and groups them together
label_image = measure.label(binary_car_image)

# print(label_image.shape[0]) #width of car img

# getting the maximum width, height and minimum width and height that a license plate can be
plate_dimensions = (0.03*label_image.shape[0], 0.08*label_image.shape[0], 0.15*label_image.shape[1], 0.3*label_image.shape[1])
plate_dimensions2 = (0.08*label_image.shape[0], 0.2*label_image.shape[0], 0.15*label_image.shape[1], 0.4*label_image.shape[1])
min_height, max_height, min_width, max_width = plate_dimensions
plate_objects_cordinates = []
plate_like_objects = []

fig, (ax1) = plt.subplots(1)
ax1.imshow(gray_car_image, cmap="gray")
flag =0
plate_objects=[]
# regionprops creates a list of properties of all the labelled regions
for region in regionprops(label_image):
    # print(region)
    if region.area < 50:
        #if the region is so small then it's likely not a license plate
        continue
        # the bounding box coordinates
    min_row, min_col, max_row, max_col = region.bbox
    # print(min_row)
    # print(min_col)
    # print(max_row)
    # print(max_col)

    region_height = max_row - min_row
    region_width = max_col - min_col
    # print(region_height)
    # print(region_width)

    # ensuring that the region identified satisfies the condition of a typical license plate
    if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        flag = 1
        plate_objects.append( (min_row,min_col,max_row,max_col))
        plate_like_objects.append(binary_car_image[min_row:max_row,
                                  min_col:max_col])
        plate_objects_cordinates.append((min_row, min_col,
                                         max_row, max_col))
        rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                       linewidth=2, fill=False)
        ax1.add_patch(rectBorder)
        # let's draw a red rectangle over those regions




##########################################################################################################################################
##########       Firebase                 ##############################################################################################
import urllib 
import json
import requests
def update_entry(plate , slot , boolean , full , length=4 ):
    
    my_data = dict()
    my_data["slot"] = slot
    my_data["carNumber"] = plate
    my_data["extra charge"] = boolean
    my_data["Full Parking"]=full
    my_data["length"] = length
    json_data = json.dumps(my_data).encode()
    request = urllib.request.Request("https://hackitout-6ee0b.firebaseio.com/alloted slot.json", data=json_data, method="PATCH")

    try:
        loader = urllib.request.urlopen(request)
    except urllib.error.URLError as e:
        message = json.loads(e.read())
        print(message["error"])
    else:
        print(loader.read())

#################################################################################################
############################################################################################################################################
print(flag)
if(flag == 1):
    # print(plate_like_objects[0])
    plt.show()

    x,y,w,h=plate_objects[0]

    car_image = imread("./output/frame%d.jpg"%(count-1) )

        
    car_image = cv2.cvtColor(car_image  , cv2.COLOR_BGR2RGB)
    print(x,y,w,h)
    plate = car_image[x :w,int(y ):h]


    cv2.imshow("plate_no." , plate)
    plt.show()
    
        
    cv2.imwrite("screenshot.jpg" , plate)


    ############ PYTESSERACT 
    import pytesseract
    from PIL import Image
    import matplotlib.pyplot as plt
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    for i in range(1,2):
        value=Image.open("screenshot.jpg")
        im = plt.imread("screenshot.jpg")
        plt.imshow(im)
        plt.show()
        text = pytesseract.image_to_string(value, config='')
        print("actual text" , text)

        text=correct_text(text)
        print("text present in images:",text)
        import bookslot
        
        slot_1= bookslot.bookslot(text)
        print("slot" , slot_1)
        update_entry(text , slot_1[0],slot_1[1],slot_1[2])


if(flag==0):
    min_height, max_height, min_width, max_width = plate_dimensions2
    plate_objects_cordinates = []
    plate_like_objects = []

    fig, (ax1) = plt.subplots(1)
    ax1.imshow(gray_car_image, cmap="gray")

    # regionprops creates a list of properties of all the labelled regions
    for region in regionprops(label_image):
        if region.area < 50:
            #if the region is so small then it's likely not a license plate
            continue
            # the bounding box coordinates
        min_row, min_col, max_row, max_col = region.bbox
      

        region_height = max_row - min_row
        region_width = max_col - min_col
      
        if region_height >= min_height and region_height <= max_height and region_width >= min_width and region_width <= max_width and region_width > region_height:
        
            plate_like_objects.append(binary_car_image[min_row:max_row,
                                      min_col:max_col])
            plate_objects_cordinates.append((min_row, min_col,
                                             max_row, max_col))
            rectBorder = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red",
                                           linewidth=2, fill=False)
            ax1.add_patch(rectBorder)
           
    plt.show()
    x,y,w,h=plate_objects_cordinates[0]

    car_image = imread("./output/frame%d.jpg"%(count-1) )

        
    car_image = cv2.cvtColor(car_image  , cv2.COLOR_BGR2RGB)
    
    plate = car_image[x :w,int(y ):h]


    cv2.imshow("plate_no." , plate)
    plt.show()
    print(plate.shape)
        
    cv2.imwrite("screenshot.jpg" , plate)


    ############ PYTESSERACT 
    import pytesseract
    from PIL import Image
    import matplotlib.pyplot as plt
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    for i in range(1,2):
        value=Image.open("screenshot.jpg")
        im = plt.imread("screenshot.jpg")
        plt.imshow(im)
        plt.show()
        text = pytesseract.image_to_string(value, config='')
        print("actual text" , text)
        
        text=correct_text(text)
        print("text present in images:",text)

        import bookslot
        
        slot_1= bookslot.bookslot(text)
        print(slot_1)
        update_entry(text , slot_1[0],slot_1[1],slot_1[2])
