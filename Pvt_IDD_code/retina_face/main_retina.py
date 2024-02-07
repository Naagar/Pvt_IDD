from retinaface import RetinaFace
import cv2
import numpy as np
# import matplotlib.pyplot as plt
# Steps
# Read the image Done
# size of image Done
# detect the faces using RetinaFace (threshold=0.5) Done (face detection confidance (score): 0.50)
# check for the number of faces =! 0
# check the confidance for each face (score)
# create mask for each face (75%) (facial_area) Coordinates Bbox: x,y,w,h = [1811, 850, 1948, 1013] # one of the bounding boxes, x_min, y_min, x_max, y_max = annotation["bbox"]
# apply the mask to the image
# CoPaint in /retinaface/dir 
# use masked image for image impanting (RePaint (not good, works for small size images), CoPaint (run $conda activate copaint), Lama (links not working))
# validate using result = DeepFace.verify(img1_path = "img1.jpg", img2_path = "img2.jpg") 

#todo   todo
# Step-0: Dataset prepration
    # mask and image names is a format
        # image1_mask001.png
        # image1.png
        # image2_mask001.png
        # image2.png
# Step-1:
    # All steps for single image
    # Steps:
    # 1. Read the image using OpenCV 
    # 2. Perform face detection on the image (e.g., resizing, filtering, etc.) 
    # 3. check if image has face
    # 4. If the image has a face, save it to the output folder
    # 5. create the mask of the face (75%) 
    # 6. save the mask to the output folder with image_name_mask.png
    # 7. save the face cordinateds in the a csv file with image_name.csv and cordinates

#using opencv to read an image
#BGR Image
plate_image_1 = cv2.imread("test_01.jpeg")# ('599036_leftImg8bit.png') #("test_01.jpeg")
print("size of the image", plate_image_1.shape)  # size of the image

# RetinaFace offers a face detection function. It expects an exact path of an image as input. more: Face Recognition, extract_faces,  detect_faces.
# detect_faces(img_path = img, threshold = threshold, model = model, allow_upscaling = allow_upscaling)
resp = RetinaFace.detect_faces(plate_image_1, threshold=0.3)
print("face detection confidance (score):", resp["face_1"]['score'])

# masked image initialization
# masked_img = plate_image_1 #np.zeros((img.shape[0],img.shape[1]),dtype=np.uint8) # initialize mask
masked_img = np.ones((plate_image_1.shape[0],plate_image_1.shape[1],plate_image_1.shape[2]),np.float64) # initialize mask
masked_img = masked_img * 255 # white mask
# x_cord, y_cord = 0, 0
face_cord_list =   []
for key, value in resp.items():
    # print(key, 'hjghhgh', value)
    print(key, '', value['facial_area']) #, value['landmarks'])
    x_min, y_min, x_max, y_max = value['facial_area']
    x_range = x_max - x_min
    y_range = y_max - y_min
    x_cord = int(x_min + (x_range/2))
    y_cord = int(y_min + (y_range/2))
    # if x_range > 20 and y_range > 20:
    x_in = int(x_range * 0.10)
    y_in = int(y_range * 0.10)
    face_cord_list.append([x_cord, y_cord])
    x_min = x_min + x_in
    x_max = x_max - x_in
    y_min = y_min + y_in
    y_max = y_max - y_in
    masked_img[y_min:y_max,x_min:x_max] = 0 # fill with black pixels
    plate_image_1[y_min:y_max,x_min:x_max] = 0 # fill with black pixels
    # save mask
cv2.imwrite('test_idd_maske.png',masked_img)
cv2.imwrite('599036_leftImg8bit_masked.png', plate_image_1)

print("face area:", resp["face_1"]['facial_area'])
print("face_1 keys:", resp["face_1"].keys())

if resp.keys() != "face_1":
    print("Face detected")
print("number of detected faces:", len(resp.keys())) 
print("faces cordinates:", face_cord_list)
# print(len(resp.values()))


# print(resp.dtype)


# print(resp.shape, resp.dtype)

# Step-2
    # for all the images in the folder (test and train)

# return the facial area coordinates and some landmarks (eyes, nose and mouth) with a confidence score.
# {
#     "face_1": {
#         "score": 0.9993440508842468,
#         "facial_area": [155, 81, 434, 443],
#         "landmarks": {
#           "right_eye": [257.82974, 209.64787],
#           "left_eye": [374.93427, 251.78687],
#           "nose": [303.4773, 299.91144],
#           "mouth_right": [228.37329, 338.73193],
#           "mouth_left": [320.21982, 374.58798]
#         }
#      }
# }
