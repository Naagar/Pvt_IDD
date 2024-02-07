import os
import cv2
from retinaface import RetinaFace
import cv2
import numpy as np
# Function to process and save images

## Steps:
# 1. Read the image using OpenCV
# 2. Perform face detection on the image (e.g., resizing, filtering, etc.)
# 3. check if image has face
# 4. If the image has a face, save it to the output folder
# 5. create the mask of the face
# 6. save the mask to the output folder
# 7. save the face cordinateds in the a csv file with image_name.csv and cordinates

def process_and_save_images(input_folder, output_folder_img, output_folder_mask, output_folder_cordinate_text):
    # Create the output folder if it doesn't exist
    # if not os.path.exists(output_folder):
        # os.makedirs(output_folder)

    # List all files in the input folder
    files = os.listdir(input_folder)
    list_img = []
    images_without_face = 0
    images_with_face = 0
    for file in files:
        # Check if the file is an image (you can customize this check based on your needs)
        if file.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            # Read the image using OpenCV
            input_path = os.path.join(input_folder, file)
            image = cv2.imread(input_path)
            list_img.append(input_path)
            print("img_shape:", image.shape)
            print("image path:", input_path)
            # cv2.imread("test_01.jpeg")
            

            # Perform some operations on the image (e.g., resizing, filtering, etc.)
            # Replace the following line with your image processing code
            # processed_image = cv2.resize(image, (200, 200))
            masked_img = np.zeros((image.shape[0],image.shape[1],image.shape[2]),np.float64) # initialize mask

            resp = RetinaFace.detect_faces(image, threshold=0.5)
            # images_without_face = 0
            # images_with_face = 0
            if resp.keys() != "face_1":
                print("Face detected")
                images_with_face += 1
                # save image
                # create a mask
                # save mask
                # save cordinates in a csv file

                masked_img = masked_img * 255 # white mask
                face_cord_list =   []
                for key, value in resp.items():
                    # print(key, 'hjghhgh', value)
                    # print(key, '', value['facial_area']) #, value['landmarks'])
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
                    masked_img[y_min:y_max,x_min:x_max] = 255 # fill with black pixels
                    
                    # plate_image_1[y_min:y_max,x_min:x_max] = 0 # fill with black pixels
                    # save mask
                
                print("masked_image shape:", masked_img.shape)
                cv2.imwrite(f'{output_folder_img}{file}',image) # save image
                cv2.imwrite(f'{output_folder_mask}{file}', masked_img) # save mask
                text_file = os.path.splitext(file)[0] + ".txt"
                with open(f'{output_folder_cordinate_text}{text_file}', 'w') as f:
                    for line in face_cord_list:
                        print(line)
                        f.write(str(line[0]) + " " + str(line[1]) + "\n")
                    # with open(text_filename, "w") as text_file:
                    # for line in data_to_save:
                    #     text_file.write(line + "\n")
            else:
                images_without_face += 1
                print("No face detected")
            # masked_img = np.ones((image.shape[0],image.shape[1],image.shape[2]),np.float64) # initialize mask
            # masked_img = masked_img * 255 # white mask
            # face_cord_list =   []

            # Save the processed image to the output folder
            # output_path = os.path.join(output_folder, file)
            # cv2.imwrite(output_path, processed_image)

            print(f"Processed and saved: {file}")
    with open("idd_face_fake_big_Lama_img_list.txt", "w") as output:
        output.write(str(list_img))
    print("images_without_face:", images_without_face) 
    print("images_with_face:", images_with_face)

if __name__ == "__main__":
    # Input folder containing images
    input_folder = "idd_face_fake_bigLama/"

    # Output folder for processed images
    output_folder_img   = "idd_face_fake_big_Lama/images/"
    output_folder_mask  = "idd_face_fake_big_Lama/masks/"
    output_folder_cordinate_text = "idd_face_fake_big_Lama/cordinates/"
    # Call the function to process and save the images
    process_and_save_images(input_folder, output_folder_img, output_folder_mask, output_folder_cordinate_text)
