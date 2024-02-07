#%%
import numpy as np
import cv2

#%%
def invert_mask(mask):
    # Perform bitwise NOT operation
    inverted_mask = cv2.bitwise_not(mask)
    return inverted_mask

# Replace 'path/to/mask_image.png' with the actual path to your mask image
mask_image_path = 'idd_with_faces_all_images/masks/000216_leftImg8bit.png'

#%%
from PIL import Image
import PIL.ImageOps    

image = Image.open(mask_image_path)

inverted_image = PIL.ImageOps.invert(image)

# inverted_image.save('new_name.png')

#%%
# Read the mask image

mask = cv2.imread(mask_image_path, cv2.IMREAD_GRAYSCALE)

if mask is not None:
    # Invert the mask
    inverted_mask = invert_mask(mask)

    # Display the original and inverted masks
    cv2.imshow('Original Mask', mask)
    cv2.imshow('Inverted Mask', inverted_mask)
    cv2.imshow('Inverted Mask by PIL', inverted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Unable to read the mask image.")
 #%%