import cv2
import os
import sys
import argparse
import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from typing import Any, Dict, List
import time
from sam_segment import predict_masks_with_sam
from lama_inpaint import inpaint_img_with_lama

from stable_diffusion_inpaint import fill_img_with_sd
from utils import load_img_to_array, load_mask_to_array, save_array_to_img, dilate_mask, erode_mask, \
    show_mask, show_points, get_clicked_point

# for import error $ python -m pip install -e segment_anything

def setup_args(parser):
    parser.add_argument(
        "--input_imgs_path", type=str, required=True,
        help="Path to a input imgs",
    )
    parser.add_argument(
        "--input_masks_path", type=str, required=True,
        help="Path to a input masks",
    )
    parser.add_argument(
        "--point_labels", type=int, nargs='+', required=True,
        help="The labels of the point prompt, 1 or 0.",
    )
    parser.add_argument(
        "--text_prompt", type=str, required=True,
        help="Text prompt",
    )
    parser.add_argument(
        "--steps", type=int,
        help="steps for the denoising .",
    )
    parser.add_argument(
        "--dilate_kernel_size", type=int, default=None,
        help="Dilate kernel size. Default: None",
    )
    parser.add_argument(
        "--erode_kernel_size", type=int, default=None,
        help="erode kernel size. Default: None",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output path to the directory with results.",
    )
    parser.add_argument(
        "--sam_model_type", type=str,
        default="vit_h", choices=['vit_h', 'vit_l', 'vit_b', 'vit_t'],
        help="The type of sam model to load. Default: 'vit_h"
    )
    parser.add_argument(
        "--sam_ckpt", type=str, required=True,
        help="The path to the SAM checkpoint to use for mask generation.",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Specify seed for reproducibility.",
    )
    parser.add_argument(
        "--deterministic", action="store_true",
        help="Use deterministic algorithms for reproducibility.",
    )

    parser.add_argument(
        "--lama_config", type=str,
        default="./lama/configs/prediction/default.yaml",
        help="The path to the config file of lama model. "
             "Default: the config of big-lama",
    )
    parser.add_argument(
        "--lama_ckpt", type=str, required=True,
        help="The path to the lama checkpoint.",
    )


if __name__ == "__main__":
    """Example usage:
    python fill_anything.py \
        --input_img FA_demo/FA1_dog.png \
        --point_labels 1 \
        --text_prompt "a teddy bear on a bench" \
        --dilate_kernel_size 15 \
        --output_dir ./results \
        --sam_model_type "vit_h" \
        --sam_ckpt sam_vit_h_4b8939.pth 
    """
    parser = argparse.ArgumentParser()
    setup_args(parser)
    args = parser.parse_args(sys.argv[1:])
    print('argumenst: ', args)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    out_dir = args.output_dir # Path(args.output_dir) / img_stem
    tic = time.time()
    os.mkdir(out_dir + f'/{tic}')
    

    images =  args.input_imgs_path  #'idd_with_faces_all_images/images' #images path
    masks  =  args.input_masks_path #'idd_with_faces_all_images/masks' # masks path

    image_list_file = 'idd_with_faces_all_images/all_images_names.txt' #test_images_list.txt # list of images names, all_images_names.txt

    def read_image_list_from_file(image_list_file):
        # Read image names from the text file
        with open(image_list_file, 'r') as file:
            image_names = file.read().splitlines()
        return image_names

    image_list = read_image_list_from_file(image_list_file)

    # img_stem = Path(img_points_p).stem
    
    # out_dir.mkdir(parents=True, exist_ok=True)

    total_fill_faces = 0 # count
    for img_p in image_list:
        print('img_p image name: ', img_p)
        total_fill_faces += 1

        img_points_p =  images + img_p
        img_mask_p = masks + img_p
        img = load_img_to_array(img_points_p)
        mask = load_mask_to_array(img_mask_p)
        print('img.shape: ', img.shape)
        print('mask.shape: ', mask.shape)
        
        # dilate mask to avoid unmasked edge effect, not required for fill faces
        if args.dilate_kernel_size is not None:
            mask = dilate_mask(mask, args.dilate_kernel_size) #[dilate_mask(mask, args.dilate_kernel_size) for mask in masks]
        if args.erode_kernel_size is not None:
            mask = erode_mask(mask, args.erode_kernel_size)
        # print('masks len, number of maks created: ', len(masks))
        mask = mask[:,:,0]  # mask.shape:  (1080, 1920)
        if args.seed is not None:
            torch.manual_seed(args.seed)
        mask_p = out_dir + f"mask_{img_p}.png"
        
        img_filled_p = out_dir + f'/{tic}' + '/SD' + '_' + img_p #f"/filled_with_{Path(mask_p).name}"
        # print('mask.shape', mask.shape)
        # os.mkdir(out_dir + f'/{tic}' + '/stable_diffusion/')
        # img_filled = fill_img_with_sd(
        #     img, mask, args.text_prompt, step = args.steps, device=device)
        # save_array_to_img(img_filled, img_filled_p)
        img_filled_p = out_dir + f'/{tic}' + '/big_lama' +  '_' + img_p #f"/filled_with_{Path(mask_p).name}"
        # os.mkdir(out_dir + f'/{tic}' + '/big_lama/')
        
        img_inpainted = inpaint_img_with_lama(
            img, mask, args.lama_config, args.lama_ckpt, device=device)
        save_array_to_img(img_inpainted, img_filled_p)
        if total_fill_faces % 10 == 0:
            print('faces and images done: ', total_fill_faces)

    print('total_fill_faces: ', total_fill_faces)

    # img = load_img_to_array(args.input_img)
    # print('img.shape: ', img.shape)cd ..


    # masks, _, _ = predict_masks_with_sam(
    #     img,
    #     [latest_coords],
    #     args.point_labels,
    #     model_type=args.sam_model_type,
    #     ckpt_p=args.sam_ckpt,
    #     device=device,
    # )
    # masks = masks.astype(np.uint8) * 255
    # print('masks.shape: ', masks.shape)

    # dilate mask to avoid unmasked edge effect
    # if args.dilate_kernel_size is not None:
    #     masks = [dilate_mask(mask, args.dilate_kernel_size) for mask in masks]
    # print('masks len, number of maks created: ', len(masks))
    
    # visualize the segmentation results
    # img_stem = Path(args.input_img).stem
    # out_dir = Path(args.output_dir) / img_stem
    # out_dir.mkdir(parents=True, exist_ok=True)
    # for idx, mask in enumerate(masks):
    #     # path to the results
    #     mask_p = out_dir / f"mask_{idx}.png"
    #     img_points_p = out_dir / f"with_points.png"
    #     img_mask_p = out_dir / f"with_{Path(mask_p).name}"

    #     # save the mask
    #     save_array_to_img(mask, mask_p)

    #     # save the pointed and masked image
    #     dpi = plt.rcParams['figure.dpi']
    #     height, width = img.shape[:2]
    #     plt.figure(figsize=(width/dpi/0.77, height/dpi/0.77))
    #     plt.imshow(img)
    #     plt.axis('off')
    #     show_points(plt.gca(), [latest_coords], args.point_labels,
    #                 size=(width*0.04)**2)
    #     plt.savefig(img_points_p, bbox_inches='tight', pad_inches=0)
    #     show_mask(plt.gca(), mask, random_color=False)
    #     plt.savefig(img_mask_p, bbox_inches='tight', pad_inches=0)
    #     plt.close()

    # # fill the masked image
    # for idx, mask in enumerate(masks):
    #     if args.seed is not None:
    #         torch.manual_seed(args.seed)
    #     mask_p = out_dir / f"mask_{idx}.png"
    #     img_filled_p = out_dir / f"filled_with_{Path(mask_p).name}"
    #     print('mask.shape', mask.shape)
    #     img_filled = fill_img_with_sd(
    #         img, mask, args.text_prompt, device=device)
    #     save_array_to_img(img_filled, img_filled_p)