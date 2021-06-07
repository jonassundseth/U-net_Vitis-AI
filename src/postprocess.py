import os
import argparse
import cv2
import SimpleITK as sitk
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def single_class_dice_coefficient(mask, target, k): 
    #flatten label and prediction tensors
    mask = mask.flatten()
    target = target.flatten()
    dice = np.sum(mask[target==k]==k)*2.0 / (np.sum(mask[mask==k]==k) + np.sum(target[target==k]==k))                            
    
    return dice


def dice_coefficient(mask, target):
    '''
    Calculate Dice score
    Input:
        Mask: The predicted segmentation
        Target: The ground truth segmentation
    Output:
        Dice score
    '''
    dice = 0
    num_labels = int(np.max(target))
    for k in range(num_labels):
        dice += single_class_dice_coefficient(mask, target, k)
    return dice/num_labels #averaging


def upsample(image, h, w):
    '''
    Function which upsamples output to the same form as the ground truth segmentation mask
    Input:
        image: mask
        h, w : Target dimensions for the output mask
    '''
    # Reverse engineering the preprocessing
    img_h, img_w = np.shape(image)
    assert img_h == img_w
    #Removing the zero padding
    if h > w:
        z_h, z_w = img_h, img_h * w / h
    else:
        z_h, z_w = img_h * h / w, img_h   

    if z_h > z_w:
        num_pads = z_h - z_w
        left, right = int(num_pads/2), int(num_pads/2)
        if num_pads % 2:
            left += 1
        image = image[:, left:-right]
    else:
        num_pads = w - h
        up, down = int(num_pads/2), int(num_pads/2)
        if num_pads % 2:
            down += 1
        image = image[up:-down, :]
    image = np.array(image, dtype='uint8')

    return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)


def get_masks(root_dir):
    '''
    Input: 
        root_dir: Path to root directory containing the segmentation masks
    Returns:
        Numpy array containing a path to every input file
    '''
    file_list = []
    for _, _, files in os.walk(root_dir):
        file_list.extend(files)
        break
    
    sequences = []
    for file in files:
        if (".png" in file) and ("patient" in file):
            sequences.append(root_dir + file)
                            
    return np.array(sequences)


def get_ground_truth(sequences, root_dir):
    '''
    Input: 
        root_dir: Path to root directory containing the segmentation ground truth masks
    Returns:
        Numpy array containing a path to every input file
    '''
    ground_truth = []
    for entry in sequences:
        file_name = entry.split('_mask_')[-1]
        gt_path = root_dir + file_name.split('_')[0] + "/" + file_name[:-4] + "_gt.mhd"
        ground_truth.append(gt_path)
    return ground_truth


def _options():
    """Function for taking in arguments from user
    Returns:
        Arguments from user
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plot", 
                        action="store_true",
                        help="Plot resulting masks")
    parser.add_argument("--dim", 
                        type=str,
                        required=True, 
                        help="Dimensions to post process")

    return parser.parse_args() 


def _main():
    _opt = _options()
    fpga_output_dir = "../FPGA_output/" + _opt.dim + "x" + _opt.dim + "/"
    testing_set_dir = "models/datasets/testing/"
    masks = get_masks(fpga_output_dir)
    ground_truth = get_ground_truth(masks, testing_set_dir)
    
    avg_dice = []
    for i, mask in enumerate(masks):
        im_frame = Image.open(mask)
        img = np.array(im_frame)/64
        gt = sitk.GetArrayFromImage(sitk.ReadImage(ground_truth[i], sitk.sitkFloat32))[0]

        h, w = np.shape(gt)
        upsampled_img = upsample(img, h, w)
        
        dice = dice_coefficient(mask=upsampled_img, target=gt)
        avg_dice.append(dice)

        if _opt.plot:
            plt.gray()
            plt.suptitle("Dice score: " + str(dice))
            plt.subplot(1,3,1) 
            plt.imshow(img)
            plt.title("Mask")
            plt.subplot(1,3,2) 
            plt.imshow(upsampled_img)
            plt.title("Upsampled Mask")
            plt.subplot(1,3,3) 
            plt.imshow(gt)  
            plt.title("Ground truth") 
            plt.show(block=False)
            plt.pause(2)
            plt.close()
    

    print("Average dice score = " + str(np.mean(avg_dice)))
    print("Variance dice score = " + str(np.var(avg_dice)))


if __name__ == "__main__":
    _main()
    
    