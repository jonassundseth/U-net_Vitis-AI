from __future__ import print_function, division
import torch
import cv2
import random
import os, sys
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
from medimage import image
from torch.utils.data import Dataset
from skimage import transform
from skimage.filters import sobel
from skimage.draw import circle
from scipy.ndimage.filters import gaussian_filter

sys.path.append("../")
from supress import NoStdStreams

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Code based on code received from Gabriel Kiss, one of my supervisors

class UltrasoundData(Dataset):
    '''
    Dataset class for dataloader
    '''
    def __init__(self, root_dir, transform=None, augment_transform=None, val=False, from_num=350):

        print("---- Initializing " + ("validation" if val else "training") + " dataset ----")

        self.transform = transform
        self.augment = augment_transform
        self.val = val

        files = []
        
        for r, dirs, file in os.walk(root_dir):
            for _dir in dirs:
                for _, _, _files in os.walk(r + "/" + _dir):
                    for _file in _files:
                        files.append(r + _dir + "/" + _file)
            files.extend(file)
            break
        
        sequences = []
        
        for file in files:
            if (".mhd" in file) and ("4CH" in file) and ("gt" not in file) and ("sequence" not in file):
                if int(file.split("patient")[1].split("/")[0]) > (from_num):
                    if val:
                        sequences.append(file)    
                else:
                    if not val:
                        if int(file.split("patient")[1].split("/")[0]) > 0:
                            sequences.append(file)
        
        self.sequences = np.array(sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        with NoStdStreams():
            img = image(self.sequences[idx]).imdata.T[0]
        file_path_gt = self.sequences[idx][:-4] + "_gt.mhd"
        # Segmentation is prone to error when using medimage API.
        cardiac_view = np.array(sitk.GetArrayFromImage(sitk.ReadImage(file_path_gt, sitk.sitkFloat32)))[0]
        
        sample = {'image': img, 'cardiac_view': cardiac_view}

        if self.augment:
            sample = self.augment(sample)
        
        if self.transform:
            sample = self.transform(sample)
        
        # TODO: Remove
        plt.gray()
        plt.subplot(1,2,1), plt.imshow(sample['image'][0])
        plt.subplot(1,2,2), plt.imshow(sample['cardiac_view'])
        plt.show()

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.
    Args:
        output_size (int): Desired output size. The largest of the image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, cardiac_view = sample['image'], sample['cardiac_view']

        h, w = image[:, :].shape[:2]
       
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size * h / w, self.output_size
        else:
            new_h, new_w = self.output_size
        

        new_h_i, new_w_i = int(new_h), int(new_w)
        
        img = cv2.resize(image[:,:], (new_w_i, new_h_i))
        gt = cv2.resize(cardiac_view[:, :], (new_w_i, new_h_i))
        return {'image': img, 'cardiac_view': gt}


class Zero_pad_and_center(object):
    """
    Zero pad the image in a sample to a square dimensions.
    """

    def __call__(self, sample):
        image, cardiac_view = sample['image'], sample['cardiac_view']

        h, w = image[:, :].shape[:2]

        if h > w:
            num_pads = h - w
            left, right = int(num_pads/2), int(num_pads/2)
            if num_pads % 2:
                left += 1
            img = np.pad(image, ([0, 0], [left, right]), mode='constant')
            gt = np.pad(cardiac_view, ([0, 0], [left, right]), mode='constant')
        else:
            num_pads = w - h
            up, down = int(num_pads/2), int(num_pads/2)
            if num_pads % 2:
                down += 1
            img = np.pad(image, ([up, down], [0, 0]), mode='constant')
            gt = np.pad(cardiac_view, ([up, down], [0, 0]), mode='constant')

        return {'image': img, 'cardiac_view': gt}

class ToTensor(object):
    """
    Convert the input image to tensor.
    """

    def __call__(self, sample):
        image, cardiac_view = sample['image'], sample['cardiac_view']
        
        sample = {'image': torch.from_numpy(image).unsqueeze(0).float(),
                  'cardiac_view': torch.tensor(cardiac_view)}

        return sample

class Standardize(object):
    """
    Standardizing the image by subtracting the mean and dividing the result on the standard deviation.
    """
    def __call__(self, sample):
        image, cardiac_view = sample['image'], sample['cardiac_view']
        mean = np.mean(image)
        std = np.std(image)
        image = (image-mean)/std
        sample = {'image': image,
                  'cardiac_view': cardiac_view}

        return sample


class RandomCrop(object):
    '''
    Cropping image randomly to some random size bounded by cropping ratio
    '''
    def __init__(self, crop_ratio):
        self.crop_ratio = 1-crop_ratio

    def __call__(self, sample):
        img, gt = sample['image'], sample['cardiac_view']
        if not random.randint(0, 2):
            h, w = np.shape(img)

            new_h, new_w = int(h*self.crop_ratio), int(w*self.crop_ratio)

            random_top = np.random.randint(0, h-new_h)
            random_left = np.random.randint(0, w-new_w)

            img = img[random_top:random_top+new_h, random_left:random_left+new_w]
            gt = gt[random_top:random_top+new_h, random_left:random_left+new_w]

        return {'image':img, 'cardiac_view':gt}

class RandomRotation(object):
    '''
    Rotating the image by some random angle between -degrees and degrees
    '''
    def __init__(self, degrees):
        self.degrees = degrees

    def __call__(self, sample):
        img, gt = sample['image'], sample['cardiac_view']
        if not random.randint(0, 2):
            deg = np.random.randint(-self.degrees, self.degrees)
            img = transform.rotate(img, deg)
            gt =  transform.rotate(gt, deg)

        return {'image':img, 'cardiac_view':gt}

class Blackout_data(object):
    def __init__(self, blackout_max):
        self.blackout_max = blackout_max

    def __call__(self, sample):
        img, gt = sample['image'], sample['cardiac_view']
        image_size = np.min(np.shape(img))
        max_intensity = int(np.max(img))
        min_intensity = int(np.min(img))
        if not random.randint(0, 2):
            if self.blackout_max:
                edges = gaussian_filter(sobel(gt.astype('uint8')), 0.3)
                border = np.where(edges > 0)
                if len(border[0] != 0):
                    point = random.randint(0, len(border[0])-1)
                    rr, cc = circle(border[0][point], border[1][point], random.randint(1, self.blackout_max))
                    img[np.clip(rr, 0, image_size-1), np.clip(cc, 0, image_size-1)] = random.randint(min_intensity, max_intensity)
        return {'image':img, 'cardiac_view':gt}

class Gamma(object):
    """
    Gamma augmentation
    """
    def __init__(self, gamma_ratio, gain=1):
        self.gamma_ratio = gamma_ratio
        self.gain = gain 
    
    def __call__(self, sample):
        img, gt = sample['image'], sample['cardiac_view']
        if not random.randint(0, 2):
            g = random.uniform(1-self.gamma_ratio, 1+self.gamma_ratio)
            img = self.gain * img**g

        return {'image':img, 'cardiac_view':gt}

class Noise_injection(object):
    def __init__(self, var_max=0.05):
        self.var_max = var_max

    def __call__(self, sample):
        img, gt = sample['image'], sample['cardiac_view']
        if not random.randint(0, 2):
            h, w = np.shape(img)
            mean = 0
            sigma = random.uniform(0, self.var_max)**0.5
            gauss = np.random.normal(mean,sigma,(h,w))
            gauss = gauss.reshape(h,w)
            img = img + gauss
        return {'image':img, 'cardiac_view':gt}