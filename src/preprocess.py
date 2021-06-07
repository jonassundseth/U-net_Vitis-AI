import numpy as np
import torch
import os
import cv2

def get_test_set(root_dir):
    '''
    Input: 
        root_dir: Path to root directory containing the Camus dataset
    Returns:
        Numpy array containing a path to every input file
    '''
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
            sequences.append(file)
                            
    return np.array(sequences)



class PreProcessor(object):
    '''
    Class object performing preprocessing
    Input:
        dimension: The target dimension of the quadratic output image
    Returns:
        Preprocessed image. Rescaled, zero padded and standardized
    '''
    def __init__(self, dimension):
            self.rescale = Rescale(dimension)
            self.zero_pad_and_center = Zero_pad_and_center()
            self.standardize = Standardize()

    def __call__(self, sequence):
        sequence = self.rescale(sequence)
        sequence = self.standardize(sequence)
        sequence = self.zero_pad_and_center(sequence)
        
        sequence = torch.from_numpy(sequence).float()

        return sequence


class Rescale(object):
    '''
    Class rescaling an image so that the largest dimension becomes the target dimension
    '''
    def __init__(self, out_size):
        self.out_size = out_size

    def __call__(self, sequence):
        h, w = sequence.shape
        
        if h > w:
            new_h, new_w = self.out_size, self.out_size * w / h
        else:
            new_h, new_w = self.out_size * h / w, self.out_size
          
        new_h_i = int(new_h)
        new_w_i = int(new_w)
        
        img = cv2.resize(sequence, (new_w_i, new_h_i))
        return img


class Zero_pad_and_center(object):
    """
    Padding the shortest dimension to fit the largest dimension
    """
    def __call__(self, sequence):
        h, w = sequence.shape
        if h > w:
            num_pads = h - w
            left, right = int(num_pads/2), int(num_pads/2)
            if num_pads % 2:
                left += 1
            img = np.pad(sequence, ([0, 0], [left, right]), mode='constant')
        else:
            num_pads = w - h
            up, down = int(num_pads/2), int(num_pads/2)
            if num_pads % 2:
                down += 1
            img = np.pad(sequence, ([up, down], [0, 0]), mode='constant')

        img = img[np.newaxis, ...]

        return img

class Standardize(object):
    '''
    Standardizing the image
    '''
    def __call__(self, sequence):
        mean = np.mean(sequence)
        std = np.std(sequence)
        return (sequence-mean)/std


