from ctypes import *
from typing import List
from medimage import image
# Avoiding using the same function name
from PIL import Image as pilimg 
import cv2
import numpy as np
import xir
import vart
import os
import time
import sys
import argparse
import concurrent.futures

global threadnum
threadnum = 0

# Parts of the code is based on: https://github.com/Xilinx/Vitis-AI/blob/master/demo/Whole-App-Acceleration/resnet50_mt_py_waa/resnet50.py

# Function used to supress medimage.image()
class NoStdStreams(object):
    def __init__(self,stdout = None, stderr = None):
        self.devnull = open(os.devnull,'w')
        self._stdout = stdout or self.devnull or sys.stdout
        self._stderr = stderr or self.devnull or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush(); self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush(); self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr
        self.devnull.close()


def get_test_set(root_dir):
    '''
    Input: 
        root_dir: Path to root directory containing the Camus dataset
    Returns:
        Numpy array containing a path to every input file
        List containing corresponding patient id
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
    patient_id = []
    for file in files:
        if (".mhd" in file) and ("2CH" in file) and ("gt" not in file) and ("sequence" not in file):
            sequences.append(file)
            patient_id.append(file.split("/")[-1][:-4])
                
    return np.array(sequences), patient_id


def rescale(img, dim):
    '''
    Function rescaling an image so that the largest dimension becomes the target dimension
    Input:
        img: image to be rescaled
        dim: target dimension
    Returns
        Rescaled image in a numpy array
    '''
    h, w = np.shape(img)
 
    if h > w:
        new_h, new_w = dim, dim * w / h
    else:
        new_h, new_w = dim * h / w, dim
    
    new_h_i = int(new_h)
    new_w_i = int(new_w)

    #new_img = np.empty([new_h_i, new_w_i])
    new_img = cv2.resize(img, (new_w_i, new_h_i))
    
    return new_img


def zero_pad_and_center(img):
    '''
    Function zero padding an image so that the image becomes quadratic
    Input:
        img: image to be zero padded
    Returns
        Quadratic zero padded image
    '''
    h, w = np.shape(img)
    if h > w:
        num_pads = h - w
        left, right = int(num_pads/2), int(num_pads/2)
        if num_pads % 2:
            left += 1
        img = np.pad(img, ([0, 0], [left, right]), mode='constant')
    else:
        num_pads = w - h
        up, down = int(num_pads/2), int(num_pads/2)
        if num_pads % 2:
            down += 1
        img = np.pad(img, ([up, down], [0, 0]), mode='constant')

    return img


def standardize(img):
    '''
    Function standardizing the input image
    Input:
        img: image to be standardized
    Returns
        Standardized image in numpy array
    '''
    mean = np.mean(img)
    std = np.std(img)
    img = np.true_divide((img-mean), std)
    return img


def preprocess_one_image_fn(image_path, dim, save=False):
    '''
    Function preprocessing one image frame
    Input:
        image_path: path to image 
        dim: target dimension for image
        save: save each step of preprocessed frame for debugging purposes. default=False
    Returns
        Preprocessed image of size [dimxdim]
    '''
    imgs_to_save = []
    with NoStdStreams():
        imgs_to_save.append(image(image_path).imdata.T[0])      # Stage 0
    imgs_to_save.append(rescale(imgs_to_save[-1], dim))         # Stage 1
    imgs_to_save.append(standardize(imgs_to_save[-1]))          # Stage 2
    imgs_to_save.append(zero_pad_and_center(imgs_to_save[-1]))  # Stage 3

    if save:
        for i, img in enumerate(imgs_to_save):
            im = pilimg.fromarray(img)
            im = im.convert("L")
            im.save('unet/results/preprocessing_stage_' + str(i) + '.png')
    
    return imgs_to_save[-1]


def runUnet(runner: "Runner", img, cnt):
    '''
    Function running inference in a thread
    Input:
        img: image to be inferred
        cnt: number of images to infer on
    Returns
        numpy array containing 4 bitmasks [4xdimxdim]
    '''
    # Get tensors
    inputTensors = runner.get_input_tensors()
    outputTensors = runner.get_output_tensors()
    input_ndim = tuple(inputTensors[0].dims)

    output_ndim = tuple(outputTensors[0].dims)
    n_of_images = len(img)
    count = 0
    
    results = []
    time_list = []
    while count < cnt:
        runSize = input_ndim[0]
        # Allocate 
        inputData = [np.empty(input_ndim, dtype=np.float32, order="C")]
        outputData = [np.empty(output_ndim, dtype=np.float32, order="C")]

        # Fill buffer
        for j in range(runSize):
            imageRun = inputData[0]
            #imageRun[j, ...] = img[(count + j) % n_of_images].reshape(input_ndim[1:])
            imageRun[j, ...] = img[(count + j) % n_of_images].resize(input_ndim[1:])
        start = time.time()
        job_id = runner.execute_async(inputData, outputData)
        runner.wait(job_id)
        time_list.append(time.time()-start)
        results.append(outputData[0][0])
        count = count + runSize
    results = np.array(results)
    
    return results, time_list
    

def get_child_subgraph_dpu(graph: "Graph") -> List["Subgraph"]:
    assert graph is not None, "'graph' should not be None."
    root_subgraph = graph.get_root_subgraph()
    assert (
        root_subgraph is not None
    ), "Failed to get root subgraph of input Graph object."
    if root_subgraph.is_leaf:
        return []
    child_subgraphs = root_subgraph.toposort_child_subgraph()
    assert child_subgraphs is not None and len(child_subgraphs) > 0
    return [
        cs
        for cs in child_subgraphs
        if cs.has_attr("device") and cs.get_attr("device").upper() == "DPU"
    ]


def _options():
    """Function for taking in arguments from user
    Returns:
        Arguments from user
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", 
                        type=str,
                        default="unet/datasets/testing/",
                        help="Data set directory.")
    parser.add_argument("--xmodel", 
                        type=str, 
                        required=True, 
                        help="Trained model file path. .xmodel-file")
    parser.add_argument("--dim", 
                        type=int, 
                        required=False, 
                        default=256, 
                        help="Dimension to be used in preprocessing, dim x dim image. Default = 256")
    parser.add_argument("--threads", 
                        type=int, 
                        required=False, 
                        default=4, 
                        help="Number of threads. Default = 4")
    parser.add_argument("--save", 
                        action="store_true",
                        required=False, 
                        help="Save preprocessing stages as .png-files")

    return parser.parse_args() 


def mergemasks(imgs):
    '''
    Function merging bitmasks
    Input:
        imgs: array containing a number of bitmasks [masksxdimxdim]
    Returns
        One bitmask merged based on highest probability [dimxdim]
    '''
    return np.argmax(imgs, axis=2)


def main(opt):
    global threadnum
    listimage, patient_id = get_test_set(opt.input)
    threadAll = []
    threadnum = opt.threads
    i = 0
    global runTotall
    runTotall = len(listimage)
    g = xir.Graph.deserialize(opt.xmodel)
    subgraphs = get_child_subgraph_dpu(g)
    assert len(subgraphs) == 1  # only one DPU kernel
    all_dpu_runners = []
    for i in range(int(threadnum)):
        all_dpu_runners.append(vart.Runner.create_runner(subgraphs[0], "run"))
    
    img = []
    first = opt.save
    print("Loading data")
    for path in listimage:
        img.append(preprocess_one_image_fn(image_path=path, dim=opt.dim, save=first))
        first = False
    img = np.array(img)  
    

    """
      The cnt variable is used to control the number of times a single-thread DPU runs.
      Users can modify the value according to actual needs. It is not recommended to use
      too small number when there are few input images, for example:
      1. If users can only provide very few images, e.g. only 1 image, they should set
         a relatively large number such as 360 to measure the average performance;
      2. If users provide a huge dataset, e.g. 50000 images in the directory, they can
         use the variable to control the test time, and no need to run the whole dataset.
    """

    # High count might fill up RAM and should be changed depending on the input image size.
    assert opt.threads > 0
    cnt = 100//opt.threads 
    num_imgs = len(listimage)
    
    # Filling up buffers so that all images are inferred either in each or different threads.
    if cnt < num_imgs:
        #Split
        batch = []
        i = 0
        for _ in range(threadnum):
            if not (i+1)*cnt > num_imgs:
                # Normal batching
                batch.append(img[cnt*i:cnt*(i+1)])
                i += 1
            else:
                # Overflow
                overflow_batch = np.concatenate((img[cnt*i:], img[:(cnt*(i+1))%num_imgs]), axis=0)
                batch.append(overflow_batch)
                i = 0
    else:
        # Duplicate whole testing set for each thread
        batch = np.repeat(img[np.newaxis, :, :], threadnum, axis=0)
    
    print("Starting inference")
    time_start = time.time()
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for i in range(threadnum):
            t1 = executor.submit(runUnet, all_dpu_runners[i], batch[i], cnt)
            threadAll.append(t1)
        for i, t in enumerate(threadAll):
            if not i:
                # First thread
                results, time_list = t.result()
            else:
                if len(results) < num_imgs:
                    thread_results, thread_time = t.result()
                    results = np.concatenate((results, thread_results), axis=0)
                    time_list = np.concatenate((time_list, thread_time), axis=0)
                else:
                    #First thread obtained all the results.
                    _, _ = t.result()
            
    results = results[:num_imgs]                
    time_end = time.time()
    del all_dpu_runners
    timetotal = time_end - time_start
    latency = np.mean(time_list)
    results = np.array(results)
    total_frames = cnt * threadnum
    fps = float(total_frames / timetotal)
    
    with open("unet/results/log.txt", "w") as file:
        file.write("Res=%d, threads=%d\nFPS=%.2f, total frames = %.2f , time=%.6f seconds, latency=%.6f seconds"
        % (opt.dim, opt.threads, fps, total_frames, timetotal, latency))
    
    print("Inference done")
    
    # Saving output to .png files
    for idx, pid in enumerate(patient_id):
        if idx < cnt:
            output = np.zeros(np.shape(results)[1:3])
            output = mergemasks(results[idx])
            im = pilimg.fromarray((output*64).astype(np.uint8))
            im = im.convert("L")
            im.save('unet/results/output_mask_' + pid + '.png')
    

if __name__ == "__main__":
    opt = _options()
    main(opt)