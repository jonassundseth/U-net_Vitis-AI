import sys
import time
import numpy as np
import SimpleITK as sitk
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from medimage import image

sys.path.append("./src/models/")
sys.path.append("./src/")
from unet import unet
from preprocess import PreProcessor, get_test_set
from supress import NoStdStreams
import postprocess as pp

def mergemasks(imgs):
    '''
    Function merging bitmasks
    Input:
        imgs: array containing a number of bitmasks
    Returns
        One bitmask merged based on highest probability [dimxdim]
    '''
    return np.argmax(imgs, axis=0) 

def main(options):
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(dev)

    file_dir = opt.input
    model_path = options.model
    model = unet()
    
    # Load model parameters from state dictionary
    model.load_state_dict(torch.load(model_path, map_location=options.device)['model_state_dict'])

    # Set processing unit
    model = model.to(device)

    # Turn off layers specific for training, since evaluating here
    model.eval()
    torch.set_grad_enabled(False)

    # initialize the pipeline
    preprocessor = PreProcessor(dimension=opt.dim)

    data_input = get_test_set(file_dir)
    input_stream = []
    ground_truth = []

    #Load and preprocess images before inference.
    print("Loading dataset")
    for file in data_input:
        with NoStdStreams(): #Supress medimage
            sequence = image(file).imdata.T[0]
        gt_file = file[:-4] + "_gt.mhd"
        ground_truth.append(np.array(sitk.GetArrayFromImage(sitk.ReadImage(gt_file, sitk.sitkFloat32)))[0])
        
        sequence = preprocessor(sequence)

        input_stream.append(sequence)
    
    # Inference on preprocessed images in a sequential manner to benchmark.
    time_start = time.time()
    output = []
    print("Infering data on CPU")
    time_list = []
    for i, img in enumerate(input_stream):
        frame = 0
        start = time.time()
        model_input = img[frame, :, :]
        model_input = model_input.unsqueeze(0).unsqueeze(0).to(device)

        #run the pipeline
        model_output = model(model_input)
        model_output = F.softmax(model_output, dim=1).to("cpu")
        model_output = model_output[0,:].numpy()
        time_list.append(time.time()-start)
        output.append(mergemasks(model_output))
    print("Finished infering")
    time_end = time.time()

    time_total = time_end - time_start
    total_frames = len(input_stream)
    fps = float(total_frames/time_total)
    
    print(
        "FPS=%.2f, total frames = %.2f , time=%.6f seconds, latency=%.6f"
        % (fps, total_frames, time_total, np.mean(time_list))
    )
    avg_dice = []
    for i, mask in enumerate(output):
        # Upsample output to match ground truth
        h, w = np.shape(ground_truth[i])
        upsampled_image = pp.upsample(mask, h, w)
        dice = pp.dice_coefficient(mask=upsampled_image, target=ground_truth[i])
        avg_dice.append(dice)
        #Plotting the segmentation for each class along with input image.
        if opt.plot:
            print("Dice coefficient = " + str(dice))
            plt.gray()
            plt.subplots_adjust(0,0,1,1,0.01,0.01)
            plt.subplot(2,2,1) 
            plt.imshow(img[0])
            plt.subplot(2,2,2) 
            plt.imshow(mask)
            plt.subplot(2,2,3) 
            plt.imshow(ground_truth[i])
            plt.subplot(2,2,4) 
            plt.imshow(upsampled_image)
            plt.show(block=False)
            plt.pause(2)
            plt.close()
            
    print("Average dice score: " + str(np.mean(avg_dice)))


def _options():
    """Function for taking in arguments from user
    Returns:
        Arguments from user
    """
    parser = argparse.ArgumentParser(description=__doc__)
    
    parser.add_argument('--device', 
                        type=str, 
                        default="cpu", 
                        help='Which device to run on')
    parser.add_argument("--input", 
                        type=str, 
                        required=False,
                        default="src/models/datasets/testing/", 
                        help="Path to dataset")
    parser.add_argument("--model", 
                        type=str, 
                        required=True,
                        help="Path to model")
    parser.add_argument("--dim", 
                        type=int, 
                        required=False, 
                        default=256, 
                        help="Dimension to be used in input data, dim x dim image")
    parser.add_argument("--plot", 
                        action="store_true", 
                        help="Plot output")

    return parser.parse_args() 

if __name__ == "__main__":
    opt = _options()
    opt.input = opt.input  if ("/" == opt.input[-1]) else opt.input + "/"
    main(opt)