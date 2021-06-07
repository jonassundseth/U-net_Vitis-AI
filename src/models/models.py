# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import Subset

import sys
import random
import argparse
from tqdm import tqdm

from unet_elem import DConv, Down, Up, OutConv

from pytorch_nndct.apis import torch_quantizer

sys.path.append("./../")
import dataset

# Based on https://github.com/Xilinx/Vitis-AI/blob/master/tools/Vitis-AI-Quantizer/vai_q_pytorch/example/resnet18_quant.py

# Identical to network in unet.py, but without class functions.
class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.n_channels = 1
        self.n_classes = 4

        self.inc = DConv(self.n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1) 
        logits = self.outc(x)
        return logits 

def load_data(
    opt, 
    sample_method='random', 
    subset_len=None, 
    batch_size=5
):
    '''
    Function that composes dataloader
    '''
    data_transforms =   transforms.Compose([
                                dataset.Rescale(opt.dim),
                                dataset.Standardize(),
                                dataset.Zero_pad_and_center(),
                                dataset.ToTensor()
                                ])
    
    datasets        =   dataset.UltrasoundData( 
                                root_dir=opt.input, 
                                transform=data_transforms,
                                )
    
    if subset_len:
        assert subset_len <= len(datasets)
        if sample_method == 'random':
            datasets = Subset(
                            datasets, 
                            random.sample(range(0, len(datasets)), subset_len)
                            )
        else:
            datasets = Subset(
                            datasets, 
                            list(range(subset_len))
                            )

    dataloader = DataLoader(datasets, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=4
                            )

    return dataloader

def evaluate(
    model, 
    val_loader, 
    device, 
    loss_fn=nn.CrossEntropyLoss()
):
    '''
    Evaluation function
    '''
    model.eval()
    model = model.to(device)
    
    Loss = 0
    for _, sample_batch in tqdm(enumerate(val_loader), 
                                total=len(val_loader)):
        sample_batch['image'] = sample_batch['image'].to(device)
        
        outputs = model(sample_batch['image'])
        target = torch.LongTensor(sample_batch['cardiac_view'].long()).to(device)
        loss = loss_fn(outputs, target)
        Loss += loss.item()

    return Loss / len(val_loader.dataset)


def quantization(opt): 
    quant_mode      = opt.quant_mode
    finetune        = opt.fast_finetune
    deploy          = opt.deploy
    batch_size      = opt.batch_size
    subset_len      = opt.subset
    num_channels    = 1
    device          = torch.device(opt.device)
    '''
    Function to quantize weights
    '''
    if quant_mode != 'test' and deploy:
        deploy = False
        print(r'Warning: Exporting xmodel needs to be done in quantization test mode, turn off it in this running!')
    if deploy and (batch_size != 1 or subset_len != 1):
        print(r'Warning: Exporting xmodel needs batch size to be 1 and only 1 iteration of inference, change them automatically!')
        batch_size = 1
        subset_len = 1

    model = unet().cpu()
    model.load_state_dict(torch.load(opt.model)['model_state_dict'])

    placeholder_input = torch.randn([batch_size, num_channels, opt.dim, opt.dim])
    
    if quant_mode == 'float':
        quant_model = model
    else:
        quantizer = torch_quantizer(
                            quant_mode, 
                            model, 
                            (placeholder_input), 
                            bitwidth=opt.bit_width,
                            device=device,
                            qat_proc=opt.qat
                            )

        quant_model = quantizer.quant_model
        

    # to get loss value after evaluation
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    val_loader = load_data(
                        opt=opt,
                        subset_len=subset_len,
                        batch_size=batch_size,
                        sample_method='random',
                        )

    # fast finetune model or load finetuned parameter before test
    if finetune == True:
        ft_loader = load_data(
                            opt = opt,
                            batch_size=batch_size,
                            sample_method=None
                            )
        if quant_mode == 'calib':
            quantizer.fast_finetune(evaluate, (quant_model, ft_loader, loss_fn))
        elif quant_mode == 'test':
            quantizer.load_ft_param()

    loss_gen = evaluate(quant_model, val_loader, opt.device, loss_fn)

    # logging accuracy
    print('loss: %g' % (loss_gen))

    # handle quantization result
    if quant_mode == 'calib':
        quantizer.export_quant_config()
    if deploy:
        quantizer.export_xmodel(deploy_check=False)

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
                        default="datasets/training/",
                        help="Data set directory, when quant_mode=calib, it is for calibration, while quant_mode=test it is for evaluation")
    parser.add_argument("--model", 
                        type=str, 
                        required=True, 
                        help="Trained model file path.")
    parser.add_argument("--subset", 
                        type=int,
                        help="subset_len to evaluate model, using the whole validation dataset if it is not set")
    parser.add_argument("--batch_size", 
                        type=int, 
                        default=5, 
                        help="input data batch size to evaluate model")
    parser.add_argument('--quant_mode', 
                        default='calib',
                        choices=['float', 'calib', 'test'], 
                        help='quantization mode. 0: no quantization, evaluate float model, calib: quantize, test: evaluate quantized model')
    parser.add_argument('--fast_finetune', 
                        dest='fast_finetune', 
                        action='store_true', 
                        help='fast finetune model before calibration')
    parser.add_argument('--deploy', 
                        dest='deploy', 
                        action='store_true', 
                        help='export xmodel for deployment')
    parser.add_argument("--dim", 
                        type=int, 
                        required=False, 
                        default=256, 
                        help="Dimension to be used in training, dim x dim image")
    parser.add_argument("--bit_width", 
                        type=int, 
                        required=False, 
                        default=8, 
                        help="Global quantization bit width. Default=8")
    parser.add_argument('--qat',
                        action='store_true', 
                        help='Turn on quantize finetuing, also named quantization-aware-training (QAT)')

    return parser.parse_args()  

def main(opt):
    opt.input = opt.input  if ("/" == opt.input[-1]) else opt.input + "/"
    quantization(opt)

if __name__ == "__main__":
    opt = _options()
    main(opt)
