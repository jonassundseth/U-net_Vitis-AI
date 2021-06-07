# -*- coding: utf-8 -*-
import torch, torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader

from pathlib import Path
import datetime
import argparse
import sys
from matplotlib import pyplot as plt
from tqdm import tqdm

from unet_elem import DConv, Down, Up, OutConv

sys.path.append("./../")
import dataset

# Training algorithm is a modified of Anders Taskens codebase: https://github.com/Anderstask1/TEE_MAPSE/blob/master/dl_cardiac-view-classification/Code/train.py

class unet(nn.Module):
    def __init__(self, config=None):
        super(unet, self).__init__()
        self.config = config
        self.n_channels = 1
        self.n_classes = 4

        #Network architecture: (inspired by https://github.com/milesial/Pytorch-UNet/tree/6aa14cbbc445672d97190fec06d5568a0a004740)
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

    def train_model(self, dataloaders, loss, optimizer, num_epochs=25):
        '''
        Training algorithm
        '''
        device = torch.device(self.config.device)

        # Generate folder
        folder_path = _generate_folders(self.config.output)

        # Metadata names
        training_info_path = folder_path + "training_info.pth"
        weights_path = folder_path + "weights_"+ str(opt.dim) + "x" + str(opt.dim) +".pth"

        print("Training info path: ", training_info_path)
        print("Weights path: ", weights_path)

        criterion = nn.CrossEntropyLoss()

        train_info = {'epoch': [], 'loss': [], 'all_loss': []}
        val_info = {'epoch': [], 'loss': [], 'all_loss': []}
        best_loss = 1e10

        for epoch in range(num_epochs):

            print("Epoch {}/{}".format(epoch + 1, num_epochs))
            print("-" * 40)

            for phase in ['train', 'val']:

                print("Phase: ", phase)
                if phase == 'train':
                    self.train()
                else:
                    self.eval()

                running_loss = 0.0
                
                for _, sample_batch in tqdm(enumerate(dataloaders[phase]), 
                                            total=len(dataloaders[phase])):

                    sample_batch['image'] = sample_batch['image'].to(device)

                    optimizer.zero_grad()

                    with torch.set_grad_enabled(phase == 'train'):
                        out = self(sample_batch['image'])
                        target = torch.LongTensor(sample_batch['cardiac_view'].long()).to(device)
                        loss = criterion(out, target)

                        all_loss = loss.item()
                        running_loss += all_loss

                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                            train_info['all_loss'].append(all_loss)

                epoch_loss = running_loss / len(dataloaders[phase].dataset)

                if phase == 'train':
                    print('{{"metric": "loss", "value": {}, "epoch": {}}}'.format(
                        epoch_loss, epoch + 1))
                else:
                    print('{{"metric": "Validation loss", "value": {}, "epoch": {}}}'.format(
                        epoch_loss, epoch + 1))
                
                if phase == 'train':
                    train_info['epoch'].append(epoch + 1)
                    train_info['loss'].append(epoch_loss)
                else:
                    val_info['epoch'].append(epoch + 1)
                    val_info['loss'].append(epoch_loss)

                torch.save({
                    'epoch': epoch,
                    'train_info': train_info,
                    'val_info': val_info}, training_info_path)

                if phase == 'val' and epoch_loss <= best_loss:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict()}, weights_path)
                    best_loss = epoch_loss
                    print("Weights saved")
        print()

        
def _options():
    """Function for taking in arguments from user
    Returns:
        Arguments from user
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", 
                        required=False, 
                        action="store_true", 
                        help="Train model")
    parser.add_argument('--device', 
                        type=str, 
                        default="cpu", 
                        help='Which device to run on')
    parser.add_argument("--input", 
                        type=str, 
                        required=False,
                        default="datasets/training/", 
                        help="Path to dataset")
    parser.add_argument("--output", 
                        type=str, 
                        required=False, 
                        default="trained_models/",
                        help="Path to output")
    parser.add_argument("--dim", 
                        type=int, 
                        required=False, 
                        default=256, 
                        help="Dimension to be used in training, dim x dim image")
    parser.add_argument("--batch_size", 
                        type=int, 
                        required=False, 
                        default=5, 
                        help="Batch size used during training phase")
    parser.add_argument("--epochs", 
                        type=int, 
                        required=False, 
                        default=25, 
                        help="Number of training epochs")

    return parser.parse_args()   



def _generate_folders(start_path):
    """Generate folder where dataset weill be stored
    Returns:
        Directory to where data will be saved
    """

    location_dir = Path(start_path + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    if not location_dir.is_dir():
        location_dir.mkdir(parents=True)

    return str(location_dir) + "/"

if __name__ == "__main__":
    opt = _options()
    
    opt.input = opt.input  if ("/" == opt.input[-1]) else opt.input + "/"
    opt.output = opt.output  if ("/" == opt.output[-1]) else opt.output + "/"

    if opt.train:
        
        data_transforms =   {
                            'train':transforms.Compose([dataset.Rescale(opt.dim),
                                                        dataset.Standardize(),
                                                        dataset.Zero_pad_and_center(),
                                                        dataset.ToTensor()
                                                    ]),
                            'val':transforms.Compose([  dataset.Rescale(opt.dim),
                                                        dataset.Standardize(),
                                                        dataset.Zero_pad_and_center(),
                                                        dataset.ToTensor()
                                                    ]),
                            'aug':transforms.Compose([  dataset.RandomCrop(crop_ratio=0.1),
                                                        dataset.RandomRotation(degrees=15),
                                                        dataset.Blackout_data(100), 
                                                        dataset.Gamma(gamma_ratio=0.35),
                                                        dataset.Noise_injection()
                                                    ])
                            }

        datasets =      {
                        'train':dataset.UltrasoundData(opt.input, transform=data_transforms['train'], augment_transform=data_transforms['aug']),
                        'val':dataset.UltrasoundData(opt.input, transform=data_transforms['val'], val=True)
                        }
        
        print("Number of training samples: " + str(len(datasets['train'])))
        print("Number of validation samples: " + str(len(datasets['val'])))

        dataloaders =   {
                        'train':DataLoader(datasets['train'], batch_size=opt.batch_size, shuffle=True, num_workers=4),
                        'val':DataLoader(datasets['val'], batch_size=1, shuffle=False, num_workers=4)
                        }

        print()
        print(" + Batch size:\t\t\t{}".format(opt.batch_size))
        print(" + Number of epochs:\t\t{}".format(opt.epochs))
        print()

        print(datasets['train'])
        model = unet(opt)
        print("Model architecture: U-Net")

        model = model.to(opt.device)

        optimizer = optim.Adam(model.parameters())	

        loss = nn.L1Loss()

        print("Training model...")
        
        model.train_model(
                    dataloaders=dataloaders, 
                    loss=loss, 
                    optimizer=optimizer, 
                    num_epochs=opt.epochs
                    )
    