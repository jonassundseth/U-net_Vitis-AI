#!/bin/sh
HOST="root@192.168.2.27"
DIM="128"

echo "Copy unet_fpga.py"
scp unet_fpga.py $HOST:~/unet/
echo "Run inference"
ssh $HOST "python3 unet/unet_fpga.py --input unet/datasets/testing/ --xmodel unet/unet_deploy_"$DIM".xmodel --dim "$DIM" --threads 2"
echo "Copy results back to host"
scp $HOST:~/unet/results/* "FPGA_output/"$DIM"x"$DIM"/"