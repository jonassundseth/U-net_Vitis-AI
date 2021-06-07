# Acceleration of U-net using Vitis AI: Code base
Code base for Jonas' master thesis project. Running U-net on a Xilinx ZCU104

## Running the code provided in the codebase
### Training the model:
There are several arguments that can be parsed, use `--h` to display the options. To train the model with the default dataset and a dimension of 128x128 simply run:
```
python unet.py --dim 128 --train
```

### Quantizing the model:
Running the following steps of the pipeline requires use of the docker provided in [Vitis-AI](https://www.xilinx.com/products/design-tools/vitis/vitis-ai.html). 

```
./docker_run.sh xilinx/vitis-ai:latest
```
Thereafter simply run:
```
pip install -r requirements.txt
```

When the accuracy of the model is satisfactory the model can be quantized to int8 by running:
```
python models.py --input datasets/training/ --model trained_models/2021-03-25_03-40-20/weights_128x128.pth --quant_mode calib --dim 128
```
To test the accuracy of the quantized model simply run:
```
python models.py --input datasets/training/ --model trained_models/2021-03-25_03-40-20/weights_128x128.pth --quant_mode test --dim 128
```
To deploy simply add the deploy flag
```
python models.py --input datasets/training/ --model trained_models/2021-03-25_03-40-20/weights_128x128.pth --quant_mode test --dim 128 --deploy
```

### Compiling the model to run on an FPGA:
In the Vitis-AI docker compile the model from the previous step by running:
```
vai_c_xir -x quantize_result/unet_int.xmodel -a /opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json -o quantize_result/ -n unet_deploy
```

### Running the deployed model on FPGA:
Transfer the generated files to the FPGA using `scp` commands. Thereafter, simply run the quantized model on the FPGA using:
```
./run.sh
```
Where the output will be transferred back as .png files which can be qualitatively evaluated using the `postprocess.py`.