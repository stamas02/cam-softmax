#This is the core repo for CAM-Softmax. 

## Install
`pip install -r requirements.txt`

## Usage

### Align dataset
Before use, both the training and the test dataset must be aligned using the following code:

`python align_dataset.py --dataset [DATASET DIR] --output [OUTPUT DIR]`

### Train

`python train.py --dataset [DATASET DIR]`

For info on all the parameters run:

`python train.py --help`

## Evaluate

`python evaluate.py --dataset [DATASET DIR] --pair_file [LFW_PAIR_FILE], --model_file
log/model.chkp`

Note that you have to change the path to the model file if you choose to save the trained model elsewhere. 
