# ESPCN-TensorFlow2
ESPCN(https://arxiv.org/abs/1609.05158) implementation using TensorFlow 2.0


## Requirements
* tensorflow >= 2.0.0
* tensorflow_datasets

## Note
* The phase-shifting code is from https://github.com/kweisamx/TensorFlow-ESPCN.
* The COCO dataset by tensorflow_datsets takes up really much space! (About 77G in my case...)
* TPU code is still in development.
* This code was written for studying, so the code may be hard to understand... I'll try my best to improve code readability.

## Usage
### Train
```
python3 train.py -lr LEARNING_RATE -exp_name EXP_NAME -batch_size BATCH_SIZE [-save_dir SAVE_DIR]
```
Learning rate about 0.001 is recommended.
### Convert to TFLite
```
python3 tflite.py -exp_name EXP_NAME -model_epoch EPOCH [-saved_dir SAVED_DIR] [-tflite_dir TFLITE_DIR]
```