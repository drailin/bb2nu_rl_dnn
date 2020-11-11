# DNN with Richardson-Lucy for 2&nu;&beta;&beta; events

This repository contains code for training and evaluating a sparse implemenation of ResNet on Summit at ORNL. The expected input is Richardson-Lucy deconvolved 2&nu;&beta;&beta; and background events from the NEXT detector. 

## Instructions for running training and prediction on Summit at ORNL
Jobs are submitted through the job submission script `scn_hv.lsf` with the command
```
bsub scn_hv.lsf
```
To run the training code, change the line in `scn_hv.lsf` to execute `run_training.py`:
```
jsrun -n 24 -a 1 -c 2 -g 1   python -m  run_training.py
```
To run the evaluation code, change the line in `scn_hv.lsf` to execute `run_score_new_events.py`:
```
jsrun -n 1 -a 1 -c 2 -g 1   python -m  run_score_new_events.py
```
The `-n` flag specifies the number of GPUs to use. It should be equal to six times the number of compute nodes (which is specified above by `-nnodes`).

The input training and testing files are defined in `larcvconfig_train_lr.txt` and `larcvconfig_test_lr.txt`. All other parameters can be changed within the `dnn_larcv.py` script (learning rate, number of epochs, output file paths, etc.)
