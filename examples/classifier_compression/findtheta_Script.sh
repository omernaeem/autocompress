#!/bin/bash

time python3 compress_classifier.py -a resnet20_cifar ../../../data.cifar10 -p 30 -j=4 --lr=0.002 --out-dir logs/agp75/ --compress=schedules/final_quant.yaml --resume logs/actscheckpoint.pth.tar --epochs 50 --validation-split=0

echo 'DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE DONE  DONE  DONE  DONE  DONE  DONE  DONE  DONE '

python3 compress_classifier.py -a resnet20_cifar ../../../data.cifar10 -p 30 -j=4 --lr=0.002 --out-dir logs/agp75/ --compress=schedules/final_quant_acts.yaml --resume logs/checkpoint.pth.tar --epochs 50 --validation-split=0

