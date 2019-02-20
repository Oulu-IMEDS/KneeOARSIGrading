#!/usr/bin/env bash

cd scripts
#python train.py --backbone_depth 18
#python train.py --backbone_depth 34
#python train.py --backbone_depth 50
#python train.py --backbone_depth 50 --se True
#python train.py --backbone_depth 50 --se True --dw True

python train.py --backbone_depth 18 --weighted_sampling True
python train.py --backbone_depth 34 --weighted_sampling True
python train.py --backbone_depth 50 --weighted_sampling True
python train.py --backbone_depth 50 --se True  --weighted_sampling True
python train.py --backbone_depth 50 --se True --dw True  --weighted_sampling True
