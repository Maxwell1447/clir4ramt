#!/bin/sh

mkdir -p configs
source /nfs/RESEARCH/bouthors/anaconda3/bin/activate clir

python -m clir.train.trainer fit --model=clir.train.trainee.BiEncoder --data=clir.train.data.DataModuleMMap --print_config > configs/default.yaml
