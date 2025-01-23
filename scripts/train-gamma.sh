#!/bin/sh

mkdir -p nohup

common='--comment=gamma5 --n_env=128 --batch_size=256 --features_dim=256 --state=overworld*.state --seed=0 --policy=ObjectCnn --task=attack --action_space=zelda-all'

CUDA_VISIBLE_DEVICES=0 nohup python3 Mundus/train.py $common --gamma=0.99 > nohup/nohup.0 &
CUDA_VISIBLE_DEVICES=1 nohup python3 Mundus/train.py $common --gamma=0.95 > nohup/nohup.1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 Mundus/train.py $common --gamma=0.90 > nohup/nohup.2 &
CUDA_VISIBLE_DEVICES=3 nohup python3 Mundus/train.py $common --gamma=0.85 > nohup/nohup.3 &
CUDA_VISIBLE_DEVICES=4 nohup python3 Mundus/train.py $common --gamma=0.80 > nohup/nohup.4 &
