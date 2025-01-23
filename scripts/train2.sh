#!/bin/sh

mkdir -p nohup

common='--comment=working2 --n_env=128 --batch_size=256 --features_dim=256 --gamma=0.9 --state=overworld*.state --seed=0 --action_space=DISCRETE'

CUDA_VISIBLE_DEVICES=0 nohup python3 Mundus/train.py $common --policy=ObjectCnn --task=attack > nohup/nohup.0 &
CUDA_VISIBLE_DEVICES=1 nohup python3 Mundus/train.py $common --policy=EventExtractor --task=attack > nohup/nohup.1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 Mundus/train.py $common --policy=ObjectCnn --task=onmouse > nohup/nohup.2 &
CUDA_VISIBLE_DEVICES=3 nohup python3 Mundus/train.py $common --policy=ObjectCnn --task=onmouse2 > nohup/nohup.3 &
CUDA_VISIBLE_DEVICES=4 nohup python3 Mundus/train.py $common --policy=ObjectCnn --task=onmouse_enemy > nohup/nohup.4 &
