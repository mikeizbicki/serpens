#!/bin/sh

set -ex
commit=$(sh scripts/get_sane_commit_hash.sh)

common="--comment=gamma-$commit- --n_env=128 --batch_size=256 --features_dim=256 --gamma=0.9 --policy=EventExtractor --action_space=zelda-all --task=attack"

CUDA_VISIBLE_DEVICES=0 nohup python3 Mundus/train.py $common --seed=0 --fwat=300 > nohup/nohup.0 &
CUDA_VISIBLE_DEVICES=1 nohup python3 Mundus/train.py $common --seed=1 --fwat=300 > nohup/nohup.1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 Mundus/train.py $common --seed=0 --fwat=600 > nohup/nohup.2 &
CUDA_VISIBLE_DEVICES=3 nohup python3 Mundus/train.py $common --seed=1 --fwat=600 > nohup/nohup.3 &
CUDA_VISIBLE_DEVICES=4 nohup python3 Mundus/train.py $common --seed=0 > nohup/nohup.4 &

