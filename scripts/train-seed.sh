#!/bin/sh

set -ex
commit=$(sh scripts/get_sane_commit_hash.sh)

common="--comment=commit-$commit- --n_env=128 --batch_size=32 --features_dim=256 --gamma=0.9 --policy=ObjectCnn --action_space=zelda-all --task=attack --seed=0"

CUDA_VISIBLE_DEVICES=0 nohup python3 Mundus/train.py $common --n_env=16 --fwat=600 > nohup/nohup.0 &
CUDA_VISIBLE_DEVICES=1 nohup python3 Mundus/train.py $common --n_env=32 --fwat=600 > nohup/nohup.1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 Mundus/train.py $common --n_env=64 --fwat=600 > nohup/nohup.2 &
CUDA_VISIBLE_DEVICES=3 nohup python3 Mundus/train.py $common --n_env=16 --n_steps=64 --fwat=600 > nohup/nohup.0 &
CUDA_VISIBLE_DEVICES=4 nohup python3 Mundus/train.py $common --n_env=32 --n_steps=64 --fwat=600 > nohup/nohup.1 &
CUDA_VISIBLE_DEVICES=5 nohup python3 Mundus/train.py $common --n_env=64 --n_steps=64 --fwat=600 > nohup/nohup.2 &

tail -f nohup/nohup.?
