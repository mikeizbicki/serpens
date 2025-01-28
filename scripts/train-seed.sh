#!/bin/sh

set -ex
commit=$(sh scripts/get_sane_commit_hash.sh)

common="--comment=commit-$commit --n_env=128 --batch_size=256 --features_dim=256 --gamma=0.9 --policy=ObjectCnn --action_space=zelda-all --task=attack --seed=0"

CUDA_VISIBLE_DEVICES=0 nohup python3 Mundus/train.py $common --reset_method="spiders enemy" > nohup/nohup.0 &
CUDA_VISIBLE_DEVICES=1 nohup python3 Mundus/train.py $common --reset_method="octo enemy" > nohup/nohup.1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 Mundus/train.py $common --reset_method="spiders" > nohup/nohup.2 &
CUDA_VISIBLE_DEVICES=3 nohup python3 Mundus/train.py $common --reset_method="octo" > nohup/nohup.3 &

tail -f nohup/nohup.?
