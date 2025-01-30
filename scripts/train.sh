#!/bin/sh

set -ex
commit=$(sh scripts/get_sane_commit_hash.sh)

common="--comment=$commit --task=attack "

CUDA_VISIBLE_DEVICES=0 nohup python3 Mundus/train.py $common --reset_method='spiders octo enemy' --n_env=128 --batch_size=256 > nohup/nohup.0 &
CUDA_VISIBLE_DEVICES=1 nohup python3 Mundus/train.py $common --reset_method='spiders octo enemy' --n_env=128 --batch_size=512 > nohup/nohup.1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 Mundus/train.py $common --reset_method='spiders octo enemy' --n_env=128 --batch_size=1024 > nohup/nohup.2 &
#CUDA_VISIBLE_DEVICES=3 nohup python3 Mundus/train.py $common --reset_method='spiders octo enemy' --n_env=32 --batch_size=32 > nohup/nohup.3 &

tail -f nohup/nohup.?
