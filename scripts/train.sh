#!/bin/sh

set -ex
commit=$(sh scripts/get_sane_commit_hash.sh)

common="--comment=$commit --task=attack"

CUDA_VISIBLE_DEVICES=0 nohup python3 Mundus/train.py $common --reset_method="spiders enemy" > nohup/nohup.0 &
CUDA_VISIBLE_DEVICES=1 nohup python3 Mundus/train.py $common --reset_method="octo enemy" > nohup/nohup.1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 Mundus/train.py $common --reset_method="map enemy" > nohup/nohup.2 &
CUDA_VISIBLE_DEVICES=3 nohup python3 Mundus/train.py $common --reset_method="spiders octo enemy" > nohup/nohup.3 &

tail -f nohup/nohup.?
