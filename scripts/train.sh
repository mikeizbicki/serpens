#!/bin/sh

set -ex
commit=$(sh scripts/get_sane_commit_hash.sh)

common="--comment=$commit --task=attack"

CUDA_VISIBLE_DEVICES=0 nohup python3 Mundus/train.py $common --warmstart=1cdbbae7 > nohup/nohup.0 &
CUDA_VISIBLE_DEVICES=1 nohup python3 Mundus/train.py $common --warmstart=a7bf5c48 > nohup/nohup.1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 Mundus/train.py $common --warmstart=f99d1871 > nohup/nohup.2 &
CUDA_VISIBLE_DEVICES=3 nohup python3 Mundus/train.py $common --warmstart=51cb494b > nohup/nohup.3 &

tail -f nohup/nohup.?
