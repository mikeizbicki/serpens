#!/bin/sh

set -e
set -x

temp_dir=$(mktemp -d)

TESTPARAMS="--total_timesteps=1000 --log_dir=$temp_dir --seed=0 --n_env=1 --batch_size=8"

python3 Mundus/train.py $TESTPARAMS

warmstart=$(echo ${temp_dir}/*/model.zip)

python3 Mundus/train.py $TESTPARAMS --warmstart="$warmstart"
