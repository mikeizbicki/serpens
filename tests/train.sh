#!/bin/sh

set -e
set -x

temp_dir=$(mktemp -d)

TESTPARAMS="--total_timesteps=1000 --log_dir=$temp_dir --seed=0"

python3 Mundus/train.py $TESTPARAMS --n_env=1
python3 Mundus/train.py $TESTPARAMS --n_env=3
