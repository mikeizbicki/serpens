#!/bin/sh

set -e

mkdir -p nohup

TASK="--n_env=128 --batch_size=256 --features_dim=256 --gamma=0.9 --state=overworld*.state --seed=0 --action_space=DISCRETE --policy=ObjectCnn --task=attack"

# this command gets the current branch if a branch is checked out;
# if in detached HEAD state, gets the current commit
ORIGINAL_GIT_STATE=$(git branch --show-current | grep -q . && git branch --show-current || git rev-parse HEAD)

MAX_COMMITS=5
i=0
for HASH in $(git log -n $MAX_COMMITS --format=%H); do
    echo "HASH=$HASH"
    git checkout $HASH --quiet
    SHORT_HASH=$(echo $HASH | cut -b1-8)
    CUDA_VISIBLE_DEVICES=$i nohup python3 Mundus/train.py --comment=hash3-$SHORT_HASH $TASK > nohup/nohup.$SHORT_HASH &
    i=$(($i + 1))
    sleep 30
done

git checkout $ORIGINAL_GIT_STATE --quiet
