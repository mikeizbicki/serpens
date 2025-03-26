#!/bin/sh

. venv/bin/activate

set -ex
commit=$(sh scripts/get_sane_commit_hash.sh)

# kill any running training scripts
#pids=$(ps -ef | grep 'python3 Mundus/train.py' | cut -b 10-17)
#if [ "$pids" != "" ]; then
    #echo "$pids" | xargs kill -9 || true
    #sleep 5
    # one of the pids returned will be the grep process;
    # kill will fail on this pid because the process has finished
    # by the time this command is run;
    # adding the || true ensures that the script doesn't terminate
#fi

# start the new scripts
common="--comment=$commit" 

CUDA_VISIBLE_DEVICES=0 nohup python3 Mundus/train.py $common --features_dim=32 --pooling=max --task_regex='attack' --reset_method='spiders enemy' --difference_dim=16 --objects_extractor=ObjectEmbeddingWithDiff > nohup/nohup.0 &
CUDA_VISIBLE_DEVICES=1 nohup python3 Mundus/train.py $common --features_dim=32 --pooling=max --task_regex='attack' --reset_method='spiders enemy' --difference_dim=32 --objects_extractor=ObjectEmbeddingWithDiff > nohup/nohup.1 &
CUDA_VISIBLE_DEVICES=2 nohup python3 Mundus/train.py $common --features_dim=32 --pooling=max --task_regex='attack' --reset_method='spiders enemy' --difference_dim=64 --objects_extractor=ObjectEmbeddingWithDiff > nohup/nohup.2 &
CUDA_VISIBLE_DEVICES=3 nohup python3 Mundus/train.py $common --features_dim=32 --pooling=max --task_regex='attack' --center_player --objects_extractor=ObjectEmbeddingWithDiff > nohup/nohup.3 &

sleep 1
tail -f nohup/nohup.?
