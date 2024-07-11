#!/bin/sh

mkdir -p nohup
MAX_DEVICES=8
COMMENT=hyper3

iter_num=0
for n_steps in 32 64; do
    for batch_size in 128 256 512 1024; do
        cvd=$(($iter_num % $MAX_DEVICES))
        iter_num=$(($iter_num+1))
        name="n_steps=$n_steps;batch_size=$batch_size"
        echo $name
        CUDA_VISIBLE_DEVICES=$cvd nohup python3 Mundus/train.py --comment=$COMMENT --n_env=128 --policy=ObjectCnn --features_dim=256 --total_timesteps=1000000 --gamma=0.9 --scenario=attack --n_steps=$n_steps --batch_size=$batch_size > "nohup/$COMMENT.$name" &
    done
done

#CUDA_VISIBLE_DEVICES=1 nohup python3 Mundus/train.py --comment=new --n_env=8   --policy=ObjectCnn --features_dim=256 --gamma=0.9 --state=overworld*.state --scenario=attack > nohup/nohup.1 &
#CUDA_VISIBLE_DEVICES=2 nohup python3 Mundus/train.py --comment=new --n_env=16  --policy=ObjectCnn --features_dim=256 --gamma=0.9 --state=overworld*.state --scenario=attack > nohup/nohup.2 &
#CUDA_VISIBLE_DEVICES=3 nohup python3 Mundus/train.py --comment=new --n_env=32  --policy=ObjectCnn --features_dim=256 --gamma=0.9 --state=overworld*.state --scenario=attack > nohup/nohup.3 &
#CUDA_VISIBLE_DEVICES=4 nohup python3 Mundus/train.py --comment=new --n_env=64  --policy=ObjectCnn --features_dim=256 --gamma=0.9 --state=overworld*.state --scenario=attack > nohup/nohup.4 &
#CUDA_VISIBLE_DEVICES=5 nohup python3 Mundus/train.py --comment=new --n_env=128 --policy=ObjectCnn --features_dim=256 --gamma=0.9 --state=overworld*.state --scenario=attack > nohup/nohup.5 &
#CUDA_VISIBLE_DEVICES=6 nohup python3 Mundus/train.py --comment=new --n_env=64  --policy=ObjectCnn --features_dim=256 --gamma=0.9 --scenario=attack > nohup/nohup.6 &
#CUDA_VISIBLE_DEVICES=7 nohup python3 Mundus/train.py --comment=new --n_env=128 --policy=ObjectCnn --features_dim=256 --gamma=0.9 --scenario=attack > nohup/nohup.7 &

