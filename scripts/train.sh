#!/bin/sh

mkdir -p nohup

#for gamma in 0.9 0.95 0.99; do
    #for lr in 3e-4 3e-5; do
        #nohup python3 Mundus/train.py --warmstart=log/net_arch\=\,lr\=0.0003\,gamma\=0.99\,n_env\=60\,n_steps\=128_2/model.zip --gamma=$gamma --lr=$lr --n_env=10 > nohup/nohup.out.$gamma.$lr &
    #done
#done

for i in $(seq 0 4); do
    CUDA_VISIBLE_DEVICES=$i nohup python3 Mundus/train.py --comment=new --n_env=128 --batch_size=256 --policy=ObjectCnn --features_dim=256 --gamma=0.9 --state=overworld*.state --scenario=attack > nohup/nohup.$i &
done
