#!/bin/zsh

#$-l rt_G.small=1
#$ -l h_rt=24:00:00
#$-cwd
#$-j y

source /etc/profile.d/modules.sh
module load cuda/10.0/10.0.130 cudnn/7.4/7.4.2

cd ~/Documents/tensorflow20/

python3 autoencoder.py capsule
