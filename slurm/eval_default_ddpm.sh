#!/bin/bash
#SBATCH --gpus 1
#SBATCH -t 8:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user="felicity.duan@gmail.com"
module load Anaconda
module load gcc
conda activate robodiff
export CPATH=/home/x_yufdu/.conda/envs/robodiff/include
cd /proj/rep-learning-robotics/users/x_yufdu/daxia/diffusion/consistency-policy
# can_mh_image

python eval.py --checkpoint /proj/rep-learning-robotics/users/x_yufdu/daxia/diffusion/consistency-policy/outputs/cd/image2/th/checkpoints/epoch=0200-test_mean_score=0.267.ckpt -o data/iros/eval/th_2_latest

/proj/rep-learning-robotics/users/x_yufdu/daxia/diffusion/consistency-policy/outputs/cd/image/th/checkpoints/epoch=0250-test_mean_score=0.333.ckpt