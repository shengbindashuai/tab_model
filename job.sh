#! /bin/bash
#SBATCH --job-name=orion_shift
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:4
#SBATCH --partition=gpu-rtx3090
#SBATCH --time=04:00:00


source activate orion
./Orion-MSP-main/scripts/train_stage1.sh