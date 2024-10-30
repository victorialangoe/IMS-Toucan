#!/bin/bash
#SBATCH --job-name="ims_toucan_training"
#SBATCH --partition=a100q  # or mi100q
#SBATCH --time=7-00:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --output=ims_toucan_training_%j.out
#SBATCH --error=ims_toucan_training_%j.err

module load cuda11.8/toolkit/11.8.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate toucan_env

which python 

export ESPEAK_PATH=/home/victoria/.local/bin/espeak
export LD_LIBRARY_PATH=/home/victoria/.local/lib:$LD_LIBRARY_PATH

python run_training_pipeline.py finetuning_example_simple \
    --gpu_id 0 \
    --model_save_dir /home/victoria/IMS-Toucan/checkpoints/ \
