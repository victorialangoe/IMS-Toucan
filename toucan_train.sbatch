#!/bin/bash
#SBATCH --job-name="ims_toucan_training"
#SBATCH --partition=hgx2q       # This is the partition where g002 is located (or use the appropriate partition)
#SBATCH --time=7-00:00:00       # Maximum runtime
#SBATCH --nodes=1               # Request a single node
#SBATCH --ntasks=1              # Number of tasks
#SBATCH --cpus-per-task=4       # Number of CPU cores per task
#SBATCH --gres=gpu:a100:1       # Request 1 A100 GPU
#SBATCH --nodelist=g002         # Specify node g002 directly to target the A100 80GB GPUs
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
    --model_save_dir /home/victoria/IMS-Toucan/checkpoints/vestland/
