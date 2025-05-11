#!/bin/bash
#SBATCH --job-name="trondelag_toucan"  
#SBATCH --partition=hgx2q     
#SBATCH --time=7-00:00:00      # Maximum runtime
#SBATCH --nodes=1               
#SBATCH --ntasks=1              
#SBATCH --cpus-per-task=4       # Number of CPU cores per task
#SBATCH --gres=gpu:1            # Allocate 1 GPU (any type)
#SBATCH --output=run2_1%j.out
#SBATCH --error=run2_1%j.err
#SBATCH --gres=gpu:a100:1       # Request 1 A100 GPU
#SBATCH --nodelist=g002         # g002: to target the A100 80GB GPUs


module purge
module load slurm/21.08.8
module load cuda12.2/toolkit/12.2.2  

# Activate my conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate toucan_env

# Check Python path for debud
which python

# Check GPU allocation for debug
echo "Checking GPU allocation with nvidia-smi:"
nvidia-smi

# Enable eskeap for TTS
export ESPEAK_PATH=/home/victoria/.local/bin/espeak
export LD_LIBRARY_PATH=/home/victoria/.local/lib:$LD_LIBRARY_PATH

# Run training script
srun python run_training_pipeline.py finetuning_example_simple \
    --gpu_id $CUDA_VISIBLE_DEVICES \
    --model_save_dir /home/victoria/data/IMS-Toucan/checkpoints/trondelag/ \