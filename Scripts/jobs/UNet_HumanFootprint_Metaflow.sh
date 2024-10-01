#!/bin/bash
#SBATCH --job-name=unet_humanfootprint
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64GB
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-a100
#SBATCH --output=/home/irina.terenteva/Logs/unet_humanfootprint%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irina.terenteva@ucalgary.ca

# Load necessary modules
module load cuda/11.3.0
module load anaconda3

# Set environment variable for CUDA memory management
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate your Conda environment
source /home/irina.terenteva/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu_env

# Log GPU usage
nvidia-smi

export ENVIRONMENT=hpc

# Clear any GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Move to the directory containing your Metaflow script
cd /home/irina.terenteva/HumanFootprint/Scripts/training

# Run your Metaflow flow with correct paths
python U-Net_HumanFootprint.py run
