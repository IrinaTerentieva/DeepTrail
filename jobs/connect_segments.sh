#!/bin/bash
#SBATCH --job-name=kirby_connect_segments
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128GB
#SBATCH --time=24:00:00
#SBATCH --partition=cpu2021
#SBATCH --output=/home/irina.terenteva/Logs/kirby_connect_segments%j.txt
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=irina.terenteva@ucalgary.ca

module load anaconda3

# Activate your Conda environment
source /home/irina.terenteva/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_gpu_env

# Clear any GPU cache
python -c "import torch; torch.cuda.empty_cache()"

# Move to the directory containing your Metaflow script
cd /home/irina.terenteva/HumanFootprint/Scripts/postprocessing

# Run your Metaflow flow with correct paths
python connect_segments.py run
