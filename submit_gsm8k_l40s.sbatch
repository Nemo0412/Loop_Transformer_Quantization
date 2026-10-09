#!/bin/bash
#SBATCH -J ouro_gsm
#SBATCH -c 8
#SBATCH -t 08:00:00
#SBATCH --mem=64G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=torch_pr_111_general
#SBATCH -o /scratch/ll5914/logs/ouro_gsm_%j.out
#SBATCH -e /scratch/ll5914/logs/ouro_gsm_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD
source /scratch/ll5914/conda_envs/ouro_tf455/bin/activate

export HF_HOME=/scratch/ll5914/hf_home
export HF_DATASETS_CACHE=/scratch/ll5914/hf_datasets
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd /home/ll5914/Loop_Trans_Q
python -u run_ouro_gsm8k.py "$@"

echo "Job finished: $(date)"
