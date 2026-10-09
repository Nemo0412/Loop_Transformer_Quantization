#!/bin/bash
#SBATCH -J ouro_vfy
#SBATCH -c 8
#SBATCH -t 03:00:00
#SBATCH --mem=64G
#SBATCH --partition=l40s_public
#SBATCH --gres=gpu:l40s:1
#SBATCH --account=torch_pr_111_general
#SBATCH -o /scratch/ll5914/logs/ouro_vfy_%j.out
#SBATCH -e /scratch/ll5914/logs/ouro_vfy_%j.err

set -euo pipefail

echo "Job started: $(date)"
echo "Node: $(hostname)"

source /share/apps/anaconda3/2025.06/etc/profile.d/conda.sh
conda activate /scratch/ll5914/conda_envs/SVD
source /scratch/ll5914/conda_envs/ouro_tf455/bin/activate

export HF_HOME=/scratch/ll5914/hf_home
export OMP_NUM_THREADS=8
export PYTHONUNBUFFERED=1

cd /home/ll5914/Loop_Trans_Q

echo ""
echo "########## 1. quantizer is really low-bit ##########"
python -u verify_quant.py

echo ""
echo "########## 2. error decomposition e = p + q ##########"
python -u run_ouro_error.py --decompose --configs INT8 INT4 FP8 \
    --loops 8 --eval-seqs 8 --seq-len 512 --batch-size 2

echo ""
echo "########## 3. GSM8K generation smoke ##########"
python -u run_ouro_gsm8k.py --smoke

echo "Job finished: $(date)"
