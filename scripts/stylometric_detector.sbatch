#!/usr/bin/env bash
#SBATCH --job-name=stylometric-detector
#SBATCH --account=cs-masters-2023
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-39%4
#SBATCH --output=/mnt/scratch/users/%u/logos/logs/%x-%A_%a.log

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/logos/bin/activate
cd ~/logos

export HF_HOME=~/scratch/hf_cache
export HF_HUB_OFFLINE=1

python scripts/run_stylometric_detector.py --task "$SLURM_ARRAY_TASK_ID"
