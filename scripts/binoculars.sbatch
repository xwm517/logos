#!/usr/bin/env bash
#SBATCH --job-name=binoculars
#SBATCH --account=cs-masters-2023
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=1-00:00:00
#SBATCH --output=/mnt/scratch/users/%u/logos/logs/%x-%j.log

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/logos/bin/activate
cd ~/logos

export HF_HOME=~/scratch/hf_cache
export HF_HUB_OFFLINE=1

python scripts/run_binoculars.py