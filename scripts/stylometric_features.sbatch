#!/usr/bin/env bash
#SBATCH --job-name=stylometric
#SBATCH --account=cs-masters-2023
#SBATCH --partition=nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=/mnt/scratch/users/%u/logos/logs/%x-%j.log

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/logos/bin/activate
cd ~/logos

python scripts/run_stylometric_features.py
