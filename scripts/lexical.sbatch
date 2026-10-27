#!/usr/bin/env bash
#SBATCH --job-name=lexical
#SBATCH --account=cs-masters-2023
#SBATCH --partition=nodes
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --array=0-39
#SBATCH --output=/mnt/scratch/users/%u/logos/logs/%x-%A_%a.log

set -e
module purge
module load Python/3.11.3-GCCcore-12.3.0
source ~/venvs/logos/bin/activate
cd ~/logos

python scripts/run_lexical.py --task "$SLURM_ARRAY_TASK_ID"