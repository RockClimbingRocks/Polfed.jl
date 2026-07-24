#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --job-name=sleepy-job
#SBATCH --output=./files/%x/%A.out
#SBATCH --error=./files/%x/%A.err
#SBATCH --time=2-00:00:00
#SBATCH --mem=500G
#SBATCH --partition=grace
#SBATCH --nodelist=g[01-08]
#SBATCH --gres=gpu:1
#SBATCH --exclusive

echo "Starting GPU sleep job"
hostname
date

sleep 2d

echo "Ending GPU sleep job"
date