#!/bin/bash
#SBATCH --job-name=Tuyen
#SBATCH --time=240:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=80G
#SBATCH --gres=gpu:1
#SBATCH --output=/home/user09/tuyen/Data-Filtering/logs/%x_%j.out
#SBATCH --error=/home/user09/tuyen/Data-Filtering/logs/%x_%j.err

python solve.py