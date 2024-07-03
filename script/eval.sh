#!/bin/bash

#SBATCH --job-name=ibl-fm
#SBATCH --output=ibl-fm.out
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH -t 2-12:00:00 
#SBATCH --mem=64g

. ~/.bashrc
echo $TMPDIR
conda activate ibl-mm

cd ../

python src/eval_multi_modal.py --mask_mode temporal \
                               --mask_ratio 0.3 \
                               --eid 51e53aff-1d5d-4182-a684-aba783d50ae5 \
                               --seed 42 \
                               --base_path ./ \
                               --save_plot \
                               --mask_type embd \
                               --wandb 
cd script

conda deactivate