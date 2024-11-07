#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 12            # number of cores
#SBATCH --gres=gpu:a100:1
#SBATCH -t 0-24:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q public       # QOS
#SBATCH -e slurm.%A_%a.err # file to save job's STDERR
#SBATCH --export=NONE   # Purge the job-submitting shell environment
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=svishnu6@asu.edu

# Load necessary modules
echo "Loading mamba module..."
module load mamba

# Activate your Python environment
echo "Activating conda environment 'mm_contra_learn'..."
source activate mm_contra_learn

#Pip install dependencies
pip install -r requirements.txt

# Change to the project directory
echo "Changing to project directory '/CrisisMMD'..."
cd CrisisMMD/

# Train a model with three input views (img0, img1, txt0)
echo "Starting training..."

python main.py --datapath crisismmd_datasplit_all --image_folder CrisisMMD_v2.0 --train_steps 100000 --encoding_size 8 --batch_size 32 --lr 1e-4 --log_interval 100 --save_interval 1000 --phase pretrain --save_dir pretrained_models --task informative

echo "Training completed."


# Print completion time
echo "Job finished with exit code $? at: $(date)"