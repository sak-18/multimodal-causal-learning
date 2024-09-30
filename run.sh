#!/bin/bash
#SBATCH -N 1            # number of nodes
#SBATCH -c 12            # number of cores
#SBATCH --gres=gpu:a100:1
#SBATCH -t 0-22:00:00   # time in d-hh:mm:ss
#SBATCH -p general      # partition
#SBATCH -q public       # QOS
#SBATCH -e slurm.%A_%a.err # file to save job's STDERR
#SBATCH --export=NONE   # Purge the job-submitting shell environment
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=svishnu6@asu.edu

# Load necessary modules
echo "Loading mamba module..."
module load mamba

# Change to the project directory
#echo "Changing to project directory '/multimodal-contrastive-learning'..."
# cd multimodal-contrastive-learning/

# Activate your Python environment
echo "Activating conda environment 'mm_contra_learn'..."
source activate mm_contra_learn

pip install -r requirements.txt

# Train a model with three input views (img0, img1, txt0)
# echo "Starting training..."
# python main_imgtxt.py --datapath "m3di" --model-id "imgtxt_exp_run_encoding_size_04" --encoding-size 4 --workers 1
# echo "Training completed."

# # Evaluate the model
# echo "Starting evaluation..."
# python main_imgtxt.py --datapath "m3di" --model-id "imgtxt_exp_run_encoding_size_04" --encoding-size 4 --workers 1 --load-args --evaluate
# echo "Evaluation completed."

# Train a model with three input views (img0, img1, txt0)
echo "Starting training..."
python main_imgtxt.py --datapath "m3di" --model-id "imgtxt_exp_run_encoding_size_08" --encoding-size 8 --workers 1
echo "Training completed."

# # Evaluate the model
# echo "Starting evaluation..."
# python main_imgtxt.py --datapath "m3di" --model-id "imgtxt_exp_run_encoding_size_08" --encoding-size 8 --workers 1 --load-args --evaluate
# echo "Evaluation completed."

# Print completion time
echo "Job finished with exit code $? at: $(date)"