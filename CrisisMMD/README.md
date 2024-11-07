# README

This README provides detailed instructions for training and evaluating models on the CrisisMMD dataset. There are two different training approaches available: contrastive learning and classification learning. Each approach can be used for various tasks, such as informative classification, humanitarian classification, or severity assessment.

## Setup

### Training Approaches

- **Contrastive Learning (Representation Learning)**: This approach aims to learn good feature representations using a contrastive InfoNCE loss. It involves encoding images and texts, comparing similar (positive) pairs to maximize agreement, and contrasting different (negative) pairs.
- **Classification Learning**: This is a supervised approach to predict the task-specific labels, such as `informative`, `humanitarian`, or `severity` labels.

### Prerequisites

- Python >= 3.7
- PyTorch >= 1.8
- [wandb](https://wandb.ai) for logging

Install the dependencies:

```bash
pip install -r requirements.txt
```

## Commands to Run Training and Evaluation
```bash
python main.py --datapath crisismmd_datasplit_all --image_folder CrisisMMD_v2.0 --task severity --batch_size 32 --train_steps 100000 --lr 1e-4 --train_mode classification
```

### Contrastive Learning Training

Use the following commands for training the model using contrastive learning with InfoNCE loss for each task:

## Pretraining Command

python main.py --datapath crisismmd_datasplit_all --image_folder CrisisMMD_v2.0 --train_steps 100 --encoding_size 8 --batch_size 32 --lr 1e-4 --log_interval 10 --save_interval 10000 --phase pretrain --save_dir pretrained_models --task informative

## Fine-Tuning Command

python main.py --datapath crisismmd_datasplit_all --image_folder CrisisMMD_v2.0 --task informative --train_steps 5000 --batch_size 32 --lr 1e-4 --log_interval 1000 --phase finetune --pretrain_dir pretrained_models --save_dir finetuned_models

## Evaluation Command

python main.py --datapath crisismmd_datasplit_all --image_folder CrisisMMD_v2.0 --task damage --batch_size 32 --lr 1e-4 --log_interval 1000 --phase evaluate --pretrain_dir pretrained_models --finetune_dir finetuned_models --encoding_size 8


## Arguments

- `--datapath`: Path to the folder containing the dataset annotation files.
- `--image_folder`: Path to the folder containing the images.
- `--task`: Task type (`informative`, `humanitarian`, `severity`).
- `--train_mode`: Training mode (`classification` or `contrastive`).
- `--train_steps`: Number of training steps.
- `--batch_size`: Batch size for training.
- `--lr`: Learning rate.
- `--evaluate`: Flag to evaluate the model instead of training.

