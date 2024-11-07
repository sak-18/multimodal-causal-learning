import os
import torch
import wandb
import argparse
import numpy as np
import random
import json
import csv  # New import for CSV logging
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils import clip_grad_norm_
from torchvision.datasets.folder import pil_loader
import torchviz  # New import for torchviz

from datasets import CrisisMMDataset
from encoders import ImageEncoder, TextEncoder2D, TaskHead
from utils.losses import infonce_loss
from utils.infinite_iterator import InfiniteIterator


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str, required=True)
    parser.add_argument("--image_folder", type=str, required=True)
    parser.add_argument("--task", type=str, choices=['informative', 'humanitarian', 'damage'], required=False)
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--log_interval", type=int, default=1000)
    parser.add_argument("--save_interval", type=int, default=10000, help="Interval for saving models during pretraining.")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--save_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--encoding_size", type=int, default=128)
    parser.add_argument("--tau", type=float, default=1.0)
    
    # New argument to specify which phase to run
    parser.add_argument("--phase", type=str, choices=['pretrain', 'finetune', 'evaluate'], required=True, 
                        help="Specify which phase to run: pretrain, finetune, or evaluate.")
    parser.add_argument("--pretrain_dir", type=str, help="Directory to load pretrained encoder models from.")
    parser.add_argument("--finetune_dir", type=str, help="Directory to load finetuned task-specific head models from.")
    parser.add_argument("--metrics_file", type=str, default="eval_metrics.csv", help="CSV file to store evaluation metrics.")
    args = parser.parse_args()
    return args


def pretrain_step(batch, img_encoder, txt_encoder, loss_func, optimizer, params, device, step):
    image, text = batch['image'].to(device), batch['text'].to(device)

    # Get embeddings from image and text encoders
    hz_image = img_encoder(image)
    hz_text = txt_encoder(text)

    # **Visualize the model on the first batch**
    if step == 0:
        viz = torchviz.make_dot((hz_image, hz_text), params=dict(img_encoder.named_parameters()))
        viz.render(f"pretrain_model_viz_step_{step}", format="png")

    # Compute InfoNCE loss
    loss_value1 = loss_func(hz_image, hz_text)
    loss_value2 = loss_func(hz_text, hz_image)
    loss = 0.5 * (loss_value1 + loss_value2)  # Symmetric loss

    # Backpropagate and optimize
    optimizer.zero_grad()
    loss.backward()
    clip_grad_norm_(params, max_norm=2.0, norm_type=2)
    optimizer.step()

    return loss.item()


def fine_tune_step(batch, img_encoder, txt_encoder, task_head, optimizer, loss_func, device, task, step):
    """Task-specific fine-tuning step."""
    image, text, labels = batch['image'].to(device), batch['text'].to(device), batch['label'].to(device)

    # TODO: Check this. Convert labels to 1D if necessary
    if labels.dim() > 1 and task=='humanitarian':
        labels = torch.argmax(labels, dim=1)

    # Get embeddings
    hz_image = img_encoder(image)
    hz_text = txt_encoder(text)

    # Combine embeddings (concatenate, averaged, etc.)
    combined_features = (hz_image + hz_text) / 2.0

    # Pass through task-specific head
    task_output = task_head(combined_features)

    # Compute task-specific loss (CrossEntropyLoss)
    loss = loss_func(task_output, labels)

    # Backpropagate and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def evaluate(task_head, data_loader, img_encoder, txt_encoder, device, loss_func):
    """Evaluate the model on the test set and return the average loss and metrics."""
    task_head.eval()
    img_encoder.eval()
    txt_encoder.eval()

    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in data_loader:
            image, text, labels = batch['image'].to(device), batch['text'].to(device), batch['label'].to(device)
            
            # If labels are one-hot, convert to single-label format
            if labels.dim() > 1 and labels.size(1) > 1:
                labels = torch.argmax(labels, dim=1)

            hz_image = img_encoder(image)
            hz_text = txt_encoder(text)
            #see if this fusion is the best??
            #combined_features = (hz_image + hz_text) / 2.0
            combined_features = (hz_image + hz_text) / 2.0
            task_output = task_head(combined_features)

            # Compute loss
            loss = loss_func(task_output, labels)
            total_loss += loss.item()

            # Get predictions and true labels
            preds = torch.argmax(task_output, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            num_batches += 1

    # Calculate average loss
    avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    
    # Calculate classification metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return avg_loss, accuracy, precision, recall, f1


def main():
    args = parse_args()
    # Initialize wandb
    run_name = f"task_{args.task}_phase_{args.phase}_enc_size_{args.encoding_size}_batch_{args.batch_size}_lr_{args.lr}"
    wandb.init(project="CrisisMMD-Contrastive", config=args, name=run_name)

    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    # Set seed
    if args.seed is not None:
        np.random.seed(args.seed)
        random.seed(args.seed)
        torch.manual_seed(args.seed)

    # Hardcode the mean and std here (calculated separately)
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    # Specify the vocab file path
    vocab_filepath = os.path.join(args.datapath, "vocab.json")
    
    # Define dataset and dataloader
    train_dataset = CrisisMMDataset(
        annotation_file=os.path.join(args.datapath, f"task_{args.task}_text_img_train.tsv"),
        task=args.task,
        image_folder=args.image_folder,
        vocab_filepath=vocab_filepath,  # Pass the vocab file path
        transform=transform
    )

    val_dataset = CrisisMMDataset(
        annotation_file=os.path.join(args.datapath, f"task_{args.task}_text_img_dev.tsv"),
        task=args.task,
        image_folder=args.image_folder,
        vocab_filepath=vocab_filepath,
        transform=transform
    )

    test_dataset = CrisisMMDataset(
        annotation_file=os.path.join(args.datapath, f"task_{args.task}_text_img_test.tsv"),
        task=args.task,
        image_folder=args.image_folder,
        vocab_filepath=vocab_filepath,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    train_iterator = InfiniteIterator(train_loader)
    val_iterator = InfiniteIterator(val_loader)
    test_iterator = InfiniteIterator(test_loader)

    # Define model components
    img_encoder = ImageEncoder(output_size=args.encoding_size).to(device)
    txt_encoder = TextEncoder2D(output_size=args.encoding_size, sequence_length=train_dataset.max_sequence_length, vocab_size=train_dataset.vocab_size).to(device)
    # Define separate task heads for each task
    task_heads = {
        "informative": TaskHead(input_size=args.encoding_size, num_classes=len(val_dataset.data['label'].unique())).to(device),
        "humanitarian": TaskHead(input_size=args.encoding_size, num_classes=len(val_dataset.data['label'].unique())).to(device),
        "damage": TaskHead(input_size=args.encoding_size, num_classes=len(val_dataset.data['label'].unique())).to(device)
    }

    # Load pretrained encoders if applicable
    if args.pretrain_dir and args.phase in ['finetune', 'evaluate']:
        img_encoder.load_state_dict(torch.load(os.path.join(args.pretrain_dir, "img_encoder_final.pt")))
        txt_encoder.load_state_dict(torch.load(os.path.join(args.pretrain_dir, "txt_encoder_final.pt")))

    if args.phase == "pretrain":
        # Final model saving after pretraining is complete
        os.makedirs(args.save_dir, exist_ok=True)
        # save args to disk (only for training)        
        with open(os.path.join(args.save_dir, 'args.json'), 'w') as fp:
            json.dump(args.__dict__, fp)

        # Pretraining loop
        pretrain_optimizer = torch.optim.Adam(list(img_encoder.parameters()) + list(txt_encoder.parameters()), lr=args.lr)
        sim_metric = torch.nn.CosineSimilarity(dim=-1)
        criterion = torch.nn.CrossEntropyLoss()
        contrastive_loss = lambda z1, z2: infonce_loss(z1, z2, criterion=criterion, sim_metric=sim_metric, tau=args.tau)

        step = 0
        while step < args.train_steps:
            batch = next(train_iterator)
            print(f"Loaded batch {step + 1}")
            loss = pretrain_step(batch, img_encoder, txt_encoder, contrastive_loss, pretrain_optimizer, 
                                 list(img_encoder.parameters()) + list(txt_encoder.parameters()), device, step)

            # Log pretraining loss to wandb
            if (step + 1) % args.log_interval == 0:
                print(f"Pretrain Step [{step + 1}], Loss: {loss:.4f}")
                wandb.log({"pretrain_loss": loss})

            # Save model every save_interval steps
            if (step + 1) % args.save_interval == 0:
                img_save_path = os.path.join(args.save_dir, f"img_encoder_step_{step+1}.pt")
                txt_save_path = os.path.join(args.save_dir, f"txt_encoder_step_{step+1}.pt")
                torch.save(img_encoder.state_dict(), img_save_path)
                torch.save(txt_encoder.state_dict(), txt_save_path)
                print(f"Saved models at step {step + 1}")

            step += 1


        torch.save(img_encoder.state_dict(), os.path.join(args.save_dir, "img_encoder_final.pt"))
        torch.save(txt_encoder.state_dict(), os.path.join(args.save_dir, "txt_encoder_final.pt"))
        
    elif args.phase == "finetune":
        # Fine-tuning loop (requires pre-trained models)
        fine_tune_optimizer = torch.optim.Adam(list(task_heads[args.task].parameters()), lr=args.lr)
        task_loss = torch.nn.CrossEntropyLoss()

        step = 0
        while step < args.train_steps:
            batch = next(val_iterator)
            loss = fine_tune_step(batch, img_encoder, txt_encoder, task_heads[args.task], fine_tune_optimizer, task_loss, device, args.task, step)#batch, img_encoder, txt_encoder, task_head, optimizer, loss_func, device, step

            # Log fine-tuning loss to wandb
            if (step + 1) % args.log_interval == 0:
                print(f"Fine-tune Step [{step + 1}], Loss: {loss:.4f}")
                wandb.log({"fine_tune_loss": loss})

            step += 1

        # Save the fine-tuned task-specific head
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(task_heads[args.task].state_dict(), os.path.join(args.save_dir, f"task_{args.task}_enc_size_{args.encoding_size}.pt"))

    elif args.phase == "evaluate":
        # Load the task-specific head from the fine-tuned directory
        task_heads[args.task].load_state_dict(torch.load(os.path.join(args.finetune_dir, f"task_{args.task}_enc_size_{args.encoding_size}.pt")))

        # Perform evaluation
        print("Evaluating on test set...")
        test_loss, accuracy, precision, recall, f1 = evaluate(
            task_heads[args.task], test_loader, img_encoder, txt_encoder, device, torch.nn.CrossEntropyLoss()
        )
        print(f"Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        
        # Log evaluation metrics to wandb
        wandb.log({
            "test_loss": test_loss,
            "test_accuracy": accuracy,
            "test_precision": precision,
            "test_recall": recall,
            "test_f1": f1
        })

        # Save evaluation metrics to CSV
        with open(args.metrics_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Test Loss", "Accuracy", "Precision", "Recall", "F1-score"])
            writer.writerow([test_loss, accuracy, precision, recall, f1])


if __name__ == "__main__":
    main()
