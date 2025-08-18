import torch
import os
import argparse
import tempfile
from pathlib import Path

from termcolor import colored

import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim

from nmt_cmr_parallels.models.encdec_recall_model import EncoderDecoderRecallmodel
from nmt_cmr_parallels.utils.checkpoint_utils import save_recall_model, load_recall_model
from torch.utils.data import DataLoader
from nmt_cmr_parallels.data.sequence_data import (create_dataset_from_csv,
                                              create_pretrained_semantic_embedding)
from nmt_cmr_parallels.data.word_vectors import initialize_lang_data
from nmt_cmr_parallels.utils.loss import LSAALoss
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

module_directory = Path(__file__).resolve()
module_directory = module_directory.parents[0]

def train_human_fits(checkpoint_load_path, logging_path, augmentation_factor=10, epochs=1000, lr=1e-4):
    os.makedirs(logging_path, exist_ok=True)

    data_dir = os.path.expanduser(os.path.join('~', '.seq_nlp_data'))
    peers_data_dir = os.path.join(module_directory,'resource','peers_human_data')

    train_path = os.path.join(peers_data_dir,'peers_train.csv')
    val_path = os.path.join(peers_data_dir,'peers_val.csv')
    test_path = os.path.join(peers_data_dir,'peers_test.csv')

    df1 = pd.read_csv(train_path)
    df2 = pd.read_csv(val_path)
    df3 = pd.read_csv(test_path)
    data = pd.concat([df1, df2, df3], ignore_index=True)

    subjects = data['Subject'].unique().tolist()

    embedding_dim = 50
    batch_size = 1
    augment_data = True
    free_recall = False

    best_valid_loss = float('inf')  # initialize with a high value
    epochs_without_improvement = 0  # counter for early stopping

    os.makedirs(data_dir, exist_ok=True)
    vocab = initialize_lang_data(data_dir, embedding_dim, glove_embedding=True)
    vocab_size = len(vocab)

    # Extract pretrained semantic embedding from Word2Vec
    pretrained_embedding = create_pretrained_semantic_embedding(data_dir, embedding_dim)

    # Function to initialize the model
    def initialize_model(checkpoint_load_path, device='cuda'):
        model, _ = load_recall_model(checkpoint_load_path, return_vocab=True, device=device)
        model.to(device)
        model.set_device(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        return model, optimizer

    # If we're testing for free recall, order of recall doesn't matter
    # so we use a simple set comparison loss
    if free_recall:
        criterion = LSAALoss()
    else:
        criterion = nn.CrossEntropyLoss()

    def get_teacher_forcing_ratio(epoch, total_epochs, initial_ratio=0.75, final_ratio=0.1):
        return initial_ratio - (initial_ratio - final_ratio) * (epoch / total_epochs)

    accumulation_steps = 4
    for subject in subjects:
        print(f"Tuning Subject {subject}")
        subject_dir = os.path.join(logging_path, f"subject_{subject}")
        os.makedirs(subject_dir,exist_ok=True)
        subject_df = data[data['Subject'] == subject]
        train_df, val_df = train_test_split(subject_df, test_size=0.1, random_state=42)
        
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_train_csv:
            train_df.to_csv(temp_train_csv.name, index=False)
            csv_training = temp_train_csv.name

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_val_csv:
            val_df.to_csv(temp_val_csv.name, index=False)
            csv_validation = temp_val_csv.name

        train_loader, val_loader, vocab = create_dataset_from_csv(csv_training=csv_training,
                                                        vocab_source='peers_vocab.json',
                                                        batch_size=batch_size,
                                                        augment_data=augment_data,
                                                        data_multiplier=augmentation_factor,
                                                        csv_validation=csv_validation)
        vocab_size = len(vocab)
        
        # Continue from a previous best fit if available
        if os.path.exists(os.path.join(subject_dir,f"subject_{subject}_final_checkpoint.pt")):
            model, optimizer = initialize_model(os.path.join(subject_dir,f"subject_{subject}_final_checkpoint.pt"))
            print("Continuing from previous final checkpoint...")
        else:
            model, optimizer = initialize_model(checkpoint_load_path)

        early_stopping = 20
        epochs_without_improvement = 0
        for epoch in range(epochs):
            # Training phase
            model.train()
            total_train_loss = 0
            optimizer.zero_grad()  # Move this line before the batch loop
            for i, (batch_inputs, batch_targets, seq_lengths) in enumerate(train_loader, 1):
                if not isinstance(batch_inputs, list):
                    batch_inputs, batch_targets = batch_inputs.to('cuda'), batch_targets.to('cuda')
                target_len = 16 if free_recall else batch_targets.shape[1]
                teacher_ratio = get_teacher_forcing_ratio(epoch, epochs, initial_ratio=0.75, final_ratio=0.05)
                output = model(batch_inputs, target_len, target_sequence=batch_targets, teacher_forcing_ratio=teacher_ratio)
                
                if free_recall:
                    loss = criterion(output, batch_inputs)
                else:
                    loss = criterion(output.reshape(-1, vocab_size), batch_targets.reshape(-1))
                
                loss = loss / accumulation_steps  # Normalize the loss
                loss.backward()
                
                if i % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                total_train_loss += loss.item() * accumulation_steps  # Denormalize the loss
            
            if i % accumulation_steps != 0:
                optimizer.step()
                optimizer.zero_grad()
            
            avg_train_loss = total_train_loss / len(train_loader)
            if epoch % 10 == 0:
                print(colored(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}", "red"))
                checkpoint_name = os.path.join(subject_dir, f"subject_{subject}_checkpoint.pt")
                save_recall_model(checkpoint_name, model)

            # Validation phase
            model.eval()
            total_val_loss = 0

            with torch.no_grad():
                for batch_inputs, batch_targets, seq_lengths in val_loader:
                    if not isinstance(batch_inputs, list):
                        batch_inputs, batch_targets = batch_inputs.to('cuda'), batch_targets.to('cuda')

                    target_len = 16 if free_recall else batch_targets.shape[1]
                    output = model(batch_inputs, target_len)

                    if free_recall:
                        loss = criterion(output, batch_inputs)
                    else:
                        loss = criterion(output.reshape(-1, vocab_size), batch_targets.reshape(-1))
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            if avg_val_loss < best_valid_loss:
                checkpoint_name = os.path.join(subject_dir,f"best_subject_{subject}_checkpoint.pt")
                save_recall_model(checkpoint_name, model)
                best_valid_loss = avg_val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if epochs_without_improvement > early_stopping:
                break
            
            if epoch % 10 == 0:
                print(colored(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}", "green"))
            if avg_train_loss < 0.5:
                break

        os.unlink(csv_training)
        os.unlink(csv_validation)

        final_checkpoint_name = os.path.join(subject_dir,f"subject_{subject}_final_checkpoint.pt")
        save_recall_model(final_checkpoint_name, model)

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for loading checkpoints.")
    
    parser.add_argument('--checkpoint_load_path', type=str, 
                        default=os.path.join(module_directory,'resource','pretrained_checkpoints', 'attention_128dim.pt'),
                        help='Path to the checkpoint file to be loaded')
    parser.add_argument('--logging_path', type=str, 
                        default=".",
                        help='Directory to save individual subject fits.')
    parser.add_argument('--augmentation_factor', type=int, 
                        default=5,
                        help='Sample multiplier for data augmentation')
    parser.add_argument('--epochs', type=int, 
                        default=500,
                        help='Max number of epochs')
    parser.add_argument('--lr', type=float, 
                        default=1e-3,
                        help='Learning rate')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    train_human_fits(args.checkpoint_load_path, args.logging_path, augmentation_factor=args.augmentation_factor,
                     epochs=args.epochs, lr=args.lr)

if __name__ == "__main__":
    main()