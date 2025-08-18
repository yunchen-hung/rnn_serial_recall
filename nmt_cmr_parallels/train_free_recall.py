import torch
import logging
import argparse
import os
import json
import itertools

from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from nmt_cmr_parallels.models.encdec_recall_model import EncoderDecoderRecallmodel
from nmt_cmr_parallels.utils.checkpoint_utils import save_recall_model, load_recall_model
from nmt_cmr_parallels.utils.training_utils import InverseSquareRootSchedule
from nmt_cmr_parallels.data.sequence_data import (create_sequence_dataset,
                                              create_dataset_from_csv,
                                              create_dataloaders,
                                              load_dataset,
                                              save_dataset, 
                                              create_pretrained_semantic_embedding)
from nmt_cmr_parallels.data.word_vectors import initialize_lang_data
from nmt_cmr_parallels.utils.loss import LSAALoss
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

DEFAULT_DATA_PATH = os.path.expanduser(os.path.join('~', '.seq_nlp_data'))

def train_seq_model(num_sequences=1000,
                    sequence_length=10,
                    vocab_size = 500,
                    embedding_dim=300,
                    hidden_dim=50,
                    rnn_mode="LSTM",
                    model_type="encoderdecoder",
                    attention_type="luong",
                    checkpoint_path='sequence_model.pth',
                    checkpoint_load_path=None,
                    lr=0.001,
                    early_stopping_patience=3,
                    batch_size=8,
                    disable_semantic_embedding=False,
                    serial_recall=False,
                    dropout=None,
                    epochs=10, 
                    data_dir=DEFAULT_DATA_PATH,
                    log_dir="seq_model_runs", 
                    peers_vocab=False, 
                    load_dataset_path=None,
                    save_dataset_path=None, 
                    checkpoint_save_freq=None,
                    seq_tokens=False,
                    csv_training=None,
                    csv_validation=None,
                    augment_data=False,
                    data_multiplier=10,
                    random_decoding_init=False, **kwargs):
                    
    
    # Initialize the Tensorboard writer
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    best_valid_loss = float('inf')  # initialize with a high value
    epochs_without_improvement = 0  # counter for early stopping
    
    os.makedirs(data_dir,exist_ok=True)
    vocab = initialize_lang_data(data_dir, embedding_dim, glove_embedding=True)
    vocab_size = len(vocab)
    logging.info("Vocab loaded.")

    # Generate datasets
    train_loader, val_loader = None, None
    if load_dataset_path is None and csv_training is None and csv_validation is None:
        dataset, vocab = create_sequence_dataset(sequence_length=sequence_length,
                                                num_related_words=0,
                                                num_sequences=num_sequences,
                                                vocab=vocab,
                                                vocab_size=vocab_size,
                                                scrub_vocab=True,
                                                vocab_source='peers_vocab.json' if peers_vocab else None,
                                                use_seq_tokens=seq_tokens)
    elif csv_training is not None and csv_validation is not None:
        train_loader, val_loader, vocab = create_dataset_from_csv(csv_training=csv_training,
                                                                vocab_source='peers_vocab.json' if peers_vocab else None,
                                                                batch_size=batch_size,
                                                                csv_validation=csv_validation,
                                                                augment_data=augment_data,
                                                                data_multiplier=data_multiplier,
                                                                use_seq_tokens=seq_tokens)
    else:
        dataset = load_dataset(load_dataset_path)
        vocab = dataset.vocab
    vocab_size = len(vocab)
    if save_dataset_path is not None:
        save_dataset(dataset,save_dataset_path)
    if train_loader is None:
        train_loader, val_loader = create_dataloaders(dataset,batch_size=batch_size,split_ratio=0.9)

    logging.info("Sequence dataset generated.")
    
    # Extract pretrained semantic embedding from Word2Vec
    pretrained_embedding = None
    if not disable_semantic_embedding:
        pretrained_embedding = create_pretrained_semantic_embedding(data_dir,embedding_dim)
        logging.info("Pretrained embedding loaded.")

    # Initialize the model and optimizer
    if checkpoint_load_path is not None:

        # load the checkpoint
        model, vocab = load_recall_model(checkpoint_load_path, return_vocab=True)
        vocab_size = len(vocab)
        logging.info('Model loaded successfully.')

    else:
        use_attention = False if attention_type=='none' else True
        if model_type == 'encoderdecoder':
            model = EncoderDecoderRecallmodel(vocab_size, embedding_dim, hidden_dim, 
                                            use_attention=use_attention, 
                                            pretrained_embedding=pretrained_embedding,
                                            rnn_mode=rnn_mode,
                                            attention_type=attention_type, 
                                            vocab=vocab,
                                            dropout=dropout,
                                            device='cuda')

    model.to('cuda')
    model.set_device('cuda')
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = InverseSquareRootSchedule(optimizer, warmup_steps=4000)

    # If we're testing for free recall, order of recall doesn't matter
    # so we use a simple set comparison loss
    if serial_recall:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = LSAALoss()

    for epoch in range(epochs):

        # Training phase
        model.train()
        total_train_loss = 0
        
        for batch_inputs, batch_targets, seq_lengths in tqdm(train_loader, desc=f"Training epoch {epoch+1}/{epochs}"):
            optimizer.zero_grad()

            if not isinstance(batch_inputs, list):
                batch_inputs, batch_targets = batch_inputs.to('cuda'), batch_targets.to('cuda')

            if model_type == 'encoderdecoder':
                output = model(batch_inputs, seq_lengths[0], rand_init_decoder=random_decoding_init)
            else:            
                output = model(batch_inputs, rand_init_decoder=random_decoding_init)

            if batch_targets.shape[1] < output.shape[1]:
                output = output[:,:batch_targets.shape[1], :]

            if serial_recall:
                loss = criterion(output.reshape(-1, vocab_size), batch_targets.reshape(-1))
            else:
                loss = criterion(output, batch_targets)
                
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step(avg_train_loss)
        logging.info(f"Epoch {epoch+1}/{epochs}, Training Loss: {avg_train_loss}")

        # Log the training loss to TensorBoard
        writer.add_scalar("Loss/train", avg_train_loss, epoch)

        # Validation phase
        model.eval()
        total_val_loss = 0

        with torch.no_grad():
            for batch_inputs, batch_targets, seq_lengths in tqdm(val_loader, desc=f"Validating epoch {epoch+1}/{epochs}"):

                if not isinstance(batch_inputs, list):
                    batch_inputs, batch_targets = batch_inputs.to('cuda'), batch_targets.to('cuda')


                if model_type == 'encoderdecoder':
                    output = model(batch_inputs, seq_lengths[0], rand_init_decoder=random_decoding_init)
                else:            
                    output = model(batch_inputs)

                if batch_targets.shape[1] < output.shape[1]:
                    output = output[:,:batch_targets.shape[1], :]

                if serial_recall:
                    loss = criterion(output.reshape(-1, vocab_size), batch_targets.reshape(-1))
                else:
                    loss = criterion(output, batch_targets)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        logging.info(f"Epoch {epoch+1}/{epochs}, Validation Loss: {avg_val_loss}")

        # Log the validation loss to TensorBoard
        writer.add_scalar("Loss/val", avg_val_loss, epoch)

        # Check for improvement in validation loss
        if avg_val_loss < best_valid_loss:
            best_valid_loss = avg_val_loss
            epochs_without_improvement = 0
            # Save the best model parameters
            save_recall_model(checkpoint_path, model)
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            logging.info("Early stopping triggered.")
            break

        # Save checkpoint if necessary
        if checkpoint_save_freq is not None and epoch % checkpoint_save_freq == 0:
            stem, ext = os.path.splitext(checkpoint_path)
            save_recall_model(stem+f"_epoch_{epoch}"+ext, model)

    writer.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a recurrent network for sequence recall.")

    # Training Options
    
    parser.add_argument("--epochs", type=int, 
                        default=10, 
                        help="Number of training epochs.")
    
    parser.add_argument("--embedding_dim", type=int, 
                        default=50, choices=[50,100,200,300],
                        help="Dimension of embeddings (Available embedding sizes coincide with GloVe embedding sizes).")
    
    parser.add_argument("--rnn_mode", type=str, 
                        default="GRU",
                        choices=["LSTM","GRU"], 
                        help="Select type of recurrent network cell to be used in recall model.")
    
    parser.add_argument("--model_type", type=str, 
                        default="encoderdecoder",
                        choices=["encoderdecoder"], 
                        help="Select recall model configuration.")
    
    parser.add_argument("--attention_type", type=str, 
                        default="none",
                        choices=["bahdanau", "luong","none"], 
                        help="Select attention mechanism.")
    
    parser.add_argument("--vocab_size", type=int, 
                        default=5000, 
                        help="Size of testing vocabulary for generated dataset.")
    
    parser.add_argument("--hidden_dim", type=int, 
                        default=50, 
                        help="Dimension of hidden states.")
    
    parser.add_argument("--checkpoint_save_freq", type=int, 
                        default=None, 
                        help="Save a checkpoint every n episodes (default to only saving best).")
    
    parser.add_argument("--lr", type=float, 
                        default=0.001, 
                        help="Learning rate.")
    
    parser.add_argument("--batch_size", type=int, 
                        default=8, 
                        help="Batch size for training.")
    
    parser.add_argument("--early_stopping_patience", type=int, 
                        default=5, 
                        help="Dimension of hidden states.")
    
    parser.add_argument("--log_dir", type=str, 
                        default="seq_model_runs", 
                        help="Tensorboard logging directory.")
    
    parser.add_argument("--checkpoint_load_path", type=str, 
                        default=None, 
                        help="Path to a checkpoint path to load from.")
    
    parser.add_argument("--checkpoint_path", type=str, 
                        default='sequence_model.pth', 
                        help="Path for saving model checkpoint.")
    
    parser.add_argument("--grid_file_path", type=str, 
                        default=None, 
                        help="Path to hyperparameter grid JSON file.")
    
    # Dataset options
    parser.add_argument("--num_sequences", type=int, 
                        default=30000, 
                        help="Number of sequences to generate for training.")
    
    parser.add_argument("--sequence_length", type=int, 
                        default=14, 
                        help="Length of sequences for training.")

    parser.add_argument("--seq_tokens", action="store_true", 
                        default=False, 
                        help="Use sequence tokens to denote beginning and end of sequence.")
    
    parser.add_argument("--peers_vocab", action="store_true", 
                        default=False, 
                        help="Use cached PEERS dataset vocabulary.")
    
    parser.add_argument("--csv_training", type=str,
                         default=None,
                         help="Path to a CSV file of training samples.")
    
    parser.add_argument("--csv_validation", type=str,
                         default=None,
                         help="Path to a CSV file of validation samples.")
    
    parser.add_argument("--augment_data", action="store_true",
                        default=False,
                        help="Enable data augmentation (via synonomous word substitution) for human data.")
    
    parser.add_argument("--data_multiplier", type=int,
                        default=3,
                        help="Data multiplier for number of samples in augmented data set.")
    
    # Model options
    parser.add_argument("--disable_semantic_embedding", action="store_true", 
                        default=False, 
                        help="Disable pretrained embedding layer for semantic association (i.e. no GloVe embeddings.).")
    
    parser.add_argument("--serial_recall", action="store_true", 
                        default=False, 
                        help="Enable regular serial recall (i.e. CELoss).")

    parser.add_argument("--random_decoding_init", action="store_true",
                        default=False,
                        help="Set to use random decoder input for first decoding step (rather than 0).")

    parser.add_argument("--dropout", type=float, 
                        default=None, 
                        help="Rate of dropout before recurrent layers.")
    ######################
    
    parser.add_argument("--data_dir", type=str, 
                        default=DEFAULT_DATA_PATH, 
                        help="Path for NLTK downloading data.")
    
    parser.add_argument("-v", "--verbose", action="store_true", 
                        help="Enable verbose logging mode.")
    parser.add_argument("-d", "--debug", action="store_true", 
                        help="Enable debugging logging mode.")
    
    args  = parser.parse_args()

    if args.verbose:
        logging_level = logging.INFO
    elif args.debug:
        logging_level = logging.DEBUG
    else:
        logging_level = logging.WARNING
    logging.basicConfig(level=logging_level, format='%(asctime)s - %(levelname)s - %(message)s')
    args_dict = vars(args)
    args_dict['checkpoint_path'] = os.path.join(args_dict['log_dir'], args_dict['checkpoint_path'])

    if args.grid_file_path is not None:

        os.makedirs('grid_experiment',exist_ok=True)
        with open(args.grid_file_path, 'r') as f:
            param_grid = json.load(f)

        for i, params in enumerate(itertools.product(*param_grid.values())):
            # Update args_dict with the current combination of hyperparameters
            current_params = dict(zip(param_grid.keys(), params))
            args_dict.update(current_params)

            # Create a new checkpoint path and log_dir for the current run configuration
            log_dir = os.path.join('grid_experiment',f"grid_logs_{i}")
            os.makedirs(log_dir,exist_ok=True)
            checkpoint_path = os.path.join(log_dir,"best_checkpoint.pth")
            with open(os.path.join(log_dir,"parameters.json"), 'w') as f:
                json.dump(current_params,f,indent=2)

            args_dict['checkpoint_path'] = checkpoint_path
            args_dict['log_dir'] = log_dir

            logging.info(f"Current Experiment Configuration: {args_dict}")
            train_seq_model(**args_dict)

    else:

        train_seq_model(**args_dict)