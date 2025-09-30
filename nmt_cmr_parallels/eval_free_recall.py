import torch
import argparse
import logging
import json
import os
import re

from torch.utils.data import DataLoader
from nmt_cmr_parallels.data.sequence_data import (create_sequence_dataset, collate_fn, 
                                              create_test_dataset_from_csv)
from nmt_cmr_parallels.utils.checkpoint_utils import load_recall_model

from copy import deepcopy

def get_checkpoints(directory, check_num):

    all_files = os.listdir(directory)
    checkpoint_files = [f for f in all_files if f.endswith('.pt')]
    checkpoint_files.sort(key=lambda f: int(re.findall('\d+', f)[0]) if re.findall('\d+', f) else 0)

    # Select first n checkpoints
    if check_num is not None:
        checkpoint_files = checkpoint_files[:check_num]

    return [os.path.join(directory, f) for f in checkpoint_files]

def evaluate_model(checkpoint_path,
                   num_sequences=1000, 
                   sequence_length=10, 
                   num_related_words=2,
                   test_csv = None,
                   results_path="recalls_results.json", 
                   peers_vocab=True,
                   model_type="encoderdecoder",
                   human_set_file=None,**kwargs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize the model and load the checkpoint
    model, vocab = load_recall_model(checkpoint_path, return_vocab=True, device=device)
    vocab_size = len(vocab)
    model.eval().to(device)
    model.set_device(device)

    # Create DataLoader for evaluation
    if test_csv is not None:
        dataset, vocab = create_test_dataset_from_csv(test_csv,
                                                      vocab=vocab,
                                                      vocab_source='peers_vocab.json' if peers_vocab else None,
                                                      batch_size=1)
    else:
        dataset, vocab = create_sequence_dataset(sequence_length=sequence_length,
                                                num_related_words=0,
                                                num_sequences=num_sequences,
                                                vocab=vocab,
                                                vocab_size=vocab_size,
                                                scrub_vocab=False,
                                                vocab_source='peers_vocab.json' if peers_vocab else None,
                                                human_set_file=human_set_file)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    inv_vocab = {v: k for k, v in vocab_index.items()}

    results = {'original_sequences': [], 'predicted_sequences': []}
    with torch.no_grad():
        for batch_inputs, batch_targets, seq_lengths in test_loader:

            if not isinstance(batch_inputs, list):
                batch_inputs, batch_targets = batch_inputs.to(device), batch_targets.to(device)

            if model_type == 'encoderdecoder':
                output = model(batch_inputs, seq_lengths[0])
            else:            
                output = model(batch_inputs)

            predicted_indices = torch.argmax(output, dim=-1)
            logging.debug(f"Predicted Tokens: {predicted_indices}")

            # Convert tensor of token indices to actual sequences of words
            original_sequences = [inv_vocab[idx] for idx in batch_inputs[0].tolist()]
            predicted_sequences = [inv_vocab[idx] if idx < len(inv_vocab) else 'Unknown' for idx in predicted_indices[0].tolist()]

            logging.info(50*'-')
            logging.info(f"Original Sequence: {original_sequences}")
            logging.info(f"Predicted Sequence: {predicted_sequences}")

            results['original_sequences'].append(original_sequences)
            results['predicted_sequences'].append(predicted_sequences)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {results_path}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a recurrent network for sequence recall.")
    
    parser.add_argument("--checkpoint_path", type=str, 
                        default='sequence_model.pth', 
                        help="Path for saving model checkpoint.")
    
    parser.add_argument('--checkpoint_dir', type=str, default=None, 
                        help='Directory containing model checkpoints')
    
    parser.add_argument('--check_num', type=int, 
                        default=None, 
                        help='Number of checkpoints to evaluate in directory')
    
    parser.add_argument("--results_path", type=str, 
                        default='recall_results.json', 
                        help="Path to json file for recall results.")
    
    parser.add_argument("--test_csv", type=str, 
                        default=None, 
                        help="Path to csv file containing human subject.")
    
    parser.add_argument("--human_set_file", type=str, 
                        default=None, 
                        help="Path to json file containing human subject data.")
                        
    # Dataset options
    parser.add_argument("--num_sequences", type=int, 
                        default=1000, 
                        help="Number of sequences to generate for training.")
    parser.add_argument("--sequence_length", type=int, 
                        default=16, 
                        help="Length of sequences for training.")
    
    parser.add_argument("--model_type", type=str, 
                        default="encoderdecoder",
                        choices=["encoderdecoder"], 
                        help="Select recall model configuration.")
    
    parser.add_argument("--peers_vocab", action="store_true", 
                        default=False, 
                        help="Use cached PEERS dataset vocabulary.")
    
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
    

    if args.checkpoint_dir is not None:

        checkpoints = get_checkpoints(args.checkpoint_dir, args.check_num)
        eval_dir = os.path.join(args.checkpoint_dir, "evaluations")
        os.makedirs(eval_dir, exist_ok=True)
        for checkpoint in checkpoints:
            args_dict = vars(args)
            args_dict['checkpoint_path'] = checkpoint
            results_file = os.path.splitext(os.path.split(checkpoint)[1])[0]
            args_dict['results_path'] = os.path.join(eval_dir, results_file + '_eval.json')
            evaluate_model(**args_dict)
    

    args_dict = vars(args)
    evaluate_model(**args_dict)