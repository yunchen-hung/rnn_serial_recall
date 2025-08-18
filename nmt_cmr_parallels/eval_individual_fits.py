import torch
import os
import tempfile
import random
import json
import argparse 

from termcolor import colored

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

from nmt_cmr_parallels.models.encdec_recall_model import EncoderDecoderRecallmodel
from nmt_cmr_parallels.utils.checkpoint_utils import load_recall_model
from torch.utils.data import DataLoader
from nmt_cmr_parallels.data.sequence_data import (create_test_dataset_from_csv, collate_fn)
from nmt_cmr_parallels.utils.evaluation_utils import (calculate_conditional_recall_probability, 
                                                  calculate_recall_probability,
                                                  calculate_first_recall_probability,
                                                  calculate_conditional_recall_vs_similarity)
torch.autograd.set_detect_anomaly(True)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

data_dir = os.path.expanduser(os.path.join('~', '.seq_nlp_data'))
module_directory = Path(__file__).resolve()
module_directory = module_directory.parents[0]

def calculate_rmse(trajectory1, trajectory2):
    return np.sqrt(np.mean((np.array(trajectory1) - np.array(trajectory2)) ** 2))

def average_behavior_curves(*lists):
    if len(set(len(lst) for lst in lists)) != 1:
        raise ValueError("All lists must have the same length")
    
    num_lists = len(lists)
    averaged_list = []

    for items in zip(*lists):
        averaged_value = sum(items) / num_lists
        averaged_list.append(averaged_value)
    
    return averaged_list

def evaluate_subject_fits(root_dir, calc_semantic=False):

    peers_data_dir = os.path.join(module_directory,'resource','peers_human_data')
    test_path = os.path.join(peers_data_dir,'peers_test.csv')
    df = pd.read_csv(test_path)
    dataframe = pd.concat([df], ignore_index=True)
    subjects = dataframe['Subject'].unique().tolist()

    omit_first_k = 1
    end_position = 16

    # Function to initialize the model
    def initialize_model(checkpoint_load_path, device='cuda'):
        model = load_recall_model(checkpoint_load_path, return_vocab=False, device=device)
        model.to(device)
        model.set_device(device)
        return model

    subject_likelihoods = []
    spc_rmses, crp_rmses, pfr_rmses, semantic_rmses = [], [], [], []
    spc_vals, neg_crp_vals, pos_crp_vals, frp_vals = [], [], [], []
    for subject in subjects:

        print(f"Evaluating Subject {subject}")
        subject_dir = os.path.join(root_dir, f"subject_{subject}")
        os.makedirs(subject_dir,exist_ok=True)
        subject_df = dataframe[dataframe['Subject'] == subject]

        # Load Subject model
        subject_model_path = os.path.join(subject_dir,f"subject_{subject}_final_checkpoint.pt")
        if not os.path.exists(subject_model_path):
            print(f"No valid checkpoint found.")
            continue
        model = initialize_model(subject_model_path)
        
        # Generate temporary file from subject test data
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.csv') as temp_test_csv:
            subject_df.to_csv(temp_test_csv.name, index=False)
            csv_test = temp_test_csv.name

        # Create 
        test_dataset, vocab = create_test_dataset_from_csv(csv_test=csv_test,
                                                        vocab_source='peers_vocab.json')
        test_loader = DataLoader(test_dataset, shuffle=False, collate_fn=collate_fn)
        vocab_size = len(vocab)
        
        # Inverse vocabulary
        vocab_index = {word: idx for idx, word in enumerate(vocab)}
        inv_vocab = {v: k for k, v in vocab_index.items()}

        # Perform predictions on test data
        results = {'original_sequences': [], 'predicted_sequences': [], 'human_sequences': []}
        subject_likelihood = 0
        with torch.no_grad():
            model.eval()
            for batch_inputs, batch_targets, seq_lengths in test_loader:

                if not isinstance(batch_inputs, list):
                    batch_inputs, batch_targets = batch_inputs.to('cuda'), batch_targets.to('cuda')

                output = model(batch_inputs, seq_lengths[0])
                gt_likelihood = model.compute_sequence_likelihood(batch_inputs, batch_targets,use_target_vocab_only=True)
                subject_likelihood += gt_likelihood.item()

                # Convert tensor of token indices to actual sequences of words
                predicted_indices = torch.argmax(output, dim=-1)
                original_sequences = [inv_vocab[idx] for idx in batch_inputs[0].tolist()]
                human_sequences = [inv_vocab[idx] for idx in batch_targets[0].tolist()]
                predicted_sequences = [inv_vocab[idx] if idx < len(inv_vocab) else 'Unknown' for idx in predicted_indices[0].tolist()]
                results['original_sequences'].append(original_sequences)
                results['predicted_sequences'].append(predicted_sequences)
                results['human_sequences'].append(human_sequences)
        subject_likelihood /= len(test_loader)
        subject_likelihoods.append(subject_likelihood)
        os.unlink(csv_test)

        output_log = os.path.join(subject_dir, "evaluation")
        os.makedirs(output_log, exist_ok=True)
        with open(os.path.join(output_log, "evaluation_results.json"), 'w') as f:
                json.dump(results, f, indent=4)
        data = results

        # Plot recall vs serial position
        plt.figure(figsize=(10, 6))
        plt.xlabel('Serial Position')
        plt.ylabel('Serial Probability')
        plt.ylim((0.0,1.0))
        plt.title(f"Serial Position Curve: {subject}")
        if omit_first_k is not None:
            data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
            data['human_sequences'] = [x[omit_first_k:] for x in data['human_sequences']]
            data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

        recall_probabilities = calculate_recall_probability(data['original_sequences'], data['predicted_sequences'])
        human_recall_probs = calculate_recall_probability(data['original_sequences'], data['human_sequences'])
        
        if end_position is not None:
            positions = list(range(1, end_position))
            min_length = min(len(recall_probabilities), end_position - 1)
            plot_vals = recall_probabilities[:min_length]
            plt.plot(positions[:min_length], plot_vals[:min_length], marker='o', label="Model", alpha=0.6)
            plt.plot(positions[:min_length], human_recall_probs[:min_length], marker='o', label="Human", alpha=0.6)

        else:
            positions = list(range(1, len(recall_probabilities) + 1))
            plt.plot(positions, recall_probabilities, marker='o', label="Model")
            plt.plot(positions, human_recall_probs, marker='o', label="Human")

        spc_rmse = calculate_rmse(plot_vals, human_recall_probs)
        spc_vals.append(plot_vals)
            
        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(output_log,"recall_vs_position.png"), bbox_inches='tight')

        # Plot crp vs lag
        plt.figure(figsize=(10, 6))
        plt.xlabel('Lag')
        plt.ylabel('Conditional Recall Probability')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.title('Conditional Recall Probability vs. Lag')

        if omit_first_k is not None:
            data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
            data['human_sequences'] = [x[omit_first_k:] for x in data['human_sequences']]
            data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

        probabilities, lags = calculate_conditional_recall_probability(data['original_sequences'], data['predicted_sequences'])
        positive_lags, pos_idxs = [lag for lag in lags if lag > 0], [i for i, lag in enumerate(lags) if lag > 0]
        positive_probs = [probabilities[idx] for idx in pos_idxs]
        negative_lags, neg_idxs = [lag for lag in lags if lag < 0], [i for i, lag in enumerate(lags) if lag < 0]
        negative_probs = [probabilities[idx] for idx in neg_idxs]
        neg_crp_vals.append(negative_probs)
        pos_crp_vals.append(positive_probs)

        h_probabilities, h_lags = calculate_conditional_recall_probability(data['original_sequences'], data['human_sequences'])
        h_positive_lags, h_pos_idxs = [lag for lag in lags if lag > 0], [i for i, lag in enumerate(h_lags) if lag > 0]
        h_positive_probs = [h_probabilities[idx] for idx in pos_idxs]
        h_negative_lags, h_neg_idxs = [lag for lag in lags if lag < 0], [i for i, lag in enumerate(h_lags) if lag < 0]
        h_negative_probs = [h_probabilities[idx] for idx in neg_idxs]
        crp_rmse = calculate_rmse(probabilities, h_probabilities)

        random_color = (random.random(), random.random(), random.random())

        # Plotting negative lags
        plt.plot(negative_lags, negative_probs, marker='o', color="skyblue", label="Model", alpha=0.6)
        plt.plot(h_negative_lags, h_negative_probs, marker='o', color="gold", label="Human", alpha=0.6)

        # Plotting positive lags
        plt.plot(positive_lags, positive_probs, marker='o', color="skyblue")
        plt.plot(h_positive_lags, h_positive_probs, marker='o', color="gold")

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(output_log,"crp_vs_position.png"), bbox_inches='tight')

        # Plot probability of first recall vs serial position
        plt.figure(figsize=(10, 6))
        plt.xlabel('Serial Position')
        plt.ylabel('First Recall Probability')
        plt.ylim((0.0,1.0))
        plt.title('First Recall Probability vs. Serial Position')

        if omit_first_k is not None:
            data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
            data['human_sequences'] = [x[omit_first_k:] for x in data['human_sequences']]
            data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

        recall_probabilities = calculate_first_recall_probability(data['original_sequences'], data['predicted_sequences'])
        human_recall_probs = calculate_first_recall_probability(data['original_sequences'], data['human_sequences'])
        pfr_rmse = calculate_rmse(recall_probabilities, human_recall_probs)
        label = f"Subject {subject}"
        frp_vals.append(recall_probabilities)

        if end_position is not None:
            positions = list(range(1, end_position))
            min_length = min(len(recall_probabilities), end_position - 1)
            plt.plot(positions[:min_length], recall_probabilities[:min_length], marker='o', label="Model", alpha=0.6)
            plt.plot(positions[:min_length], human_recall_probs[:min_length], marker='o', label="Human", alpha=0.6)

        else:
            positions = list(range(1, len(recall_probabilities) + 1))
            plt.plot(positions, recall_probabilities, marker='o', label="Model Pred.")
            plt.plot(positions, human_recall_probs, marker='o', label="Human")

        plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
        plt.savefig(os.path.join(output_log,"prob_first_recall_vs_position.png"), bbox_inches='tight')

            # Plot crp vs semantic sim (if desired)
        semantic_rmse = 0
        if calc_semantic:

            # Plot crp vs semantic similarity
            plt.figure(figsize=(10, 6))
            plt.xlabel('Semantic Similarity')
            plt.ylabel('Conditional Recall Probability')
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)
            plt.axhline(0, color='black', linewidth=0.5)
            plt.title('Conditional Recall Probability vs. Semantic Similarity')


            if omit_first_k is not None:
                data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
                data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

            #probs, similarity_bins = calculate_conditional_recall_vs_similarity(data['original_sequences'], data['predicted_sequences'])
            import spacy

            # Load a spacy model with word vectors
            nlp = spacy.load('en_core_web_lg')
            probs, similarity_bins = calculate_conditional_recall_vs_similarity(data['original_sequences'], 
                                                                                data['predicted_sequences'],
                                                                                nlp)
            human_probs, human_similarity_bins = calculate_conditional_recall_vs_similarity(data['original_sequences'], 
                                                                                            data['human_sequences'],
                                                                                            nlp)
            
            # Extracting the semantic similarities and recall probabilities
            similarities = list(similarity_bins[:-1])
            similarities = list(similarity_bins)
            probabilities = list(probs)
            plt.plot(similarities, probabilities, marker='o', label="Model Pred.")

            human_similarity_bins = list(human_similarity_bins)
            human_probs = list(human_probs)
            plt.plot(human_similarity_bins, human_probs, marker='o', label="Human")

            plt.legend()
            plt.savefig(os.path.join(output_log,"crp_vs_semanticsim.png"))
            semantic_rmse = calculate_rmse(probabilities, human_probs)
    
        spc_rmses.append(spc_rmse)
        crp_rmses.append(crp_rmse)
        pfr_rmses.append(pfr_rmse)
        semantic_rmses.append(semantic_rmse)

    with open(os.path.join(root_dir, 'likelihood_log.json'), 'w') as f:
        likelihood_data = {'subject_ids': [int(sub) for sub in subjects], 'avg_likelihood': [float(likelihood) for likelihood in subject_likelihoods]}
        json.dump(likelihood_data, f)

    with open(os.path.join(root_dir, 'behavior_curve_rmses.json'), 'w') as f:
        behavior_curve_data = {
            'subject_ids': [int(sub) for sub in subjects], 
            'spc_rmses': [float(rmse) for rmse in spc_rmses],
            'crp_rmses': [float(rmse) for rmse in crp_rmses],
            'pfr_rmses': [float(rmse) for rmse in pfr_rmses],
            'semantic_rmses': [float(rmse) for rmse in semantic_rmses],
            'seq2seq_spcs': [[float(spc) for spc in spc_list] for spc_list in spc_vals],
            'seq2seq_frps': [[float(frp) for frp in frp_list] for frp_list in frp_vals],
            'seq2seq_neg_crps': [[float(crp) for crp in crp_list] for crp_list in neg_crp_vals],
            'seq2seq_pos_crps': [[float(crp) for crp in crp_list] for crp_list in pos_crp_vals],
        }
        json.dump(behavior_curve_data, f)

    fig = plt.figure(figsize = (15,7.5)) 
    ax = sns.histplot(subject_likelihoods, kde=True)
    ax.set(xlabel='Avg. Likelihood per Subject Trials', title="Seq2Seq Predicted List Likelihoods for Subject-Specific Trials")
    plt.savefig(os.path.join(root_dir, "likelihood_plot.png"))

    fig = plt.figure(figsize = (15,7.5)) 
    ax = sns.histplot(spc_rmses, kde=True)
    ax.set(xlabel='RMSE', title="RMSE between Serial Position Curves for Subject Data and Seq2Seq Model")
    plt.savefig(os.path.join(root_dir, "spc_rmse.png"))

    fig = plt.figure(figsize = (15,7.5)) 
    ax = sns.histplot(crp_rmses, kde=True)
    ax.set(xlabel='RMSE', title="RMSE between Conditional Recall Probability Curves for Subject Data and Seq2Seq Model")
    plt.savefig(os.path.join(root_dir, "crp_rmse.png"))

    fig = plt.figure(figsize = (15,7.5)) 
    ax = sns.histplot(pfr_rmses, kde=True)
    ax.set(xlabel='RMSE', title="RMSE between First Recall Probability Curves for Subject Data and Seq2Seq Model")
    plt.savefig(os.path.join(root_dir, "pfr_rmse.png"))

    fig = plt.figure(figsize = (15,7.5)) 
    ax = sns.histplot(pfr_rmses, kde=True)
    ax.set(xlabel='RMSE', title="RMSE between Semantic Similarity Curves for Subject Data and Seq2Seq Model")
    plt.savefig(os.path.join(root_dir, "semantic_rmse.png"))

    averaged_rmse = average_behavior_curves(spc_rmses, crp_rmses, pfr_rmses)
    fig = plt.figure(figsize = (15,7.5)) 
    ax = sns.histplot(averaged_rmse, kde=True)
    ax.set(xlabel='RMSE', title="RMSE between Behavior Curves for Subject Data and Seq2Seq Model")
    plt.savefig(os.path.join(root_dir, "averaged_rmse.png"))

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script for loading checkpoints.")
    
    parser.add_argument('--root_dir', type=str, 
                        default=".",
                        help='Directory of individual subject fits.')
    parser.add_argument('--calculate_semantic', action="store_true",
                        default=False,
                        help="Flag to calculate semantic similarity behavior.")
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    evaluate_subject_fits(args.root_dir, calc_semantic=args.calculate_semantic)

if __name__ == "__main__":
    main()






