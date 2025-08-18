import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import logging
import json
import argparse
import random
import os

from nmt_cmr_parallels.utils.evaluation_utils import (calculate_conditional_recall_probability, 
                                                  calculate_recall_probability,
                                                  calculate_first_recall_probability,
                                                  calculate_semantic_intrusion_frequency, 
                                                  calculate_conditional_recall_vs_similarity)

def generate_recall_plots(results_files,
                          output_log="recall_plots",
                          end_position=None, 
                          semantic_sim=False, 
                          semantic_intrusions=False,
                          omit_first_k=None, **kwargs):
    
    os.makedirs(output_log, exist_ok=True)

    # Plot recall vs serial position
    plt.figure(figsize=(10, 6))
    plt.xlabel('Serial Position')
    plt.ylabel('Serial Probability')
    plt.ylim((0.0,1.0))
    plt.title('Recall Probability vs. Serial Position')
    for file in results_files:
        with open(file,'r') as f:
            data = json.load(f)

        if omit_first_k is not None:
            data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
            data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

        recall_probabilities = calculate_recall_probability(data['original_sequences'], data['predicted_sequences'])
        label = os.path.splitext(os.path.basename(file))[0]

        if end_position is not None:
            positions = list(range(1, end_position))
            min_length = min(len(recall_probabilities), end_position - 1)
            plt.plot(positions[:min_length], recall_probabilities[:min_length], marker='o', label=label, alpha=0.6)

        else:
            positions = list(range(1, len(recall_probabilities) + 1))
            plt.plot(positions, recall_probabilities, marker='o', label=label)
        
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(output_log,"recall_vs_position.png"), bbox_inches='tight')

    # Plot crp vs lag
    plt.figure(figsize=(10, 6))
    plt.xlabel('Lag')
    plt.ylabel('Conditional Recall Probability')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.title('Conditional Recall Probability vs. Lag')
    for file in results_files:
        with open(file,'r') as f:
            data = json.load(f)

        if omit_first_k is not None:
            data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
            data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

        probabilities, lags = calculate_conditional_recall_probability(data['original_sequences'], data['predicted_sequences'])
        positive_lags, pos_idxs = [lag for lag in lags if lag > 0], [i for i, lag in enumerate(lags) if lag > 0]
        positive_probs = [probabilities[idx] for idx in pos_idxs]
        negative_lags, neg_idxs = [lag for lag in lags if lag < 0], [i for i, lag in enumerate(lags) if lag < 0]
        negative_probs = [probabilities[idx] for idx in neg_idxs]

        label = os.path.splitext(os.path.basename(file))[0]
        random_color = (random.random(), random.random(), random.random())

        # Plotting negative lags
        plt.plot(negative_lags, negative_probs, marker='o', color=random_color, label=label, alpha=0.6)

        # Plotting positive lags
        plt.plot(positive_lags, positive_probs, marker='o', color=random_color)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(output_log,"crp_vs_position.png"), bbox_inches='tight')

    # Plot probability of first recall vs serial position
    plt.figure(figsize=(10, 6))
    plt.xlabel('Serial Position')
    plt.ylabel('First Recall Probability')
    plt.ylim((0.0,1.0))
    plt.title('First Recall Probability vs. Serial Position')
    for file in results_files:
        with open(file,'r') as f:
            data = json.load(f)

        if omit_first_k is not None:
            data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
            data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

        recall_probabilities = calculate_first_recall_probability(data['original_sequences'], data['predicted_sequences'])
        label = os.path.splitext(os.path.basename(file))[0]

        if end_position is not None:
            positions = list(range(1, end_position))
            min_length = min(len(recall_probabilities), end_position - 1)
            plt.plot(positions[:min_length], recall_probabilities[:min_length], marker='o', label=label, alpha=0.6)

        else:
            positions = list(range(1, len(recall_probabilities) + 1))
            plt.plot(positions, recall_probabilities, marker='o', label=label)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.savefig(os.path.join(output_log,"prob_first_recall_vs_position.png"), bbox_inches='tight')

    # Plot crp vs semantic sim (if desired)
    if semantic_sim:

        # Plot crp vs semantic similarity
        plt.figure(figsize=(10, 6))
        plt.xlabel('Semantic Similarity')
        plt.ylabel('Conditional Recall Probability')
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.axhline(0, color='black', linewidth=0.5)
        plt.title('Conditional Recall Probability vs. Semantic Similarity')
        for file in results_files:
            with open(file,'r') as f:
                data = json.load(f)

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

            # Extracting the semantic similarities and recall probabilities
            #similarities = list(similarity_bins[:-1])
            similarities = list(similarity_bins)
            probabilities = list(probs)
            label = os.path.splitext(os.path.basename(file))[0]
            plt.plot(similarities, probabilities, marker='o', label=label)
        plt.legend()
        plt.savefig(os.path.join(output_log,"crp_vs_semanticsim.png"))
    
    if semantic_intrusions:

        # Plot semantic intrusion histogram
        plt.figure(figsize=(10, 6))
        plt.title('Frequency of Semantically Related Intrusions')
        plt.xlabel('Semantic Similarity')
        plt.ylabel('Frequency')
        plt.xlim(0.0, 1.0)
        for file in results_files:
            with open(file,'r') as f:
                data = json.load(f)

            if omit_first_k is not None:
                data['original_sequences'] = [x[omit_first_k:] for x in data['original_sequences']]
                data['predicted_sequences'] = [x[omit_first_k:] for x in data['predicted_sequences']]

            import spacy

            # Load a spacy model with word vectors
            nlp = spacy.load('en_core_web_lg')
            max_similarities = calculate_semantic_intrusion_frequency(data['original_sequences'], 
                                                                    data['predicted_sequences'],
                                                                    nlp)

            # Extracting the semantic similarities and recall probabilities
            #similarities = list(similarity_bins[:-1])
            label = os.path.splitext(os.path.basename(file))[0]
            plt.hist(max_similarities, bins=20, edgecolor='black', range=(0.0, 1.0), label=label, alpha=0.5)
            
        plt.legend()
        plt.savefig(os.path.join(output_log,"semantic_intrusion_freq.png"))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train a recurrent network for sequence recall.")
    
    parser.add_argument("--results_files", nargs='+', default=None, help="JSON files containing sequence data.")

    parser.add_argument("--results_dir", default=None, help="Directory containing evaluation files (.json).")
    
    parser.add_argument("--output_log", type=str, 
                        default='recall_plots', 
                        help="Directory for plots.")
    
    parser.add_argument("--end_position", type=int, 
                        default=None, 
                        help="Sequence position to end graph plotting on.")
    
    parser.add_argument("--omit_first_k", type=int, 
                        default=None, 
                        help="Indicate how many sequence elements to omit when calculating metrics.")
    
    parser.add_argument("--semantic_sim", default=False,action="store_true", 
                        help="Enable calculation of semantic similarities.")
    
    parser.add_argument("--semantic_intrusions", default=False,action="store_true", 
                        help="Enable calculation of semantic intrusions.")

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

    if args.results_dir is not None:
        results_files = [os.path.join(args.results_dir, x) for x in os.listdir(args.results_dir) if x.endswith('.json')]
        args_dict['results_files'] = results_files

    if args_dict['results_files'] is None:
        print("You must provide evaluation json files.")
        exit()

    generate_recall_plots(**args_dict)

