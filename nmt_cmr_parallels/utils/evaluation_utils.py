import os
import numpy as np
from tqdm import tqdm
from gensim.models import KeyedVectors
from collections import defaultdict
DEFAULT_DATA_PATH = os.path.expanduser(os.path.join('~', '.seq_nlp_data'))

def compute_similarity(word1, word2, glove_model):
    if word1 in glove_model and word2 in glove_model:
        return glove_model.similarity(word1, word2)
    return 0

def calculate_conditional_recall_vs_similarity(original_sequences, predicted_sequences, nlp, 
                                               data_path=DEFAULT_DATA_PATH):

    """
    Calculate the conditional recall probability based on the semantic similarity consecutively recalled items.

    Parameters:
    - original_sequences (list of lists): The original sequences of words.
    - predicted_sequences (list of lists): The sequences of words predicted/recalled by the model/user.

    Returns:
    - probabilities: list of conditional recall probabilities
    - similarity_bins: list of binned similarity values
    """

    similarities = list(np.arange(0.0, 1.0, step=0.05))
    total_counts = [1 for _ in similarities]
    recalled_counts = [0 for _ in similarities]

    # Precompute vectors for unique words in original sequences
    unique_words = {word: nlp(word.lower()) for seq in original_sequences for word in seq}
    similarity_bins = {sim: i for i, sim in enumerate(similarities)}

    for original, predicted in zip(original_sequences, predicted_sequences):
        similarity_pairs = {}
        for orig_word in original:
            for orig_word2 in original[1:]:
                pair_key = tuple(sorted([orig_word, orig_word2]))
                if pair_key not in similarity_pairs:
                    similarity = unique_words[orig_word].similarity(unique_words[orig_word2])
                    similarity_pairs[pair_key] = similarity
                    # Find the bin for the similarity
                    bin_index = min(similarity_bins, key=lambda x: abs(similarity - x))
                    total_counts[similarity_bins[bin_index]] += 1

        unique_predicted = []
        _ = [unique_predicted.append(x) for x in predicted if x not in unique_predicted]

        for i in range(len(unique_predicted) - 1):
            pred_pair = tuple(sorted([unique_predicted[i], unique_predicted[i + 1]]))
            if pred_pair in similarity_pairs:
                similarity = similarity_pairs[pred_pair]
                # Find the bin for the similarity
                bin_index = min(similarity_bins, key=lambda x: abs(similarity - x))
                recalled_counts[similarity_bins[bin_index]] += 1

    probs = [r / t if t != 0 else 0 for r, t in zip(recalled_counts, total_counts)]
    return probs, similarities

def calculate_recall_probability(original_sequences, predicted_sequences, omit_repeats=True):
    """
    Calculate the free recall probability for each position in a sequence.

    Parameters:
    - original_sequences (list of lists): The original sequences of words.
    - predicted_sequences (list of lists): The sequences of words predicted/recalled by the model/user.

    Returns:
    - probabilities -  list of probabilities for recall vs. serial position
    """
    
    position_counts = [0] * len(original_sequences[0])
    correct_recalls = [0] * len(original_sequences[0])

    for orig_seq, pred_seq in zip(original_sequences, predicted_sequences):
        if omit_repeats:
            pred_seq = [x for i, x in enumerate(pred_seq) if pred_seq.index(x) == i]
        for idx, word in enumerate(orig_seq):
            if word in pred_seq:
                correct_recalls[idx] += 1
            position_counts[idx] += 1
    
    probabilities = [correct / total if total != 0 else 0 for correct, total in zip(correct_recalls, position_counts)]
    return probabilities

def calculate_first_recall_probability(original_sequences, predicted_sequences,omit_repeats=True):
    """
    Calculate the probability of first recall across each sequence.

    Parameters:
    - original_sequences (list of lists): The original sequences of words.
    - predicted_sequences (list of lists): The sequences of words predicted/recalled by the model/user.

    Returns:
    - probabilities -  list of probabilities for first recall position
    """
    
    first_recalls = [0] * len(original_sequences[0])

    for orig_seq, pred_seq in zip(original_sequences, predicted_sequences):
        if omit_repeats:
            pred_seq = [x for i, x in enumerate(pred_seq) if pred_seq.index(x) == i]
        for word in pred_seq:
            if word in orig_seq:
                orig_position = orig_seq.index(word)
                first_recalls[orig_position] += 1
                break
    
    total_recalls = sum(first_recalls)
    probabilities = [count / total_recalls if total_recalls != 0 else 0 for count in first_recalls]
    return probabilities

def calculate_conditional_recall_probability(original_sequences, predicted_sequences, omit_repeats=True):

    """
    Calculate the conditional recall probability based on the lag between consecutively recalled items.

    Parameters:
    - original_sequences (list of lists): The original sequences of words.
    - predicted_sequences (list of lists): The sequences of words predicted/recalled by the model/user.

    Returns:
    - probabilities: list of conditional recall probabilities
    - lags: list of lag values
    """

    lags = [n for n in range(-len(original_sequences[0])+1,len(original_sequences[0]))]
    lag_counts = [0] * len(lags)

    for orig_seq, pred_seq in zip(original_sequences, predicted_sequences):
        if omit_repeats:
            pred_seq = [x for i, x in enumerate(pred_seq) if pred_seq.index(x) == i]

        for idx, current_word in enumerate(pred_seq):
            if current_word not in orig_seq:
                continue

            original_idx_current = orig_seq.index(current_word)

            # Find the next word in pred_seq that is also in orig_seq
            next_correct_idx = None
            for next_word in pred_seq[idx + 1:]:
                if next_word in orig_seq and next_word != current_word:
                    next_correct_idx = orig_seq.index(next_word)
                    break

            # Calculate lag and add to counter
            if next_correct_idx is not None:
                lag = next_correct_idx - original_idx_current
                if -len(orig_seq) < lag < len(orig_seq):
                    lag_counts[lags.index(lag)] += 1
            
    total_recall = sum(lag_counts)
                
    # Calculate conditional probabilities
    probabilities = [count / total_recall if total_recall != 0 else 0 for count in lag_counts]
    return probabilities, lags

def calculate_semantic_intrusion_frequency(original_sequences, predicted_sequences, nlp, data_path=DEFAULT_DATA_PATH):

    """
    Calculate the frequency of highly semantically similar intrusions in the predicted sequences.

    Parameters:
    - original_sequences (list of lists): The original sequences of words.
    - predicted_sequences (list of lists): The sequences of words predicted/recalled by the model/user.

    Returns:
    - max_similarities: list of the maximum semantic similarities found in the ground-truth sequence for each incorrect
                        predicted word
    """
    max_similarities = []
    for original, predicted in tqdm(zip(original_sequences, predicted_sequences), desc="Evaluating semantic intrustions..."):

        orig_vectors = [nlp(x.lower()) for x in original]
        words_visited = set()
        for word in predicted:
            if word in words_visited:
                continue
            words_visited.add(word)
            if word not in original:
                predicted_word = nlp(word.lower())
                max_similarities.append(max([predicted_word.similarity(original_word) for original_word in orig_vectors]))
            
    return max_similarities