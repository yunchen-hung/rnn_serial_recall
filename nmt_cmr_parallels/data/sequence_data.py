import random
import nltk
import torch
import os
import logging
import pickle
import json
import ast

import pandas as pd
from tqdm import tqdm

from nltk.corpus import wordnet
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from gensim.models import KeyedVectors
from torch.utils.data import Dataset, DataLoader, random_split
from nmt_cmr_parallels.data.read_peers_data import load_cached_vocabulary
from nmt_cmr_parallels.data.data_augmentation import augment_data_by_substitution

word_list = nltk.corpus.words.words()
tagged_words = nltk.pos_tag(word_list)
non_proper_nouns = [word for word, pos in tagged_words if pos != 'NNP' and pos != 'NNPS']

class SequenceDataset(Dataset):
    def __init__(self, sequences, vocab, multi_sequence=False, num_sequences=2):
        self.sequences = sequences
        self.vocab=vocab
        self.multi_sequence = multi_sequence
        self.num_sequences = num_sequences 

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        if self.multi_sequence:
            
            # Randomly select additional unique indices besides the current one
            all_indices = list(range(len(self.sequences)))
            all_indices.remove(index)  # Ensure we do not pick the same sequence
            additional_indices = random.sample(all_indices, (self.num_sequences-1))

            multi_sequences = [self.sequences[index]] + [self.sequences[i] for i in additional_indices]

            return multi_sequences

        return self.sequences[index]
    
def save_dataset(dataset, file_path):
    with open(file_path, 'wb') as f:
        pickle.dump(dataset, f)

def load_dataset(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def create_pretrained_semantic_embedding(data_dir, embedding_dim):

    glove_dir = os.path.join(data_dir,"glove")
    for txt_f in [os.path.join(glove_dir,x) for x in os.listdir(glove_dir)]:
        if str(embedding_dim) in txt_f:
            model = KeyedVectors.load_word2vec_format(txt_f, binary=False)
            return model

def get_related_words(word, num_words=3):
    related_words = []
    
    for synset in wordnet.synsets(word):
        related_words.extend(synset.lemma_names())

    # Remove duplicates
    related_words = list(set(related_words) - {word})

    # Check if there are enough related words to sample from
    if len(related_words) < num_words:
        return related_words

    return random.sample(related_words, num_words)

def generate_sequence(length=10, related_group_size=2, vocab=None, include_seq_tokens=False):
    
    if vocab is None:
        vocab = non_proper_nouns

    sequence = []

    # Reduce sequence length by 2 to accomodate adding sequence tokens
    if include_seq_tokens:
        length -= 2

    while len(sequence) < length:

        random_word = random.choice(vocab)
        related_words = get_related_words(random_word, related_group_size)
        if not all([word in vocab for word in related_words]):
            continue

        sequence.append(random_word)
        sequence += related_words

    random.shuffle(sequence)
    sequence = sequence[:length]

    if include_seq_tokens:
        sequence = ['<SoS>'] + sequence + ['<EoS>']

    return sequence

def create_new_vocab(sequences, vocab_size=500):

    # Get all words in generated sequences
    new_vocab = set()
    for seq in sequences:
        for word in seq:
            new_vocab.add(word)

    # Extend vocab to desired size with random words
    while len(new_vocab) < vocab_size:
        new_vocab.add(random.sample(non_proper_nouns, 1)[0])

    return new_vocab

def tokenize_sequences(sequences, word_to_index):

    tokenized_sequences = []
    for seq in tqdm(sequences, desc="Tokenizing input sequences..."):
        tokenized_sequences.append([word_to_index[word] for word in seq if word in word_to_index])
    return tokenized_sequences

def collate_fn(data):

    try:

        sequences, targets = zip(*data)

    except:

        batch_sequences_padded = []
        batch_seq_lengths = []
        batch_targets_padded = []
        
        for item in data:
            src_seqs, target_seqs = [],[]
            for seq, target in item:
                src_seqs.append(seq)
                target_seqs.append(target)

            seq_tensors = [torch.tensor(seq, dtype=torch.long) for seq in src_seqs]
            target_tensors = [torch.tensor(target, dtype=torch.long) for target in target_seqs]


            sequences_padded = torch.nn.utils.rnn.pad_sequence(seq_tensors, batch_first=True, padding_value=0)  # Adjust padding_value if needed
            targets_padded = torch.nn.utils.rnn.pad_sequence(target_tensors, batch_first=True, padding_value=0)  # Adjust padding_value if needed
            

            batch_sequences_padded.append(sequences_padded)
            batch_seq_lengths.append([len(seq) for seq in src_seqs])
            batch_targets_padded.append(targets_padded)
        
        return batch_sequences_padded, batch_targets_padded, batch_seq_lengths

    seq_lengths = [len(seq) for seq in sequences]
    sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True)
    targets_padded = torch.nn.utils.rnn.pad_sequence(targets, batch_first=True)

    return sequences_padded, targets_padded, seq_lengths

def create_sequence_dataset(sequence_length=10,
                          num_related_words=3,
                          num_sequences=1000,
                          vocab=None,
                          vocab_size=500,
                          vocab_source=None,
                          human_set_file=None,
                          use_seq_tokens=False, 
                          multi_sequence=False,
                          num_multi_seq=3,**kwargs):

    vocab = list(vocab) if vocab is not None else None
    if vocab_source is not None:
        vocab = load_cached_vocabulary(vocab_source)
    else:
        if vocab is None:
            vocab = create_new_vocab(sequences, vocab_size=vocab_size)

    # Add start-, end-, and null sequence tokens if desired
    if use_seq_tokens and '<SoS>' not in vocab:
        vocab = ['<null>'] + vocab + ['<SoS>','<EoS>']
    # If sequence tokens were already present in vocab, assume we should be using them
    elif '<SoS>' in vocab:
        use_seq_tokens = True    

    if human_set_file is not None:
        with open(human_set_file,'r') as f:
            human_data = json.load(f)
        sequences = human_data['original_sequences']
    else:
        sequences = [generate_sequence(length=sequence_length, related_group_size=num_related_words, 
                                       vocab=vocab, include_seq_tokens=use_seq_tokens) for _ in tqdm(range(num_sequences), desc="Generating sequences")]
        
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    sequences = tokenize_sequences(sequences, word_to_index)

    inputs = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    targets = [torch.tensor(seq, dtype=torch.long) for seq in sequences]
    dataset = SequenceDataset(list(zip(inputs, targets)), vocab, multi_sequence=multi_sequence, num_sequences=num_multi_seq)
    return dataset, vocab

def create_dataset_from_csv(csv_training, csv_validation,
                            vocab=None,
                            vocab_source=None,
                            batch_size=16,
                            use_seq_tokens=False,
                            multi_sequence=False,
                            num_multi_seq=False, 
                            augment_data=False, 
                            data_multiplier=2,**kwargs):
    
    """
    Create a SequenceDataset (and dataloaders) from CSV files containing training and validation targets
    """

    # Load data samples as Pandas Dataframes
    train_data = pd.read_csv(csv_training)
    val_data = pd.read_csv(csv_validation)

    # Extract presentations and targets from Dataframes
    train_pres = [ast.literal_eval(x) for x in train_data['Trial Presented Words']]
    train_target = [ast.literal_eval(x) for x in train_data['Trial Recall Words']]
    val_pres = [ast.literal_eval(x) for x in val_data['Trial Presented Words']]
    val_target = [ast.literal_eval(x) for x in val_data['Trial Recall Words']]

    vocab = list(vocab) if vocab is not None else None
    if vocab_source is not None:
        vocab = load_cached_vocabulary(vocab_source)

    if augment_data:
        train_pres, train_target = augment_data_by_substitution(train_pres, train_target, vocab, num_augmented_samples=data_multiplier)
    
    # Add start-, end-, and null sequence tokens if desired
    if use_seq_tokens and '<SoS>' not in vocab:
        vocab = ['<null>'] + vocab + ['<SoS>','<EoS>']
    # If sequence tokens were already present in vocab, assume we should be using them
    elif '<SoS>' in vocab:
        use_seq_tokens = True

    # Tokenize word sequences
    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    train_pres = tokenize_sequences(train_pres, word_to_index)
    train_target = tokenize_sequences(train_target, word_to_index)
    val_pres = tokenize_sequences(val_pres, word_to_index)
    val_target = tokenize_sequences(val_target, word_to_index)

    # Convert sequences to Torch tensors
    train_inputs = [torch.tensor(seq, dtype=torch.long) for seq in train_pres]
    train_targets = [torch.tensor(seq, dtype=torch.long) for seq in train_target]
    val_inputs = [torch.tensor(seq, dtype=torch.long) for seq in val_pres]
    val_targets = [torch.tensor(seq, dtype=torch.long) for seq in val_target]

    # Convert sequences to TensorDataset
    train_dataset = SequenceDataset(list(zip(train_inputs, train_targets)), vocab, 
                              multi_sequence=multi_sequence, 
                              num_sequences=num_multi_seq)
    val_dataset = SequenceDataset(list(zip(val_inputs, val_targets)), vocab, 
                              multi_sequence=multi_sequence, 
                              num_sequences=num_multi_seq)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                            shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, vocab

def create_test_dataset_from_csv(csv_test,
                                vocab=None,
                                vocab_source=None,
                                batch_size=1,
                                use_seq_tokens=False,
                                multi_sequence=False,
                                num_multi_seq=False, **kwargs):
    
    """
    Create a SequenceDataset from parsed CSV files
    """

    # Load data samples as Pandas Dataframes
    test_data = pd.read_csv(csv_test)
    test_pres = [ast.literal_eval(x) for x in test_data['Trial Presented Words']]
    test_target = [ast.literal_eval(x) for x in test_data['Trial Recall Words']]

    vocab = list(vocab) if vocab is not None else None
    if vocab_source is not None:
        vocab = load_cached_vocabulary(vocab_source)
    
    # Add start-, end-, and null sequence tokens if desired
    if use_seq_tokens and '<SoS>' not in vocab:
        vocab = ['<null>'] + vocab + ['<SoS>','<EoS>']
    # If sequence tokens were already present in vocab, assume we should be using them
    elif '<SoS>' in vocab:
        use_seq_tokens = True

    word_to_index = {word: idx for idx, word in enumerate(vocab)}
    test_pres = tokenize_sequences(test_pres, word_to_index)
    test_target = tokenize_sequences(test_target, word_to_index)

    test_inputs = [torch.tensor(seq, dtype=torch.long) for seq in test_pres]
    test_targets = [torch.tensor(seq, dtype=torch.long) for seq in test_target]
    test_dataset = SequenceDataset(list(zip(test_inputs, test_targets)), vocab, 
                              multi_sequence=multi_sequence, 
                              num_sequences=num_multi_seq)
    
    return test_dataset, vocab

def create_dataloaders(dataset,
                    batch_size=32,
                    split_ratio=0.9):

    # Assuming dataset is your entire dataset
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    return train_loader, val_loader
