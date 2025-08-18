import random
from nltk.corpus import wordnet

def augment_data_by_substitution(pres_sequences, target_sequences, vocab, substitution_rate=0.75, num_augmented_samples=2):
    
    augmented_pres_sequences = []
    augmented_target_sequences = []
    for pres_seq, target_seq in zip(pres_sequences, target_sequences):
        for _ in range(num_augmented_samples):
            augmented_pres_seq = []
            augmented_target_seq = []
            substitution_map = {}
            
            # Apply word substitution to the presentation sequence
            for pres_word in pres_seq:
                if pres_word in vocab and random.random() < substitution_rate:
                    # Find synonyms of the presentation word
                    synonyms = []
                    for synset in wordnet.synsets(pres_word.lower()):
                        for lemma in synset.lemmas():
                            synonym = lemma.name().upper()
                            if synonym != pres_word and synonym in vocab:
                                synonyms.append(synonym)
                    
                    if synonyms:
                        # Randomly select a synonym from the available options
                        synonym = random.choice(synonyms)
                        augmented_pres_seq.append(synonym)
                        substitution_map[pres_word] = synonym
                    else:
                        augmented_pres_seq.append(pres_word)
                else:
                    augmented_pres_seq.append(pres_word)
            
            # Replace any substituted words in the target sequence
            for target_word in target_seq:
                if target_word in substitution_map:
                    augmented_target_seq.append(substitution_map[target_word])
                else:
                    augmented_target_seq.append(target_word)
            
            augmented_pres_sequences.append(augmented_pres_seq)
            augmented_target_sequences.append(augmented_target_seq)
    
    return augmented_pres_sequences, augmented_target_sequences