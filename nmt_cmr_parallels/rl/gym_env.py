import gymnasium as gym
from gymnasium import spaces
from nmt_cmr_parallels.data.read_peers_data import load_cached_vocabulary
import numpy as np
import random
import torch

class SequenceEnv(gym.Env):
    def __init__(self, vocab=None, 
                 vocab_source=None, 
                 sequence_length=16, 
                 seq_tokens=False,
                 timeout=300):
        
        super(SequenceEnv, self).__init__()

        if vocab_source is not None:
            vocab = load_cached_vocabulary(vocab_source)
        self.vocab = vocab
        self.sequence_length = sequence_length
        self.start_new_seq = False
        self.seq_tokens = seq_tokens
        self.timeout = timeout

        self.action_space = spaces.Discrete(len(self.vocab))
        if self.seq_tokens:
            self.vocab = ['<null>'] + self.vocab + ['<SoS>','<EoS>']
            self.observation_space = spaces.MultiDiscrete([len(self.vocab) for _ in range(self.sequence_length+2)])
        else:
            self.observation_space = spaces.MultiDiscrete([len(self.vocab) for _ in range(self.sequence_length)])

        self.current_predicted_sequence = []

    def reset(self, **kwargs):
        self.episode_steps, self.episode_reward = 0,0
        self.current_predicted_sequence = []
        self.observation, self.word_sequence = self.generate_observation()
        return self.observation, {"word_seq": set(self.word_sequence), "episode":{"r":0,"l":0}}

    def generate_observation(self):
        if self.seq_tokens:
            self.current_sequence = random.sample(self.vocab[:-2], self.sequence_length)
            self.current_sequence = ['<SoS>'] + self.current_sequence + ['<EoS>']
        else:
            self.current_sequence = random.sample(self.vocab, self.sequence_length)

        # Tokenize word sequences
        word_to_index = {word: idx for idx, word in enumerate(self.vocab)}
        self.current_tokenized_sequence = np.array([word_to_index[word] for word in self.current_sequence],dtype=np.int32)

        return self.current_tokenized_sequence, self.current_sequence
    
    def seed(self, seed):
        pass

    def _adjust_sequence(self, sequence):

        # Padding if the sequence is shorter than the observation space length
        if len(sequence) < self.observation_space.shape[0]:
            padded_sequence = sequence + [0] * (self.observation_space.shape[0] - len(sequence))  # Assuming 0 is the padding token
        # Truncating from the beginning if the sequence is longer than observation space length
        elif len(sequence) > self.observation_space.shape[0]:
            padded_sequence = sequence[-self.observation_space.shape[0]:]
        else:
            padded_sequence = sequence

        return padded_sequence

    def step(self, action):

        if self.start_new_seq:
            self.current_predicted_sequence = []
            self.observation, self.word_sequence = self.generate_observation()
            self.start_new_seq = False
            return self.observation, 0, False, None, set(self.word_sequence)
        
        # Reward Structure
        if action < len(self.vocab) and self.vocab[action] == '<EoS>':

            # Receive negative penalty for every non-recalled word if ending the sequence prematurely
            orig_set = set(self.current_tokenized_sequence)
            predict_set = set(self.current_predicted_sequence)
            missing = orig_set.difference(predict_set)
            reward = -10 * len(missing)

        elif action in self.current_tokenized_sequence and action not in self.current_predicted_sequence:
            reward = 1.0
        elif action in self.current_tokenized_sequence and action in self.current_predicted_sequence:

            # Penalize agent proportionately to the number of times it repeatedly guesses the same word
            reward = -0.5 * self.current_predicted_sequence.count(action)
            
        elif action not in self.current_tokenized_sequence and action not in self.current_predicted_sequence:
            reward = -1.0
        else:
            reward = -1.0
        self.current_predicted_sequence.append(action)

        self.episode_reward += reward
        self.episode_steps += 1

        output_seq = np.array(self._adjust_sequence(self.current_predicted_sequence),dtype=np.int64)
    
        done = False
        if not self.seq_tokens:
            if len(self.current_predicted_sequence) >= len(self.current_sequence):
                done = True
        else:
            if action < len(self.vocab) and self.vocab[action] == '<EoS>':
                done = True
        if self.episode_steps > self.timeout:
            done = True

        return output_seq, reward, done, None, {"word_seq": set(self.word_sequence), "episode":{"r":self.episode_reward,"l":self.episode_steps}}

    def render(self, mode='human'):
        word_sequence = [self.vocab[i] for i in self.current_sequence]
        pred_word_sequence = [self.vocab[i] for i in self.current_predicted_sequence]
        print(f"Current Sequence: {word_sequence}")
        print(f"Current Predicted Sequence: {pred_word_sequence}")