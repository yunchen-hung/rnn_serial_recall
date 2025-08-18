import torch

import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class CriticNetwork(nn.Module):
    def __init__(self, embedding_layer, hidden_dim, embedding_dim, critic_hidden_dim):
        super(CriticNetwork, self).__init__()
        self.embedding = embedding_layer
        self.gru = nn.GRU(input_size=hidden_dim + 2*embedding_dim, hidden_size=critic_hidden_dim, batch_first=True)
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(critic_hidden_dim, critic_hidden_dim // 2)
        self.fc2 = nn.Linear(critic_hidden_dim // 2, 1)

    def forward(self, encoder_outputs, predicted_sequence, source_sequence):

        embedded_source = self.embedding(source_sequence)
        embedded_seq = self.embedding(predicted_sequence) 
        combined_input = torch.cat((encoder_outputs, embedded_seq, embedded_source), dim=2)
        output, _ = self.gru(combined_input) 
        pooled_output = self.pooling(output.transpose(1, 2)).squeeze()
        x = torch.relu(self.fc1(pooled_output))
        value = self.fc2(x)

        return value


class Agent(nn.Module):
    def __init__(self, envs, sequence_model, sequence_length, embedding_dim, model_type='encoderdecoder', device='cpu'):
        super().__init__()
        
        self.actor_type = model_type
        self.actor = sequence_model
        self.embedding = self.actor.embedding
        self.critic = CriticNetwork(self.embedding, self.actor.hidden_dim, self.actor.embedding_dim, 50)
        self.encoder_outputs = None
        self.original_sequence = None
        self.hidden_state = None
        self.device = device
   
    def get_value(self, encoder_outputs, x, source_sequence):

        return self.critic(encoder_outputs, x, torch.tensor(source_sequence).to(dtype=torch.int64).to(self.actor.device))

    def get_action_and_value(self, x, action=None, encode=False, prev_hidden_state = None, prev_encoder_outputs=None, prev_sequences=None, return_states=False):

        if self.hidden_state is None:
            self.hidden_state = self.actor.init_hidden(x.size(0))

        if self.actor_type ==  'encoderdecoder':

            if encode:
                self.original_sequence = x.to(dtype=torch.int64)
                self.encoder_outputs, self.hidden_state = self.actor.encode(x.to(dtype=torch.int64))

            if prev_encoder_outputs is None:
                curr_encoded = self.encoder_outputs
            else:
                curr_encoded = prev_encoder_outputs

            if prev_hidden_state is None:
                curr_hidden = self.hidden_state
            else:
                curr_hidden = prev_hidden_state

            logits, curr_encoded, curr_hidden = self.actor.decode(x.to(dtype=torch.int64), 1, curr_hidden, curr_encoded, return_states=True)

            if prev_encoder_outputs is None and prev_hidden_state is None:
                self.encoder_outputs = curr_encoded
                self.hidden_state = curr_hidden
        else:
            logits = self.actor(x.to(dtype=torch.int64))

        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        log_prob = probs.log_prob(action)
        
        if prev_sequences is None:
            source_sequences = self.original_sequence
        else:
            source_sequences = prev_sequences

        if return_states:
            return action, log_prob, probs.entropy(), self.get_value(curr_encoded, x.to(dtype=torch.int64), source_sequences), curr_hidden, curr_encoded
        else:
            return action, log_prob, probs.entropy(), self.get_value(curr_encoded, x.to(dtype=torch.int64), source_sequences)