import torch
import math
import logging

import torch.nn as nn
import torch.nn.functional as F
from nmt_cmr_parallels.data.word_vectors import load_pretrained_inverse_embedding

class InverseEmbeddingMLP(nn.Module):
    def __init__(self, embedding_dim, vocab_size, device='cpu'):
        super(InverseEmbeddingMLP, self).__init__()
        self.device = device
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, vocab_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class EpisodicBuffer(nn.Module):

    def __init__(self, embedding_layer, 
                 memory_size=1000, 
                 hidden_size=256,
                 seq_tokens=False, 
                 pretrained_inverse_embedding=True,
                 retrieval_temperature=1e-5,
                 retrieval_discount=0.25, 
                 device='cpu'):

        super(EpisodicBuffer, self).__init__()
        self.memory_size = memory_size
        self.hidden_size = hidden_size
        self.retrieval_temperature = retrieval_temperature
        self.retrieval_discount = retrieval_discount
        self.device = device

        self.contexts = nn.Parameter(torch.zeros(memory_size, hidden_size), requires_grad=True)
        self.embeddeds = nn.Parameter(torch.zeros(memory_size, embedding_layer.weight.shape[1]), requires_grad=True)
        self.inverse_embedding = InverseEmbeddingMLP(embedding_layer.weight.shape[1], embedding_layer.weight.shape[0]).to(self.device)
        if pretrained_inverse_embedding:
            self.inverse_embedding = load_pretrained_inverse_embedding(self.inverse_embedding, seq_tokens=seq_tokens)
            for param in self.inverse_embedding.parameters():
                param.requires_grad = False
        self.write_head = 0

    def write(self, encode_context, input_embedding):

        batch_size = input_embedding.size(0)
        seq_len = encode_context.size(1)
        for i in range(batch_size):
            for j in range(seq_len):

                with torch.no_grad():
                    self.contexts[self.write_head] = encode_context[i,j].detach().clone()
                    self.embeddeds[self.write_head] = input_embedding[i,j].detach().clone()
                    self.write_head = (self.write_head + 1) % self.memory_size

    def read(self, query_context, return_embedding=False):

        batch_size, seq_len = query_context.size(0), query_context.size(1)

        query_context = query_context.contiguous().view(-1,self.hidden_size)
        similarities = F.cosine_similarity(query_context.unsqueeze(1), self.contexts.clone().unsqueeze(0), dim=2)
        similarities = similarities.view(batch_size, seq_len, self.memory_size)
        mem_weights = F.softmax(similarities/0.01, dim=2)
        retrieved_memories = torch.matmul(mem_weights, self.embeddeds.clone())
        decoded_memories = self.inverse_embedding(retrieved_memories)

        if return_embedding:
            return decoded_memories, retrieved_memories
        return decoded_memories

    def poll(self, inv_vocab):
        """
        Print the contents of the memory table.
        
        Args:
            inv_vocab: Dict = mapping from indices to words in vocabulary
        """
        
        for i in range(self.memory_size):
            context = self.contexts[i].clone()
            if not torch.all(context == 0):
                embedded = self.embeddeds[i].clone()
                
                decoded_memory = self.inverse_embedding(embedded)
                predicted_indx = torch.argmax(decoded_memory, dim=0).item()
                
                print(f"Memory Slot {i}:")
                print(f" Encoded Word: {inv_vocab[predicted_indx] }", flush=True)
                print()
    
    def reset(self):

        nn.init.zeros_(self.contexts)
        nn.init.zeros_(self.embeddeds)
        self.retrieval_mask = torch.ones(self.memory_size).to(self.device)
        self.write_head = 0



