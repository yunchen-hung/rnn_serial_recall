import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
import logging
from copy import deepcopy

from .memory_modules.attention import BahdanauAttention, LuongAttention
from .memory_modules.episodic_buffer import EpisodicBuffer

class Encoder(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, 
                 rnn_mode="LSTM",
                 num_layers=1, bidirectional=False, 
                 dropout=None,**kwargs):
        super(Encoder, self).__init__()
        self.rnn_mode = rnn_mode
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.dropout=None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        # Instantiate RNN model
        if rnn_mode == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim,
                                num_layers=num_layers,
                                bidirectional=bidirectional,
                                batch_first=True)
        elif rnn_mode == "GRU":
            self.rnn = nn.GRU(embedding_dim,
                              hidden_dim,
                              num_layers=num_layers,
                              batch_first=True)

    def forward(self, x, hidden):
        output, hidden = self.rnn(x, hidden)

        if self.dropout is not None:
            output = self.dropout(output)
        return output, hidden

class Decoder(nn.Module):
    def __init__(self, input_dim, 
                 hidden_dim, 
                 output_dim,
                 rnn_mode='LSTM',         
                 num_layers=1, 
                 bidirectional=False,
                 use_attention=True,
                 attention_type="luong",
                 embedding_layer=None,
                 dropout=None,
                 device='cpu'):
        
        super(Decoder, self).__init__()
        self.rnn_mode = rnn_mode
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.embedding_layer = embedding_layer
        self.use_attention =use_attention
        self.decode_top_k = 1
        self.device = device

        self.dropout = None
        if dropout is not None:
            self.dropout = nn.Dropout(dropout)

        if rnn_mode == "LSTM":
            self.rnn = nn.LSTM(input_dim, hidden_dim,
                                num_layers=num_layers,
                                bidirectional=bidirectional,
                                batch_first=True)
        elif rnn_mode == "GRU":
            self.rnn = nn.GRU(input_dim,
                              hidden_dim,
                              num_layers=num_layers,
                              batch_first=True)

        if self.use_attention and attention_type == 'bahdanau':
            self.attention = BahdanauAttention(self.hidden_dim)
        elif self.use_attention and attention_type == 'luong':
            self.attention = LuongAttention(self.hidden_dim) 

        rnn_output_dim = hidden_dim * 2 if bidirectional else hidden_dim

        self.fc = nn.Linear(rnn_output_dim, output_dim)
    
    def init_hidden(self, batch_size, cell_state=False):
        hidden_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
        if cell_state:
            cell_state = torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device)
            return (hidden_state, cell_state)
        else:
            return hidden_state

    def forward_step(self, x, hidden, encoder_outputs):
        batch_size = x.size(0)
        output_size = x.size(1)

        embedded_x = self.embedding_layer(x.long())
        if self.rnn_mode == "LSTM" and not isinstance(hidden, tuple):
            hidden = (hidden, torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(self.device))
        output, hidden = self.rnn(embedded_x, hidden)
        if self.dropout is not None:
            output = self.dropout(output)

        attn = None
        if self.use_attention and encoder_outputs is not None:
            output, attn = self.attention(output, encoder_outputs)

        predicted_softmax = self.fc(output.contiguous().view(-1, self.hidden_dim)).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs, encoder_hidden=None, encoder_outputs=None, target_sequence=None,
                teacher_forcing_ratio=0.0, max_predictions=None, return_intermediates=False, return_attention=False):

        batch_size = inputs.size(0)
        if max_predictions is not None:
            max_length = max_predictions
        else:
            max_length = inputs.size(1)

        # Use the final encoder hidden state as the initial decoder hidden state
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = (encoder_hidden[0][:, -1, :].unsqueeze(0), encoder_hidden[1][:, -1, :].unsqueeze(0))
            decoder_hidden = self.init_hidden(batch_size, cell_state=True)
            decoder_hidden[0][:, -1, :] = encoder_hidden[0][:, -1, :]
            decoder_hidden[1][:, -1, :] = encoder_hidden[1][:, -1, :]
        else:
            encoder_hidden = encoder_hidden[:, -1, :].unsqueeze(0)
            decoder_hidden = self.init_hidden(batch_size)
            decoder_hidden[:, -1, :] = encoder_hidden[:, -1, :]

        decoder_outputs = []
        sequence_symbols = []
        decoder_hidden_states = []
        def decode(step_output):
            decoder_outputs.append(step_output)

            # Take log soft max with temperature
            log_probs = F.log_softmax(torch.Tensor(step_output).to(self.device), dim=1)

            # Collect top k most likely tokens and sample from them
            probs = torch.exp(log_probs)
            symbols = []
            for i in range(batch_size):
                top_probs, top_indices = probs[i].topk(self.decode_top_k)
                sampled_token_index = top_indices[random.choices(range(self.decode_top_k), weights=top_probs, k=1)[0]]
                symbols.append(sampled_token_index)

            # Convert list of symbols to tensor
            symbols = torch.tensor(symbols, device=self.device).unsqueeze(1)
            sequence_symbols.append(symbols)
            return symbols

        decoder_input = inputs[:, 0].unsqueeze(1)
        attentions = []
        for decode_step in range(max_length):

            if random.random() < teacher_forcing_ratio and target_sequence is not None:
                teacher_input = target_sequence[:,decode_step]
                if len(teacher_input.shape) < 2:
                    teacher_input = teacher_input.unsqueeze(0)
                decoder_output, decoder_hidden, step_attn = self.forward_step(teacher_input, decoder_hidden, encoder_outputs)
            else:
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)

            attentions.append(step_attn)
            step_output = decoder_output.squeeze(1)
            if isinstance(decoder_hidden, tuple):
                decoder_hidden_states.append((decoder_hidden[0].squeeze(1), decoder_hidden[1].squeeze(1)))
            else:
                decoder_hidden_states.append(decoder_hidden.squeeze(1))
            symbols = decode(step_output)
            decoder_input = symbols

        decoder_outputs = torch.stack(decoder_outputs, dim=1)
        if return_attention:
            return decoder_outputs, decoder_hidden, sequence_symbols, attentions
        if return_intermediates:

            if isinstance(decoder_hidden_states[0], tuple):
                decoder_hidden_states = tuple(torch.stack(hidden_state, dim=1) for hidden_state in decoder_hidden_states)
                decoder_hidden_states = tuple(hidden_state.squeeze(0) for hidden_state in decoder_hidden_states)
                return decoder_outputs, decoder_hidden, sequence_symbols, decoder_hidden_states
            else:
                decoder_hidden_states = torch.stack(decoder_hidden_states,dim=1)
                return decoder_outputs, decoder_hidden, sequence_symbols, decoder_hidden_states.squeeze(0)

        return decoder_outputs, decoder_hidden, sequence_symbols


class EncoderDecoderRecallmodel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, hidden_dim, 
                 rnn_mode="LSTM", 
                 attention_type="none",
                 use_attention=False, 
                 bidirectional=False, 
                 vocab=None, 
                 frozen_embedding=False,
                 num_layers=1, 
                 pretrained_embedding=None,
                 dropout=None, device='cpu'):
        super(EncoderDecoderRecallmodel, self).__init__()

        self.vocab_size = vocab_size
        self.rnn_mode = rnn_mode
        self.attention_type = attention_type
        self.vocab_size=vocab_size
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.vocab=vocab
        self.model_type = 'encoderdecoder'
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.use_attention = use_attention

        # Initialize with pretrained semantic embedding, if available
        self.frozen_embedding = frozen_embedding
        if pretrained_embedding is not None:
            
            # Initialize the weights of the Embedding layer with the GloVe vectors
            word_to_index = {word: idx for idx, word in enumerate(vocab)}
            for word, index in word_to_index.items():
                word = word.lower()
                if word in pretrained_embedding:
                    self.embedding.weight.data[index] = torch.tensor(pretrained_embedding[word], dtype=torch.float32)

            # Freeze the embedding layer
            for param in self.embedding.parameters():
                param.requires_grad = False
            self.frozen_embedding = True
            logging.info("Pretrained embedding layer initialized.")

        decoder_output_size = hidden_dim
        self.encoder = Encoder(embedding_dim, hidden_dim, rnn_mode, num_layers, bidirectional,embedding_layer=self.embedding,
                               dropout=dropout)
        self.decoder = Decoder(embedding_dim, hidden_dim, decoder_output_size, rnn_mode, num_layers, bidirectional,
                               use_attention=use_attention,attention_type=self.attention_type, embedding_layer=self.embedding,dropout=dropout,device=device)
        
        self.episodic_buffer = EpisodicBuffer(self.embedding, hidden_size=self.hidden_dim, device=self.device,
                                                seq_tokens=True if '<SoS>' in vocab else False)
        self.decoder.episodic_buffer = self.episodic_buffer


    def set_device(self, device):
        self.device = device
        self.decoder.device = device
        self.episodic_buffer.device = device

    def init_hidden(self, batch_size, cell_state=False):
        hidden_state = torch.zeros(self.encoder.num_layers, batch_size, self.encoder.hidden_dim).to(self.device)
        if cell_state:
            cell_state = torch.zeros(self.encoder.num_layers, batch_size, self.encoder.hidden_dim).to(self.device)
            return (hidden_state, cell_state)
        else:
            return hidden_state

    def compute_sequence_likelihood(self, x, target_sequence, use_target_vocab_only=False):
        """
        Compute the likelihood of a given target sequence.

        Args:
            x: Input sequence (source sequence).
            target_sequence: Target sequence for which we want to compute the likelihood.
            use_target_vocab_only: Flag to indicate likelihood calculations should only be relative to 
                                   words appearing in the target sequence

        Returns:
            likelihood: The likelihood of the target sequence given the input sequence.
        """
        batch_size = x.size(0)
        target_len = target_sequence.size(1)

        # Encoder part
        embedded = self.embedding(x)
        encoder_output, encoder_hidden = self.encode(x)
        self.episodic_buffer.write(encoder_output, embedded)

        # Initialize output tensor
        likelihoods = torch.zeros(batch_size).to(x.device)

        # Decoder input starts with the first token of the target sequence
        decoder_input = target_sequence[:, 0].unsqueeze(1)

        # Get the unique tokens in the target sequence if the flag is set
        if use_target_vocab_only:
            target_vocab = target_sequence.unique()

        # Initialize decoder hidden state with the encoder's final hidden state
        decoder_hidden = encoder_hidden

        for t in range(1, target_len):
            decoder_output, decoder_hidden, _ = self.decoder.forward_step(decoder_input, decoder_hidden, encoder_output)
            
            # Get the probability of the actual next token in the target sequence
            next_token = target_sequence[:, t]
            prob_next_token = self.episodic_buffer.read(decoder_output.unsqueeze(1)).squeeze(1)

            if use_target_vocab_only:
                # Mask out the probabilities of tokens not in the target sequence
                mask = torch.zeros_like(prob_next_token).scatter_(1, target_vocab.unsqueeze(0), 1)
                prob_next_token = prob_next_token * mask
                prob_next_token = prob_next_token / prob_next_token.sum(dim=1, keepdim=True)
                prob_next_token = F.log_softmax(prob_next_token, dim=1)

            mask = torch.arange(self.vocab_size).to(x.device).unsqueeze(0) == next_token.unsqueeze(1)
            token_likelihood = prob_next_token[mask].view(batch_size)
            likelihoods += token_likelihood

            # Prepare the next decoder input
            decoder_input = target_sequence[:, t].unsqueeze(1)

        self.episodic_buffer.reset()

        return likelihoods
        
    def encode(self, x, hidden=None,**kwargs):

        batch_size = x.size(0)
        embedded = self.embedding(x)
        if hidden is None:
            hidden = self.init_hidden(batch_size, cell_state=self.rnn_mode=='LSTM')
        encoder_output, hidden = self.encoder(embedded, hidden)
        self.episodic_buffer.write(encoder_output, embedded)

        return encoder_output, hidden
    
    def masked_decode(self, masked_target_seq, encoder_hidden, encoder_output):
        batch_size = masked_target_seq.size(0)
        target_len = masked_target_seq.size(1)
        target_vocab_size = self.decoder.fc.out_features

        # Initialize output tensor
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(masked_target_seq.device)

        # Decoder input starts with the first token of the masked target sequence
        decoder_input = masked_target_seq[:, 0].unsqueeze(1)

        # Initialize decoder hidden state with the encoder's final hidden state
        decoder_hidden = encoder_hidden

        for t in range(1, target_len):
            # Perform one step of decoding
            decoder_output, decoder_hidden, _ = self.decoder.forward_step(decoder_input, decoder_hidden, encoder_output)

            # Store the decoder output at the current time step
            outputs[:, t] = decoder_output.squeeze(1)

            # Get the next token from the masked target sequence
            decoder_input = masked_target_seq[:, t].unsqueeze(1)

        decoder_hidden = outputs.clone().detach()
        outputs = self.episodic_buffer.read(outputs)
        self.episodic_buffer.reset()

        return outputs
    
    def decode(self, x, target_len, encoder_hidden, encoder_output, 
               predict_tokens=False, return_states=False):

        batch_size = x.size(0)
        target_vocab_size = self.decoder.fc.out_features
        decoder = self.decoder

        # Initialize output tensor
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(x.device)

        # Decoder part
        inputs = torch.zeros_like(x).long().to(self.device)
        if return_states:
            outputs, _, symbols, decoder_hidden = self.decoder(inputs, encoder_hidden=encoder_hidden, 
                                                                encoder_outputs=encoder_output, 
                                                                max_predictions=target_len,
                                                                return_intermediates=True)
        else:
            outputs, _, symbols = self.decoder(inputs, encoder_hidden=encoder_hidden, 
                                                encoder_outputs=encoder_output, max_predictions=target_len)

        decoder_hidden = outputs.clone().detach()
        outputs = self.episodic_buffer.read(outputs)
        self.episodic_buffer.reset()

        if predict_tokens:
            return symbols
        
        if return_states:
            return outputs, encoder_output, decoder_hidden
        return outputs
    
    def forward(self, x, target_len, target_sequence=None, hidden_state=None, predict_tokens=False, return_states=False, 
                return_all_states=False, return_attention = False, decode_head=None, no_reset=False, teacher_forcing_ratio=0.0,
                rand_init_decoder=False):
        
        """
        Forward encode-decode call

        Returns:
            x: input sequence
            target_len: Length of output sequence
            hidden_state: Initial hidden state for encoding, will default to generating a null starting state
            predict_tokens: Flag to enable token translation before return
            return_states: Flag to enable return encoder hidden state
            return_all_states: Flag to enable return of intermediate decoder hidden states
            decode_head: int indicating index of decoder head to use if enc-dec model has multiple decoder heads 
            no_reset: Flag to maintain episodic buffer after decoding step (no reset)
            teacher_forcing_ratio: Probability of using ground-truth target for decoding process
            rand_init_decoder: Use a randomized vector to initialize decoding rather than a null vector
        """

        decoder = self.decoder
        batch_size = x.size(0)
        target_vocab_size = decoder.fc.out_features

        # Initialize output tensor
        outputs = torch.zeros(batch_size, target_len, target_vocab_size).to(x.device)

        # Encoding
        embedded = self.embedding(x)

        if hidden_state is not None:
            hidden = hidden_state
        else:
            hidden = self.init_hidden(batch_size, cell_state=self.rnn_mode=='LSTM')
        encoder_output, hidden = self.encoder(embedded, hidden)
        self.episodic_buffer.write(encoder_output, embedded)

        # Decoding
        inputs = torch.zeros_like(x).long().to(self.device)

        if rand_init_decoder:
            inputs = torch.randint(low=0, high=self.vocab_size, size=x.shape).long().to(self.device)
            
        if return_attention:
            outputs,  _, symbols, attention = decoder(inputs, encoder_hidden=hidden, encoder_outputs=encoder_output, target_sequence=target_sequence, 
                                                            max_predictions=target_len, return_attention=True, teacher_forcing_ratio=teacher_forcing_ratio)
            pre_retrieval_output = deepcopy(outputs)
            outputs = self.episodic_buffer.read(outputs)
            if not no_reset:
                self.episodic_buffer.reset()
            return outputs, attention, pre_retrieval_output, encoder_output

        outputs,  _, symbols, decoder_intermediates = decoder(inputs, encoder_hidden=hidden, encoder_outputs=encoder_output, target_sequence=target_sequence, 
                                                            max_predictions=target_len, return_intermediates=True, teacher_forcing_ratio=teacher_forcing_ratio)
        
        outputs = self.episodic_buffer.read(outputs)
        if not no_reset:
            self.episodic_buffer.reset()

        if predict_tokens:
            return symbols
        
        if return_all_states:
            return outputs, encoder_output, decoder_intermediates.swapaxes(0,1)

        if return_states:
            return outputs, encoder_output, hidden
        return outputs
