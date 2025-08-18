import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):
    r"""
    Applies an attention mechanism on the output features from the decoder.
    Source: https://github.com/IBM/pytorch-seq2seq/blob/master/seq2seq/models/attention.py

    Args:
        dim(int): The number of expected features in the output
        is_local(bool): Indicate if attention should be global or local
        window_size(int): If attention is local, indicate how large the attention window should be

    Inputs: output, context
        - **output** (batch, output_len, dimensions): tensor containing the output features from the decoder.
        - **context** (batch, input_len, dimensions): tensor containing features of the encoded input sequence.

    Outputs: output, attn
        - **output** (batch, output_len, dimensions): tensor containing the attended output features from the decoder.
        - **attn** (batch, output_len, input_len): tensor containing attention weights.

    Attributes:
        linear_out (torch.nn.Linear): applies a linear transformation to the incoming data: :math:`y = Ax + b`.
        mask (torch.Tensor, optional): applies a :math:`-inf` to the indices specified in the `Tensor`.

    """
    def __init__(self, dim, is_local=False, window_size=3):
        super(LuongAttention, self).__init__()
        self.dim = dim
        self.is_local = is_local
        self.window_size = window_size
        self.linear_out = nn.Linear(self.dim*2, self.dim)
        self.mask = None

    def set_mask(self, mask):
        """
        Sets indices to be masked

        Args:
            mask (torch.Tensor): tensor containing indices to be masked
        """
        self.mask = mask

    def forward(self, output, context):
        batch_size = output.size(0)
        hidden_size = output.size(2)
        input_size = context.size(1)

        # Global attention
        attn = torch.bmm(output, context.transpose(1, 2))
        if self.mask is not None:
            attn.data.masked_fill_(self.mask, -float('inf'))
        attn = F.softmax(attn.view(-1, input_size), dim=1).view(batch_size, -1, input_size)

        mix = torch.bmm(attn, context)
        combined = torch.cat((mix, output), dim=2)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)

        return output, attn

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.key_layer = nn.Linear(hidden_dim, hidden_dim)
        self.query_layer = nn.Linear(hidden_dim, hidden_dim)
        self.energy_layer = nn.Linear(hidden_dim, 1)
        self.linear_out = nn.Linear(self.hidden_dim*2, self.hidden_dim)
        self.mask = None

    def set_mask(self, mask):
        self.mask = mask

    def _calculate_scores(self, query, key):
        # query = [batch size, query len, value len, hidden dim]
        # key = [batch size, key len, hidden dim]
        key = key.unsqueeze(1)
        
        keys = self.key_layer(key)
        queries = self.query_layer(query)
        scores = self.energy_layer(torch.tanh(keys + queries)).squeeze(-1)
        
        return scores

    def forward(self, output, context):
        # query = [batch size, query len, hidden dim]
        # keys = [batch size, key len, hidden dim]
        batch_size = output.size(0)
        hidden_size = output.size(2)

        interim_out = output.unsqueeze(2).repeat(1, 1, context.size(1), 1)
        scores = self._calculate_scores(interim_out, context)
        if self.mask is not None:
            scores = scores.masked_fill(self.mask == 0, -1e10)
    
        attn = F.softmax(scores, dim=-1)
        context = torch.bmm(attn, context)
        combined = torch.cat((context, output), dim=2)
        output = torch.tanh(self.linear_out(combined.view(-1, 2 * hidden_size))).view(batch_size, -1, hidden_size)
        
        return output, attn
    