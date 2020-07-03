from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
)
from repairer.utils import try_gpu

#https://medium.com/@adam.wearne/seq2seq-with-pytorch-46dc00ff5164

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.linear_in = nn.Linear(hidden_size, hidden_size, bias=False)

    def score(self, hidden_state, encoder_states):
        """
        Args:
            hidden_state (tgt_len(=1), batch, hidden_size)
            encoder_states (src_len, batch, hidden_size)
        Return:
            score (batch, tgt_len(=1), src_len)
        """
        ##attention type: "general"
        h_t = hidden_state.transpose(0,1).contiguous() #(batch, tgt_len=1, dim)
        h_s = encoder_states.transpose(0,1).contiguous() #(batch, src_len, dim)

        src_batch, src_len, src_dim = h_s.size()
        tgt_batch, tgt_len, tgt_dim = h_t.size()

        h_t_ = h_t.view(tgt_batch * tgt_len, tgt_dim)
        h_t_ = self.linear_in(h_t_)
        h_t = h_t_.view(tgt_batch, tgt_len, tgt_dim)
        h_s_ = h_s.transpose(1, 2)

        # (batch, t_len, d) x (batch, d, s_len) --> (batch, t_len=1, s_len)
        score = torch.bmm(h_t, h_s_)
        return score #(batch, t_len=1, s_len)



    def forward(self, hidden, encoder_outputs, src_mask, tgt_mask=None):
        """
        Args:
            hidden (tgt_len(=1), batch, hidden_size)
            encoder_outputs (src_len, batch, hidden_size)
            src_mask (batch, src_len)
            tgt_mask (batch, tgt_len)
        Return:
            attn: (batch, tgt_len(=1), src_len)
        """
        tgt_len, b_size, hiddim = hidden.size()
        src_len = encoder_outputs.size(0)

        attn_scores = self.score(hidden, encoder_outputs) #(batch, t_len(=1), s_len)


        # Apply mask so network does not attend <pad> tokens
        src_mask = src_mask.unsqueeze(1).expand(b_size, tgt_len, src_len) #(batch, t_len(=1), s_len)
        attn_scores = attn_scores.masked_fill(src_mask == 0, -1e10)
        if tgt_mask is not None:
            tgt_mask = tgt_mask.unsqueeze(2).expand(b_size, tgt_len, src_len) #(batch, t_len(=1), s_len)
            attn_scores = attn_scores.masked_fill(tgt_mask == 0, -1e10)

        # Return softmax over attention scores
        return F.softmax(attn_scores, dim=2) #(batch, t_len(=1), s_len)


class Decoder(nn.Module):
    def __init__(self, embedding, embedding_size,
                 hidden_size, output_size, n_layers=2, dropout=0.3):

        super(Decoder, self).__init__()

        # Basic network params
        self.hidden_size = hidden_size
        self.output_size = output_size #this should be vocab size
        self.n_layers = n_layers
        self.dropout = dropout
        self.embedding = embedding

        self.lstm = nn.LSTM(embedding_size, hidden_size, n_layers,
                          dropout=dropout)

        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.attn = Attention(hidden_size)
        self.copy_attn = Attention(hidden_size)


    def forward(self, current_token, hidden_state, encoder_outputs, mask, context_vec, extra_feed):
        """
        current_token: [t_len=1, batch]
        hidden_state:  (h, c)
        encoder_outputs: [s_len, batch, dim]
        mask: [batch, s_len]
        extra_feed: [1, batch, dim]
        """
        _, batch_size, enc_dim = encoder_outputs.size()
        if context_vec is None:
            context_vec = try_gpu(torch.zeros(1, batch_size, enc_dim).float())

        # convert current_token to word_embedding
        embedded = self.embedding(current_token) #[1, batch, dim]
        embedded = torch.cat([embedded, context_vec], dim=2)
        if extra_feed is not None:
            embedded = torch.cat([embedded, extra_feed], dim=2)

        # Pass through LSTM
        rnn_output, hidden_state = self.lstm(embedded, hidden_state)
        #rnn_output: (seq_len=1, batch, hidden_size)

        # Calculate attention weights
        attns = {}
        p_attn = self.attn(rnn_output, encoder_outputs, mask) #(batch, 1, s_len)
        # copy_attn = self.copy_attn(rnn_output, encoder_outputs, mask)
        attns["std"] = p_attn
        attns["copy"] = p_attn #copy_attn

        # Calculate context vector
        context = p_attn.bmm(encoder_outputs.transpose(0, 1))
          #encoder_outputs.transpose(0, 1):  #(batch, s_len, dim)
          #context: (batch, 1, dim)
        context_vec = context.transpose(0,1)

        # Concatenate  context vector and RNN output
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.tanh(self.concat(concat_input)) #(batch, hidden_size)

        # Pass concat_output to final output layer
        output = concat_output #(batch, hidden_size)

        # Return output and final hidden state
        return output, hidden_state, attns, context_vec
