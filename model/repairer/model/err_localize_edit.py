from collections import defaultdict
import itertools

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence,
    pad_packed_sequence,
)
from repairer.utils import prepare_rnn_seq, recover_rnn_seq


from torch.autograd import Variable

import numpy as np

from repairer.utils import try_gpu, BOS_INDEX, EOS_INDEX, PAD_INDEX, UNK_INDEX, BOS, EOS, PAD, UNK
from repairer.model.base import Model
from repairer.model.decoder import Decoder, Attention
from repairer.model.beam_search_onmt import BeamSearch
from repairer.model.copy_generator import CopyGenerator, CopyGeneratorLoss, collapse_copy_scores
from repairer.model.attention_zoo import GraphAttentionEncoderFlow


def make_one_hot(labels, C):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    labels : torch.autograd.Variable of torch.cuda.LongTensor
        (N, ), where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C, where C is class number. One-hot encoded.
    '''
    labels = labels.unsqueeze(1)
    one_hot = torch.cuda.FloatTensor(labels.size(0), C).zero_()
    target = one_hot.scatter_(1, labels.data, 1)
    target = Variable(target)
    return target


def cross_entropy_after_probsum(pred, soft_targets):
    # pred: (batchsize, num_of_classes)
    # soft_targets: (batchsize, num_of_classes)
    logsoftmax = nn.LogSoftmax()
    softmax = nn.Softmax()
    probsum = torch.sum(soft_targets * softmax(pred), 1)
    return torch.mean(- torch.log(probsum))


class ErrLocalizeEditModel(Model):

    def __init__(self, config, meta):
        super().__init__(config, meta)
        c_model = config.model
        self.c_model = c_model
        self.meta = meta

        ##Embedding
        self.tok_embedder = nn.Embedding(
            len(meta.vocab),
            c_model.tok_embed.dim,
        )

        self.dropout = nn.Dropout(c_model.dropout)
        self.pos_embed_dim = c_model.pos_embed.dim
        assert self.pos_embed_dim % 2 == 0

        ##To combine positional encoding and lstm1 output
        combo_mlp_parts_pos_enc = []
        combo_mlp_parts_pos_enc.append(nn.Dropout(c_model.dropout))
        combo_mlp_parts_pos_enc.append(nn.Linear(c_model.tok_embed.dim + c_model.pos_embed.dim, c_model.tok_embed.dim))
        combo_mlp_parts_pos_enc.append(nn.ReLU())
        self.combo_mlp_pos_enc = nn.Sequential(*combo_mlp_parts_pos_enc)

        ##Graph attention
        graph_attn_dim = c_model.tok_embed.dim
        attn_layers = max(1, self.c_model.graph)
        self.code_msg_graph_attention = GraphAttentionEncoderFlow(num_layers=attn_layers, d_model=graph_attn_dim, heads=5, d_ff=graph_attn_dim, dropout=c_model.dropout, attention_dropout=c_model.dropout)


        ##LSTM1 & LSTM2
        # input: (seq_len, batch, tok_embed_dim)
        # output: (hiddens, (h_n, c_n))
        # hiddens: (seq_len, batch, 2 * lstm_dim)
        # h_n and c_n: (lstm_layers * 2, batch, lstm_dim)
        self.text_seq_embedder1 = nn.LSTM(
            c_model.tok_embed.dim + c_model.pos_embed.dim,
            c_model.tok_seq_embed.lstm_dim,
            num_layers=c_model.tok_seq_embed.lstm_layers -1,
            bidirectional=True,
            dropout=c_model.dropout,
        )
        self.text_linear_after_1st_lstm = nn.Linear(c_model.tok_seq_embed.lstm_dim *2, c_model.tok_embed.dim)
        self.text_seq_embedder2 = nn.LSTM(
            c_model.tok_embed.dim,
            c_model.tok_seq_embed.lstm_dim,
            num_layers=1,
            bidirectional=True,
            dropout=c_model.dropout,
        )

        self.code_seq_embedder1 = nn.LSTM(
            c_model.tok_embed.dim + c_model.pos_embed.dim,
            c_model.tok_seq_embed.lstm_dim,
            num_layers=c_model.tok_seq_embed.lstm_layers -1,
            bidirectional=True,
            dropout=c_model.dropout,
        )
        self.code_linear_after_1st_lstm = nn.Linear(c_model.tok_seq_embed.lstm_dim *2, c_model.tok_embed.dim)
        self.code_seq_embedder2 = nn.LSTM(
            graph_attn_dim,
            c_model.tok_seq_embed.lstm_dim,
            num_layers=1,
            bidirectional=True,
            dropout=c_model.dropout,
        )
        self.tok_embed_dim = (
            2 * c_model.tok_seq_embed.lstm_layers * c_model.tok_seq_embed.lstm_dim
        )
        if c_model.type in ["code_compiler_text", "code_compiler"]:
            self.msg_seq_embedder1 = nn.LSTM(
                c_model.tok_embed.dim + c_model.pos_embed.dim,
                c_model.tok_seq_embed.lstm_dim,
                num_layers=c_model.tok_seq_embed.lstm_layers -1,
                bidirectional=True,
                dropout=c_model.dropout,
            )
            self.msg_linear_after_1st_lstm = nn.Linear(c_model.tok_seq_embed.lstm_dim *2, c_model.tok_embed.dim)
            self.msg_seq_embedder2 = nn.LSTM(
                graph_attn_dim,
                c_model.tok_seq_embed.lstm_dim,
                num_layers=1,
                bidirectional=True,
                dropout=c_model.dropout,
            )

        ##To prepare lstm3 input
        # input: (batch, num_lines, combo_in_dim)
        # output: (batch, num_lines, combo_out_dim)
        combo_mlp_parts = []
        if c_model.type == "code_compiler_text":
            last_dim = 3* self.tok_embed_dim
        elif c_model.type == "code_compiler":
            last_dim = 2* self.tok_embed_dim
        elif c_model.type == "code_only":
            last_dim = self.tok_embed_dim
        else:
            raise NotImplementedError
        dims = c_model.combo_mlp.hidden_dims.to_vanilla_() + [c_model.combo_mlp.out_dim]
        for hidden_dim in dims:
            combo_mlp_parts.append(nn.Dropout(c_model.dropout))
            combo_mlp_parts.append(nn.Linear(last_dim, hidden_dim))
            combo_mlp_parts.append(nn.ReLU())
            last_dim = hidden_dim
        self.combo_mlp = nn.Sequential(*combo_mlp_parts)

        ##LSTM3 (line-level lstm)
        # input: (batch, num_lines, combo_out_dim)
        # hiddens: (batch, seq_len, 2 * lstm_dim)
        self.line_seq_embedder = nn.LSTM(
            c_model.combo_mlp.out_dim,
            c_model.line_seq_embed.lstm_dim,
            num_layers=c_model.line_seq_embed.lstm_layers,
            bidirectional=True,
            batch_first=True,
            dropout=c_model.dropout,
        )

        ###### Localize part ######
        self.add_residual = c_model.final_mlp.get_('add_residual')

        # input: (batch, num_lines, final_in_dim)
        # output: (batch, num_lines, 1)
        final_mlp_parts = []
        last_dim = 2 * c_model.line_seq_embed.lstm_dim
        if self.add_residual:
            last_dim += c_model.combo_mlp.out_dim
            print('Add residual of {} dimension (localization)'.format(c_model.combo_mlp.out_dim))
        dims = c_model.final_mlp.hidden_dims.to_vanilla_()
        for hidden_dim in dims:
            final_mlp_parts.append(nn.Dropout(c_model.dropout))
            final_mlp_parts.append(nn.Linear(last_dim, hidden_dim))
            final_mlp_parts.append(nn.ReLU())
            last_dim = hidden_dim
        final_mlp_parts.append(nn.Dropout(c_model.dropout))
        final_mlp_parts.append(nn.Linear(last_dim, 1))
        self.final_mlp = nn.Sequential(*final_mlp_parts)
        ##################


        ###### Edit part ######
        last_dim = 2 * c_model.line_seq_embed.lstm_dim

        #Decoder
        enc_total_dim = c_model.tok_seq_embed.lstm_dim *2 *c_model.tok_seq_embed.lstm_layers
        i_d = enc_total_dim +last_dim
        o_d = enc_total_dim
        self.bridge_c, self.bridge_h = [nn.Sequential(nn.Dropout(c_model.dropout), nn.Linear(i_d,i_d), nn.Tanh(), nn.Linear(i_d,o_d), nn.Tanh())] *2
        self.decoder = Decoder(self.tok_embedder,
                          c_model.tok_embed.dim + c_model.tok_seq_embed.lstm_dim*2,
                          c_model.tok_seq_embed.lstm_dim *2, #to accomodate (h, c) from *bidirectional* encoder
                          len(meta.vocab),
                          n_layers=c_model.tok_seq_embed.lstm_layers,
                          dropout=c_model.dropout)
        self.copy_generator = CopyGenerator(input_size=c_model.tok_seq_embed.lstm_dim *2, output_size=len(meta.vocab))
        ##################


        # loss
        self.copy_generator_loss = CopyGeneratorLoss(vocab_size=len(meta.vocab))



    def initialize(self, config, meta):
        """
        Initialize GloVe or whatever.
        """
        pass

    def create_mask(self, input_sequence, only_for_pad=False): #for attention
        # input_sequence: indicies
        # 1 means not masked, 0 means masked
        if only_for_pad:
            return (input_sequence != PAD_INDEX)
        else:
            return ((input_sequence != PAD_INDEX) & (input_sequence != BOS_INDEX) & (input_sequence != EOS_INDEX))


    def forward_encode(self, batch):
        """
        Assume all examples in batch have the same number of code lines.
          the dataset iteration (err_dataset.py) makes sure that each batch is within a s*-*.json file
        """

        text_stuff = None
        if self.c_model.type == "code_compiler_text":
            text_stuff, (code_stuff, code_slen), (msg_stuff, msg_slen), err_linenos, gold_code_line_stuff = self.get_stuffs_to_embed(batch)
            assert len(text_stuff) == len(msg_stuff) == len(err_linenos)
        elif self.c_model.type == "code_compiler":
            (code_stuff, code_slen), (msg_stuff, msg_slen), err_linenos, gold_code_line_stuff = self.get_stuffs_to_embed(batch)
        elif self.c_model.type == "code_only":
            (code_stuff, code_slen), gold_code_line_stuff = self.get_stuffs_to_embed(batch)
        else:
            raise NotImplementedError

        batch_size = len(code_stuff)
        num_lines = len(code_stuff[0])
        if text_stuff:
            assert len(text_stuff) == len(code_stuff)
            assert all(len(x) == num_lines for x in text_stuff)


        def prepare_for_removing_pad(mask):
            #mask: (batch, fat_len = num_lines *slen). Here we use for graph_mask
            #this function tries to remove masked tokens (those not in the graph)
            mask = mask.long()
            b_size, fat_len = mask.size()
            slim_lens = mask.sum(dim=1) #(batch,)
            max_slim_len = max(slim_lens.cpu().numpy())
            fat2slim = [] #-1 if no correspondence
            slim2fat = [] #-1 for pad
            fat_count = 0
            slim_count = 0
            for b_idx in range(b_size):
                _mask = mask[b_idx] #e.g. torch.tensor([0,0,1,1,1,0,1,0])
                positive_idxs = (_mask==1).nonzero().squeeze(1) #e.g. tensor([2, 3, 4, 6])

                _fat2slim = _mask.index_add_(0, positive_idxs, try_gpu(torch.arange(positive_idxs.size(0))) +slim_count) #tensor([0,0,1,2,3,0,4,0]) + appropriate slim_count
                _fat2slim = _fat2slim -1 #tensor([-1,-1,0,1,2,-1,3,-1]) + appropriate slim_count
                _fat2slim = _fat2slim.unsqueeze(0) #(1, fat_len)

                _slim2fat = positive_idxs + fat_count

                slim_count += max_slim_len
                fat_count += fat_len

                fat2slim.append(_fat2slim)
                slim2fat.append(_slim2fat)
            assert fat_count == b_size * fat_len
            assert slim_count == b_size * max_slim_len

            fat2slim = torch.cat(fat2slim, dim=0)
            slim2fat = pad_sequence(slim2fat, batch_first=True, padding_value=-1)
            assert slim2fat.size(1) == max_slim_len
            return fat2slim, slim2fat #(batch, fat_len), (batch, max_slim_len)


        def prep_graph_mask(true_slen): #true_slen == max(code_len, msg_len)
            # return:
            #   graph_mask: (batch, 1+num_lines * true_slen). 1 means this token is in graph, 0 means not
            _b_size_ = len(batch)
            if batch[0].graph_mask == None: return None
            _num_seqs = len(batch[0].graph_mask)
            graph_mask = try_gpu(torch.zeros(_b_size_, _num_seqs, true_slen))
            for b_id, ex in enumerate(batch):
                if ex.graph_mask == None: return None
                for seq_id, src_seq in enumerate(ex.graph_mask):
                    curr_len = len(src_seq)
                    assert (curr_len <= true_slen) #b/c we skipped long examples in err_dataset.py
                    graph_mask[b_id, seq_id, :curr_len] = torch.tensor(src_seq)
            graph_mask = graph_mask.view(_b_size_, -1)
            return graph_mask.byte()


        def prep_graph_A(slim_len):
            # return:
            #   graph_A: (batch, slim_len, slim_len) #slim_len should mean num_nodes here
            _b_size_ = len(batch)
            graph_A = try_gpu(torch.zeros(_b_size_, slim_len, slim_len))
            for b_id, ex in enumerate(batch):
                curr_nodes = len(ex.graph_A)
                assert (curr_nodes <= slim_len)
                graph_A[b_id, :curr_nodes, :curr_nodes] = torch.tensor(ex.graph_A)
            graph_A = 1 - graph_A  #need to flip. In OpenNMT self-attn implementation, 0 means not masked, 1 means masked
            return graph_A.byte()


        #wemb -> LSTM1 -> graph-attention -> LSTM2
        if self.c_model.type == "code_compiler":
            #Get word embedding
            _true_slen = max(code_slen, msg_slen)
            code_indices, code_wembs = self.embed_stuff_for_wembs(code_stuff, _true_slen)
            msg_indices, msg_wembs = self.embed_stuff_for_wembs(msg_stuff, _true_slen)
            _b_size, _num_lines, _code_slen, _wembdim = code_wembs.size() #_code_slen == _true_slen
            _, _msg_slen, _ = msg_wembs.size()

            #Concat absolute positional emb
            code_wembs = code_wembs.view(_b_size, -1, _wembdim) #(batch, num_lines * true_slen, dim)
            pos_embs = self.positional_encoding([0]*_b_size, code_wembs.size(1)) #(batch, fat_len, pos_embed_dim)
            code_wembs = torch.cat([code_wembs, pos_embs], dim=2) #(batch, fat_len, dim)
            _wembdim2 = code_wembs.size(2) #with pos_embs

            msg_mask = self.create_mask(msg_indices).unsqueeze(1) #(batch, 1, msg_slen)
            pos_embs = self.positional_encoding([0]*_b_size, msg_indices.size(1)) #(batch, msg_slen, pos_embed_dim)
            msg_wembs = torch.cat([msg_wembs, pos_embs], dim=2) #(batch, msg_slen, dim)


            #LSTM encoding1
            code_wembs = code_wembs.view(_b_size, _num_lines, _true_slen, _wembdim2) #(batch, num_lines, true_slen, dim)
            msg_embeds1, msg_embeds_c1, msg_wembs = self.embed_stuff_for_lstm(msg_indices, msg_wembs, self.msg_seq_embedder1)
            code_embeds1, code_embeds_c1, code_wembs = self.embed_stuff_for_lstm(code_indices, code_wembs, self.code_seq_embedder1)
            msg_wembs = self.msg_linear_after_1st_lstm(msg_wembs) #(batch, msg_slen, wembdim)
            code_wembs = self.code_linear_after_1st_lstm(code_wembs) #(batch, num_lines, true_slen, wembdim)

            _b_size, _num_lines, _true_slen, _wembdim = code_wembs.size()


            # Line-level Positional encoding with error index information
            pos_embeds = self.positional_encoding(err_linenos, num_lines) #(batch, num_lines, posdim)
            pos_embeds = pos_embeds.unsqueeze(2).repeat(1, 1, _true_slen, 1) #(batch, num_lines, true_slen, posdim)
            code_wembs = self.combo_mlp_pos_enc(torch.cat([code_wembs, pos_embeds], dim=3)) #(batch, num_lines, true_slen, wembdim)

            msg_code_graph_mask = prep_graph_mask(_true_slen) #(batch, 1+num_lines * true_slen)
            if msg_code_graph_mask is None:
                # print ("msg_code_graph_mask is None!")
                pass
            else:
                #Reshape and make code_wembs slim (remove tokens not in graph)
                code_indices = code_indices.view(_b_size, -1) #(batch, num_lines * true_slen)
                code_wembs = code_wembs.view(_b_size, -1, _wembdim) #(batch, num_lines * true_slen, dim)

                msg_code_wembs_orig = torch.cat([msg_wembs, code_wembs], dim=1) #(batch, 1+num_lines * true_slen, dim)
                fat2slim, slim2fat = prepare_for_removing_pad(msg_code_graph_mask) #(batch, fat_len=num_lines * true_slen), (batch, max_slim_len)
                msg_code_wembs_w_dummy = torch.cat([try_gpu(torch.zeros(1,_wembdim)).float(), msg_code_wembs_orig.view(-1,_wembdim)], dim=0)
                msg_code_wembs = F.embedding(slim2fat+1, msg_code_wembs_w_dummy) #(batch, max_slim_len, _wembdim)
                max_slim_len = slim2fat.size(1)

                #Attention (graph)
                msg_code_graph_A = prep_graph_A(max_slim_len)
                msg_code_wembs = self.code_msg_graph_attention(msg_code_wembs, msg_code_graph_A) #code_wembs: (batch, slim_len, dim)
                _, _, _out_dim = msg_code_wembs.size()
                msg_code_wembs_w_dummy = torch.cat([try_gpu(torch.zeros(1,_out_dim)).float(), msg_code_wembs.view(-1,_out_dim)], dim=0)
                msg_code_wembs = F.embedding(fat2slim+1, msg_code_wembs_w_dummy) #(batch, fat_len, _out_dim)
                msg_code_wembs = msg_code_wembs * msg_code_graph_mask.unsqueeze(2).float() + msg_code_wembs_orig * (msg_code_graph_mask==0).unsqueeze(2).float()
                msg_code_wembs = msg_code_wembs.view(_b_size, 1+_num_lines, _true_slen, _out_dim) #(batch, 1+num_lines, true_slen, dim)
                msg_wembs = msg_code_wembs[:, 0].contiguous() #(batch, true_slen, dim)
                code_wembs = msg_code_wembs[:, 1:].contiguous() #(batch, num_lines, true_slen, dim)


            code_indices = code_indices.view(_b_size, _num_lines, _true_slen) #(batch, num_lines, true_slen)

            #LSTM encoding2
            msg_embeds2, msg_embeds_c2, msg_embeds_output = self.embed_stuff_for_lstm(msg_indices, msg_wembs, self.msg_seq_embedder2)
            msg_embeds = torch.cat([msg_embeds1, msg_embeds2], dim=1) #(batch, total_lstm_layers * 2 * lstm_dim)
            msg_embeds_c = torch.cat([msg_embeds_c1, msg_embeds_c2], dim=1) #(batch, total_lstm_layers * 2 * lstm_dim)
            code_embeds2, code_embeds_c2, code_embeds_output = self.embed_stuff_for_lstm(code_indices, code_wembs, self.code_seq_embedder2)
            code_embeds = torch.cat([code_embeds1, code_embeds2], dim=2) #(batch, num_lines, total_lstm_layers * 2 * lstm_dim)
            code_embeds_c = torch.cat([code_embeds_c1, code_embeds_c2], dim=2) #(batch, num_lines, total_lstm_layers * 2 * lstm_dim)

            # Concatenate everything
            combo = torch.cat([
                    code_embeds,
                    msg_embeds.unsqueeze(1).expand(-1, num_lines, -1),
                ], dim=2)
            combo = self.combo_mlp(combo)


        elif self.c_model.type == "code_only":
            _true_slen = code_slen
            code_indices, code_wembs = self.embed_stuff_for_wembs(code_stuff, _true_slen)
            _b_size, _num_lines, _code_slen, _wembdim = code_wembs.size()

            #Concat absolute positional emb
            code_wembs = code_wembs.view(_b_size, -1, _wembdim) #(batch, num_lines * true_slen, dim)
            pos_embs = self.positional_encoding([0]*_b_size, code_wembs.size(1)) #(batch, fat_len, pos_embed_dim)
            code_wembs = torch.cat([code_wembs, pos_embs], dim=2) #(batch, fat_len, dim)
            _wembdim2 = code_wembs.size(2) #with pos_embs

            #LSTM encoding1
            code_wembs = code_wembs.view(_b_size, _num_lines, _true_slen, _wembdim2) #(batch, num_lines, true_slen, dim)
            code_embeds1, code_embeds_c1, code_wembs = self.embed_stuff_for_lstm(code_indices, code_wembs, self.code_seq_embedder1)
            code_wembs = self.code_linear_after_1st_lstm(code_wembs) #(batch, num_lines, true_slen, wembdim)

            _b_size, _num_lines, _true_slen, _wembdim = code_wembs.size()

            #LSTM encoding2
            code_embeds2, code_embeds_c2, code_embeds_output = self.embed_stuff_for_lstm(code_indices, code_wembs, self.code_seq_embedder2)
            code_embeds = torch.cat([code_embeds1, code_embeds2], dim=2) #(batch, num_lines, total_lstm_layers * 2 * lstm_dim)
            code_embeds_c = torch.cat([code_embeds_c1, code_embeds_c2], dim=2) #(batch, num_lines, total_lstm_layers * 2 * lstm_dim)

            # Concatenate everything
            combo = code_embeds
            combo = self.combo_mlp(combo)


        elif self.c_model.type == "code_compiler_text":
            _true_slen = max(code_slen, msg_slen)
            code_indices, code_wembs = self.embed_stuff_for_wembs(code_stuff, _true_slen)
            msg_indices, msg_wembs = self.embed_stuff_for_wembs(msg_stuff, _true_slen)
            text_indices, text_wembs = self.embed_stuff_for_wembs(text_stuff, 0) #text_indices: (batch, num_lines, text_slen). text_wembs: (batch, num_lines, text_slen, embed_dim)
            _b_size, _num_lines, _code_slen, _wembdim = code_wembs.size()
            _, _msg_slen, _ = msg_wembs.size()

            #Concat absolute positional emb
            code_wembs = code_wembs.view(_b_size, -1, _wembdim) #(batch, num_lines * true_slen, dim)
            pos_embs = self.positional_encoding([0]*_b_size, code_wembs.size(1)) #(batch, fat_len, pos_embed_dim)
            code_wembs = torch.cat([code_wembs, pos_embs], dim=2) #(batch, fat_len, dim)
            _wembdim2 = code_wembs.size(2) #with pos_embs

            text_wembs = text_wembs.view(_b_size, -1, _wembdim)
            pos_embs = self.positional_encoding([0]*_b_size, text_wembs.size(1))
            text_wembs = torch.cat([text_wembs, pos_embs], dim=2)

            msg_mask = self.create_mask(msg_indices).unsqueeze(1) #(batch, 1, msg_slen)
            pos_embs = self.positional_encoding([0]*_b_size, msg_indices.size(1)) #(batch, msg_slen, pos_embed_dim)
            msg_wembs = torch.cat([msg_wembs, pos_embs], dim=2) #(batch, msg_slen, dim)


            #LSTM encoding1
            code_wembs = code_wembs.view(_b_size, _num_lines, _true_slen, _wembdim2) #(batch, num_lines, true_slen, dim)
            text_wembs = text_wembs.view(_b_size, _num_lines, -1, _wembdim2)
            msg_embeds1, msg_embeds_c1, msg_wembs = self.embed_stuff_for_lstm(msg_indices, msg_wembs, self.msg_seq_embedder1)
            code_embeds1, code_embeds_c1, code_wembs = self.embed_stuff_for_lstm(code_indices, code_wembs, self.code_seq_embedder1)
            text_embeds1, text_embeds_c1, text_wembs = self.embed_stuff_for_lstm(text_indices, text_wembs, self.text_seq_embedder1)
            msg_wembs = self.msg_linear_after_1st_lstm(msg_wembs) #(batch, msg_slen, wembdim)
            code_wembs = self.code_linear_after_1st_lstm(code_wembs) #(batch, num_lines, true_slen, wembdim)
            text_wembs = self.text_linear_after_1st_lstm(text_wembs)

            _b_size, _num_lines, _true_slen, _wembdim = code_wembs.size()

            # Line-level Positional encoding with error index information
            pos_embeds = self.positional_encoding(err_linenos, num_lines) #(batch, num_lines, posdim)
            pos_embeds = pos_embeds.unsqueeze(2).repeat(1, 1, _true_slen, 1) #(batch, num_lines, true_slen, posdim)
            code_wembs = self.combo_mlp_pos_enc(torch.cat([code_wembs, pos_embeds], dim=3)) #(batch, num_lines, true_slen, wembdim)

            msg_code_graph_mask = prep_graph_mask(_true_slen) #(batch, 1+num_lines * true_slen)
            if msg_code_graph_mask is None:
                # print ("msg_code_graph_mask is None!")
                pass
            else:
                #Reshape and make code_wembs slim (remove tokens not in graph)
                code_indices = code_indices.view(_b_size, -1) #(batch, num_lines * true_slen)
                code_wembs = code_wembs.view(_b_size, -1, _wembdim) #(batch, num_lines * true_slen, dim)

                msg_code_wembs_orig = torch.cat([msg_wembs, code_wembs], dim=1) #(batch, 1+num_lines * true_slen, dim)
                fat2slim, slim2fat = prepare_for_removing_pad(msg_code_graph_mask) #(batch, fat_len=num_lines * true_slen), (batch, max_slim_len)
                msg_code_wembs_w_dummy = torch.cat([try_gpu(torch.zeros(1,_wembdim)).float(), msg_code_wembs_orig.view(-1,_wembdim)], dim=0)
                msg_code_wembs = F.embedding(slim2fat+1, msg_code_wembs_w_dummy) #(batch, max_slim_len, _wembdim)
                max_slim_len = slim2fat.size(1)


                #Attention (graph)
                msg_code_graph_A = prep_graph_A(max_slim_len)
                msg_code_wembs = self.code_msg_graph_attention(msg_code_wembs, msg_code_graph_A) #code_wembs: (batch, slim_len, dim)
                _, _, _out_dim = msg_code_wembs.size()
                msg_code_wembs_w_dummy = torch.cat([try_gpu(torch.zeros(1,_out_dim)).float(), msg_code_wembs.view(-1,_out_dim)], dim=0)
                msg_code_wembs = F.embedding(fat2slim+1, msg_code_wembs_w_dummy) #(batch, fat_len, _out_dim)

                msg_code_wembs = msg_code_wembs * msg_code_graph_mask.unsqueeze(2).float() + msg_code_wembs_orig * (msg_code_graph_mask==0).unsqueeze(2).float()
                msg_code_wembs = msg_code_wembs.view(_b_size, 1+_num_lines, _true_slen, _out_dim) #(batch, 1+num_lines, true_slen, dim)
                msg_wembs = msg_code_wembs[:, 0].contiguous() #(batch, true_slen, dim)
                code_wembs = msg_code_wembs[:, 1:].contiguous() #(batch, num_lines, true_slen, dim)


            code_indices = code_indices.view(_b_size, _num_lines, _true_slen) #(batch, num_lines, true_slen)

            #LSTM encoding2
            msg_embeds2, msg_embeds_c2, msg_embeds_output = self.embed_stuff_for_lstm(msg_indices, msg_wembs, self.msg_seq_embedder2)
            msg_embeds = torch.cat([msg_embeds1, msg_embeds2], dim=1) #(batch, total_lstm_layers * 2 * lstm_dim)
            msg_embeds_c = torch.cat([msg_embeds_c1, msg_embeds_c2], dim=1) #(batch, total_lstm_layers * 2 * lstm_dim)
            code_embeds2, code_embeds_c2, code_embeds_output = self.embed_stuff_for_lstm(code_indices, code_wembs, self.code_seq_embedder2)
            code_embeds = torch.cat([code_embeds1, code_embeds2], dim=2) #(batch, num_lines, total_lstm_layers * 2 * lstm_dim)
            code_embeds_c = torch.cat([code_embeds_c1, code_embeds_c2], dim=2) #(batch, num_lines, total_lstm_layers * 2 * lstm_dim)
            text_embeds2, text_embeds_c2, text_embeds_output = self.embed_stuff_for_lstm(text_indices, text_wembs, self.text_seq_embedder2)
            text_embeds = torch.cat([text_embeds1, text_embeds2], dim=2)
            text_embeds_c = torch.cat([text_embeds_c1, text_embeds_c2], dim=2)

            # Concatenate everything
            combo = torch.cat([
                    text_embeds,
                    code_embeds,
                    msg_embeds.unsqueeze(1).expand(-1, num_lines, -1),
                ], dim=2)
            combo = self.combo_mlp(combo)

        else:
            raise NotImplementedError


        # LSTM on top (line level)
        line_seq_hidden, _ = self.line_seq_embedder(combo) #line_seq_hidden: (batch, num_lines, 2*lstm_dim)


        all_enc_stuff = [combo, line_seq_hidden]
        all_enc_stuff += [gold_code_line_stuff]
        all_enc_stuff += [code_embeds, code_embeds_c, code_indices, code_embeds_output]
        if self.c_model.type == "code_only":
            return all_enc_stuff

        all_enc_stuff += [msg_embeds, msg_embeds_c, msg_indices, msg_embeds_output]
        if self.c_model.type == "code_compiler":
            return all_enc_stuff

        else: #"code_compiler_text"
            all_enc_stuff += [text_embeds, text_embeds_c, text_indices, text_embeds_output]
            return all_enc_stuff



    def forward_localize(self, batch, all_enc_stuff):
        if self.c_model.type == "code_only":
            combo, line_seq_hidden, gold_code_line_stuff, code_embeds, code_embeds_c, code_indices, code_embeds_output = all_enc_stuff
        elif self.c_model.type == "code_compiler":
            combo, line_seq_hidden, gold_code_line_stuff, code_embeds, code_embeds_c, code_indices, code_embeds_output, msg_embeds, msg_embeds_c, msg_indices, msg_embeds_output = all_enc_stuff
        else: #"code_compiler_text"
            combo, line_seq_hidden, gold_code_line_stuff, code_embeds, code_embeds_c, code_indices, code_embeds_output, msg_embeds, msg_embeds_c, msg_indices, msg_embeds_output, text_embeds, text_embeds_c, text_indices, text_embeds_output = all_enc_stuff

        # Compute logits
        final_input = line_seq_hidden
        if self.add_residual:
            final_input = torch.cat([combo, final_input], dim=2)
        final = self.final_mlp(final_input).squeeze(2)

        label = [ex.gold_linenos for ex in batch]
        label = try_gpu(torch.tensor(label)) #label: (batch, num_lines)
        label = label.float() / label.sum(dim=1, keepdim=True).float()
        localization_label = label
        localization_out = final

        return localization_out, localization_label


    def forward_edit(self, batch, all_enc_stuff, train_mode=True, beam_size=1, edit_lineno_specified=None):
        self.beam_size = beam_size

        if self.c_model.type == "code_only":
            combo, line_seq_hidden, gold_code_line_stuff, code_embeds, code_embeds_c, code_indices, code_embeds_output = all_enc_stuff
        elif self.c_model.type == "code_compiler":
            combo, line_seq_hidden, gold_code_line_stuff, code_embeds, code_embeds_c, code_indices, code_embeds_output, msg_embeds, msg_embeds_c, msg_indices, msg_embeds_output = all_enc_stuff
        else: #"code_compiler_text"
            combo, line_seq_hidden, gold_code_line_stuff, code_embeds, code_embeds_c, code_indices, code_embeds_output, msg_embeds, msg_embeds_c, msg_indices, msg_embeds_output, text_embeds, text_embeds_c, text_indices, text_embeds_output = all_enc_stuff


        # Get gold_linenos (line# for which we need gold for edit)
        # for edit model, ex.edit_linenos is one hot
        gold_linenos = []
        if edit_lineno_specified is None:
            for ex in batch:
                assert sum(ex.edit_linenos) == 1
                gold_linenos.append(ex.edit_linenos)
        else: #should be dev or test setting
            gold_code_line_stuff = []
            assert len(batch) == len(edit_lineno_specified) == 1

            #update src line and tgt line info
            for b_id, ex in enumerate(batch):
                lidx = edit_lineno_specified[b_id]
                tgt_seq = ex.gold_code_lines[lidx].code

                gold_code_line_stuff.append(ex.gold_code_lines[lidx].code_idxs)

                edit_linenos = [0] * len(ex.edit_linenos)
                edit_linenos[lidx] = 1
                gold_linenos.append(edit_linenos)
                ex.edit_linenos = edit_linenos
                # print ("ex.edit_linenos", ex.edit_linenos)

                if not self.c_model.decoder_attend_all:
                    if self.c_model.type in ["code_compiler", "code_only"]:
                        src_seq = ex.code_lines[lidx].code
                    else:
                        src_seq = ex.code_lines[lidx].text
                    src_vocab = defaultdict(int)
                    for tok in src_seq:
                        src_vocab[tok] += 1
                    ex.src_vocab = [PAD, UNK, BOS, EOS] + list(src_vocab.keys())
                    ex.src_vocab_x = {x: i for (i, x) in enumerate(ex.src_vocab)}
                    ex.src_map = [ex.src_vocab_x[tok] for tok in [BOS]+src_seq+[EOS]]

                    align = [UNK_INDEX] #correspond to BOS
                    for tok in tgt_seq:
                        if tok in ex.src_vocab_x:
                            align.append(ex.src_vocab_x[tok])
                        else:
                            align.append(UNK_INDEX)
                    align.append(UNK_INDEX) #correspond to EOS
                    ex.align = align


        gold_linenos = try_gpu(torch.tensor(gold_linenos)) #(batch, num_lines)
        gold_linenos_onehot = gold_linenos.unsqueeze(2).float() #(batch, num_lines, 1)


        def get_oneline_vecs(gold_linenos_onehot, embeds_h, embeds_c, embeds_output, indices):
            # gold_linenos_onehot: #(batch, num_lines, 1)
            # embeds_h & _c: (batch, num_lines, lstm_out_dim)
            # embeds_output: (batch, num_lines, seqlen, dim)
            # indices: #(batch, num_lines, s_len)
            oneline_h = (embeds_h * gold_linenos_onehot).sum(dim=1, keepdim=False) #(batch, dim)
            oneline_c = (embeds_c * gold_linenos_onehot).sum(dim=1, keepdim=False) #(batch, dim)
            oneline_enc_output = (embeds_output * gold_linenos_onehot.unsqueeze(3)).sum(dim=1, keepdim=False) #(batch, seqlen, dim)
            oneline_indices = (indices.float() * gold_linenos_onehot).sum(dim=1, keepdim=False) #(batch, s_len)
            ## Arrange
            # h_n and c_n: should be (lstm_layers * 2, batch, lstm_dim)
            # oneline_enc_output: should be (s_len, batch, lstm_dim*2)
            b_size = oneline_h.size(0)
            lstm_dim = self.c_model.tok_seq_embed.lstm_dim
            oneline_h = oneline_h.view(b_size, -1, lstm_dim).transpose(0,1)
            oneline_c = oneline_c.view(b_size, -1, lstm_dim).transpose(0,1)
            oneline_enc_output = oneline_enc_output.transpose(0,1)
            oneline_indices = oneline_indices.transpose(0,1)
            return oneline_h, oneline_c, oneline_enc_output, oneline_indices

        def format_tensor_length(in_tensor, true_slen): #in_tensor: (batch, ?, seqlen) or (batch, ?, seqlen, dim). format to true_slen
            sizes = list(in_tensor.size())
            orig_slen = sizes[2]
            if orig_slen > true_slen:
                ret = in_tensor[:, :, :true_slen]
            else:
                sizes[2] = true_slen
                ret = try_gpu(torch.zeros(*sizes).fill_(PAD_INDEX))
                ret[:, :, :orig_slen] = in_tensor
            return Variable(ret).float()


        ## prepare src_vocabs, src_map
        def prep_src_map_one_line(true_slen): #true_slen = text_indices.size(2) or code_indices.size(2)
            # src_map: (src_len, batch, svocab)
            src_vocabs = []
            src_map = [] #list(batch, _slen)
            for ex in batch:
                src_vocabs.append(ex.src_vocab)
                if ex.src_map != []:
                    src_map.append(torch.tensor(ex.src_map))
                else:
                    src_map.append(torch.tensor([0]))  # for server

            _src_map = try_gpu(pad_sequence(src_map)) #(_slen, batch)
            _slen, _b_size = _src_map.size()
            max_id = torch.max(_src_map)
            _src_map = make_one_hot(_src_map.view(-1), max_id+1).view(_slen, _b_size, max_id+1) #(_slen, batch, svocab)
            src_map = try_gpu(torch.zeros((true_slen, _b_size, max_id+1)).fill_(PAD_INDEX))
            src_map[:_slen] = _src_map
            src_map = Variable(src_map)
            return src_vocabs, src_map

        def prep_src_map_all_lines(true_slen): #true_slen = text_indices.size(2) or code_indices.size(2)
            # src_map: (`num_lines` * src_len, batch, svocab)
            src_vocabs = []
            src_map = [] #list(batch, `num_lines`(w/ msg? w/text?), _slen). NOTE: w/ msg? w/text? is handled within ex.src_map
            for ex in batch:
                src_vocabs.append(ex.src_vocab)
                __src_map = [] #(1+num_lines, true_slen)
                for src_seq in ex.src_map: #src_seq: list(vocab_index)
                    curr_len = len(src_seq)
                    if curr_len > true_slen:
                        padded_src_seq = src_seq[:true_slen]
                    else:
                        padded_src_seq = src_seq + [0]*(true_slen-curr_len)
                    __src_map.append(padded_src_seq)
                src_map.append(__src_map)
            src_map = try_gpu(torch.tensor(src_map)).transpose(0,1).transpose(1,2).contiguous() #(`num_lines`, true_slen, batch)
            _num_seqs, _slen, _b_size = src_map.size()
            max_id = torch.max(src_map)
            src_map = make_one_hot(src_map.view(-1), max_id+1).view(-1, _b_size, max_id+1) #(`num_lines` * true_slen, batch, svocab)
            src_map = Variable(src_map)
            return src_vocabs, src_map


        _b_size = line_seq_hidden.size(0)
        if self.c_model.type == "code_compiler":
            for_dec_init_h = self.bridge_h(torch.cat([code_embeds, line_seq_hidden], dim=2)) #(batch, num_lines, lstm_out_dim = lstm_layers *2 * lstm_dim)
            for_dec_init_c = self.bridge_c(torch.cat([code_embeds_c, line_seq_hidden], dim=2))
            dec_init_h, dec_init_c, code_oneline_enc_output, code_oneline_indices = get_oneline_vecs(gold_linenos_onehot, for_dec_init_h, for_dec_init_c, code_embeds_output, code_indices)

            if not self.c_model.decoder_attend_all:
                true_slen = code_indices.size(2)
                src_vocabs, src_map = prep_src_map_one_line(true_slen)
                packed_dec_input = [dec_init_h, dec_init_c, code_oneline_enc_output, code_oneline_indices, gold_code_line_stuff]
            else:
                true_slen = code_indices.size(2)
                src_vocabs, src_map = prep_src_map_all_lines(true_slen)
                _msg_indices = format_tensor_length(msg_indices.unsqueeze(1), true_slen)
                _msg_embeds_output = format_tensor_length(msg_embeds_output.unsqueeze(1), true_slen)
                all_src_indices = torch.cat([_msg_indices.long(), code_indices.long()], dim=1) ##(batch, 1+num_lines, seqlen)
                all_enc_output  = torch.cat([_msg_embeds_output, code_embeds_output], dim=1) #(batch, 1+num_lines, seqlen, dim)
                _, _, _, __dim = all_enc_output.size()
                all_src_indices = all_src_indices.view(_b_size, -1).transpose(0,1) #(1+num_lines * seqlen, batch)
                all_enc_output  = all_enc_output.view(_b_size, -1, __dim).transpose(0,1) #(1+num_lines * seqlen, batch, dim)
                packed_dec_input = [dec_init_h, dec_init_c, all_enc_output, all_src_indices, gold_code_line_stuff]

        elif self.c_model.type == "code_only":
            for_dec_init_h = self.bridge_h(torch.cat([code_embeds, line_seq_hidden], dim=2)) #(batch, num_lines, lstm_out_dim = lstm_layers *2 * lstm_dim)
            for_dec_init_c = self.bridge_c(torch.cat([code_embeds_c, line_seq_hidden], dim=2))
            dec_init_h, dec_init_c, code_oneline_enc_output, code_oneline_indices = get_oneline_vecs(gold_linenos_onehot, for_dec_init_h, for_dec_init_c, code_embeds_output, code_indices)

            if not self.c_model.decoder_attend_all:
                true_slen = code_indices.size(2)
                src_vocabs, src_map = prep_src_map_one_line(true_slen)
                packed_dec_input = [dec_init_h, dec_init_c, code_oneline_enc_output, code_oneline_indices, gold_code_line_stuff]
            else:
                true_slen = code_indices.size(2)
                src_vocabs, src_map = prep_src_map_all_lines(true_slen)
                all_src_indices = torch.cat([code_indices.long()], dim=1) ##(batch, num_lines, seqlen)
                all_enc_output  = torch.cat([code_embeds_output], dim=1) #(batch, num_lines, seqlen, dim)
                _, _, _, __dim = all_enc_output.size()
                all_src_indices = all_src_indices.view(_b_size, -1).transpose(0,1) #(num_lines * seqlen, batch)
                all_enc_output  = all_enc_output.view(_b_size, -1, __dim).transpose(0,1) #(num_lines * seqlen, batch)
                packed_dec_input = [dec_init_h, dec_init_c, all_enc_output, all_src_indices, gold_code_line_stuff]

        else: #"code_compiler_text"
            for_dec_init_h = self.bridge_h(torch.cat([text_embeds, line_seq_hidden], dim=2)) #(batch, num_lines, lstm_out_dim = lstm_layers *2 * lstm_dim)
            for_dec_init_c = self.bridge_c(torch.cat([text_embeds_c, line_seq_hidden], dim=2))
            dec_init_h, dec_init_c, text_oneline_enc_output, text_oneline_indices = get_oneline_vecs(gold_linenos_onehot, for_dec_init_h, for_dec_init_c, text_embeds_output, text_indices)
            if not self.c_model.decoder_attend_all:
                true_slen = text_indices.size(2)
                src_vocabs, src_map = prep_src_map_one_line(true_slen)
                packed_dec_input = [dec_init_h, dec_init_c, text_oneline_enc_output, text_oneline_indices, gold_code_line_stuff]
            else:
                true_slen = max(text_indices.size(2), code_indices.size(2))
                src_vocabs, src_map = prep_src_map_all_lines(true_slen)
                code_indices = format_tensor_length(code_indices, true_slen)
                code_embeds_output = format_tensor_length(code_embeds_output, true_slen)
                text_indices = format_tensor_length(text_oneline_indices.transpose(0,1).unsqueeze(1), true_slen) #(batch,1,true_slen)
                text_embeds_output = format_tensor_length(text_oneline_enc_output.transpose(0,1).unsqueeze(1), true_slen) #(batch,1,true_slen,dim)

                all_src_indices = torch.cat([code_indices, text_indices], dim=1).long() ##(batch, 2*num_lines, seqlen)
                all_enc_output  = torch.cat([code_embeds_output, text_embeds_output], dim=1) #(batch, 2*num_lines, seqlen, dim)

                _msg_indices = format_tensor_length(msg_indices.unsqueeze(1), true_slen)
                _msg_embeds_output = format_tensor_length(msg_embeds_output.unsqueeze(1), true_slen)
                all_src_indices = torch.cat([_msg_indices.long(), all_src_indices], dim=1) ##(batch, 1+2*num_lines, seqlen)
                all_enc_output  = torch.cat([_msg_embeds_output, all_enc_output], dim=1) #(batch, 1+2*num_lines, seqlen, dim)
                _, _, _, __dim = all_enc_output.size()
                all_src_indices = all_src_indices.view(_b_size, -1).transpose(0,1) #(1+2*num_lines * seqlen, batch)
                all_enc_output  = all_enc_output.view(_b_size, -1, __dim).transpose(0,1) #(1+2*num_lines * seqlen, batch, dim)
                packed_dec_input = [dec_init_h, dec_init_c, all_enc_output, all_src_indices, gold_code_line_stuff]


        dec_output, padded_gold_code_line = self.forward_helper_decode(batch, packed_dec_input, src_vocabs, src_map, train_mode) #(max_seq_len, batch_size, vocab_size)

        edit_out = dec_output
        edit_label = padded_gold_code_line

        return edit_out, edit_label



    def forward_helper_decode(self, batch, packed_dec_input, src_vocabs, src_map, train_mode):
        """
        used inside of forward
        """
        enc_h, enc_c, enc_output, src_indices, gold_code_line_stuff = packed_dec_input # src_indices is already padded
        batch_size = enc_output.size(1)
        # code_line_stuff & gold_code_line_stuff: list [batch, seqlen]
        gold_code_line = [torch.tensor([BOS_INDEX]+ seq +[EOS_INDEX]) for seq in gold_code_line_stuff]
        padded_gold_code_line = try_gpu(pad_sequence(gold_code_line)) #(seq, B)

        gold_max_seq_len = len(padded_gold_code_line)

        # Ensure dim of hidden_state can be fed into Decoder
        # enc_h: (layers * directions, B, enc_dim) -> (layers, B, 2*enc_dim)
        _, _, enc_dim = enc_h.size()
        enc_h = enc_h.transpose(0,1).view(batch_size, self.c_model.tok_seq_embed.lstm_layers, -1) #(B, layers, dim)
        enc_h = enc_h.transpose(0,1).contiguous() #(layers, B, dim)
        enc_c = enc_c.transpose(0,1).view(batch_size, self.c_model.tok_seq_embed.lstm_layers, -1)
        enc_c = enc_c.transpose(0,1).contiguous()
        hidden = (enc_h, enc_c)

        if train_mode: #training mode
            output_tokens = padded_gold_code_line #(seq, B)
            teacher_forcing_ratio = float(train_mode)
        else: #test time
            output_tokens = try_gpu(torch.zeros((max(100, gold_max_seq_len), batch_size)).long().fill_(BOS_INDEX)) #assume max length is 100
            teacher_forcing_ratio = 0
        input_tokens = src_indices #(seq, B)

        vocab_size = self.decoder.output_size

        #tensor to store decoder outputs. initialized to value 0
        max_seq_len = len(output_tokens)
        dynamic_vocab_size = vocab_size + src_map.size(2) #NEW
        outputs = try_gpu(torch.zeros(max_seq_len, batch_size, dynamic_vocab_size))

        #first input to the decoder is the <bos> tokens
        output = output_tokens[0,:]

        # Create mask (used for attention)
        mask = self.create_mask(input_tokens).transpose(0,1) #(B, seq)

        # extra_feed = final_vec.unsqueeze(0) #(1, B, dim)
        extra_feed = None
        context_vec = None #NEW (initialization)

        if train_mode:
            # Step through the length of the output sequence one token at a time
            # Teacher forcing is used to assist training
            for t in range(1, max_seq_len):
                output = output.unsqueeze(0) #(1,B)
                output, hidden, attn, context_vec = self._decode_and_generate_one_step(output, hidden, enc_output, mask, context_vec, extra_feed, src_vocabs, src_map, beam_size=1, collapse=False)
                #output: prob scores (batch_size, dynamic_vocab_size)

                outputs[t] = output
                teacher_force = (random.random() < teacher_forcing_ratio)
                top1 = output.max(1)[1]
                output = (output_tokens[t] if teacher_force else top1)

            return outputs, padded_gold_code_line

        else: #test time
            if self.beam_size > 1:
                # print ("beam search decode with beam size %d" % beam_size)
                allHyp, allScores = self.beam_decode(hidden, enc_output, mask, extra_feed, src_vocabs, src_map)
                return (allHyp, allScores), None
            else:
                # print ("greedy_decode")
                outputs = self.greedy_decode(hidden, enc_output, mask, extra_feed, src_vocabs, src_map)
                return outputs, None

    def _decode_and_generate_one_step(self, decoder_in, hidden_state, memory_bank, mask, context_vec, extra_feed, src_vocabs, src_map, beam_size=1, collapse=False, batch_offset=None):
        # decoder_in: [t_len=1, batch or beamxbatch]
        # hidden_state:  (h, c)
        # memory_bank(encoder_outputs): [s_len, batch, dim]
        # mask: [batch, s_len]
        # extra_feed: [1, batch, dim]
        # src_vocabs: list(src_vocab)
        # src_map: (src_len, batch, svocab)

        # Turn any copied words into UNKs.
        decoder_in = decoder_in.masked_fill(
            decoder_in.gt(len(self.meta.vocab) - 1), UNK_INDEX
        )
        dec_out, hidden, dec_attn, context_vec = self.decoder(
            decoder_in, hidden_state, memory_bank, mask, context_vec, extra_feed
        ) #dec_out: (batch or beamxbatch, hidden_size)
        attn = dec_attn["copy"] #(batch or beamxbatch, 1, s_len)
        scores = self.copy_generator(dec_out,
                                          attn.view(-1, attn.size(2)),
                                          src_map) #(beamxbatch, s_vocab_size)
        scores = scores.view(beam_size, -1, scores.size(-1)).transpose(0,1) #(batch_size, beam_size, dynamic_vocab_size)
        if collapse: #if test, collapse. if train, do not
            scores = collapse_copy_scores(
                    scores,
                    None,
                    self.meta.vocab_x,
                    src_vocabs,
                    batch_dim=0,
                    batch_offset=batch_offset
                ) #(batch_size, beam_size, dynamic_vocab_size)
        scores = scores.view(-1, scores.size(-1)) #(batch_size x beam_size, dynamic_vocab_size)

        return scores, hidden, attn, context_vec


    def greedy_decode(self, enc_hidden, enc_output, mask, extra_feed, src_vocabs, src_map):
        # enc_output: (s_len, batch, dim)
        # enc_hidden: (enc_h, enc_c).  enc_h: (layers, batch, 2*enc_dim)
        # mask: (batch, s_len)
        # extra_feed: (1, batch, dim)
        batch_size = enc_output.size(1)
        hidden = enc_hidden
        output_tokens = try_gpu(torch.zeros(100, batch_size)).long().fill_(BOS_INDEX)

        output = output_tokens[0,:]
        vocab_size = self.decoder.output_size
        dynamic_vocab_size = vocab_size + src_map.size(2) #NEW
        outputs = try_gpu(torch.zeros(100, batch_size, dynamic_vocab_size))
        context_vec = None #TODO
        for t in range(1, 100):
            output = output.unsqueeze(0) #(1,B)
            output, hidden, attn, context_vec = self._decode_and_generate_one_step(output, hidden, enc_output, mask, context_vec, extra_feed, src_vocabs, src_map, beam_size=1, collapse=True)
            #output: prob scores (batch_size, dynamic_vocab_size)

            outputs[t] = output
            teacher_force = 0
            top1 = output.max(1)[1]
            output = (output_tokens[t] if teacher_force else top1)

            if (output.item() == EOS_INDEX):
                return outputs
        return outputs


    def beam_decode(self, enc_hidden, enc_output, mask, extra_feed, src_vocabs, src_map): ## Only used in dev/test
        # github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/decode.py
        # enc_output: (s_len, batch, dim)
        # enc_hidden: (enc_h, enc_c).  enc_h: (layers, batch, 2*enc_dim)
        # mask: (batch, s_len)
        # extra_feed: (1, batch, dim)
        # src_vocabs: list(src_vocab)
        # src_map: (src_len, batch, svocab)

        beam_size = self.beam_size
        batch_size = enc_output.size(1)

        src_map = Variable(src_map.data.repeat(1, beam_size, 1)) #(s_len, beam_size * batch, svocab)
        enc_h_t, enc_c_t = enc_hidden
        dec_states = [
            Variable(enc_h_t.data.repeat(1, beam_size, 1)),
            Variable(enc_c_t.data.repeat(1, beam_size, 1))
        ]
        memory_bank = Variable(enc_output.data.repeat(1, beam_size, 1)) #(s_len, beam_size * batch, dim)
        memory_mask = Variable(mask.data.repeat(beam_size, 1)) #(beam_size * batch, s_len)
        extra_feed_ = Variable(extra_feed.data.repeat(1, beam_size, 1)) if extra_feed is not None else None #(1, beam_size * batch, dim)

        # (0) pt 2, prep the beam object
        max_length = 100
        beam = BeamSearch(
            beam_size,
            n_best=beam_size,
            batch_size=batch_size,
            global_scorer=None,
            pad=PAD_INDEX,
            eos=EOS_INDEX,
            bos=BOS_INDEX,
            min_length=0,
            ratio=0,
            max_length=max_length,
            mb_device="cuda",
            return_attention=False,
            stepwise_penalty=False,
            block_ngram_repeat=0,
            exclusion_tokens=[],
            memory_lengths=None)

        context_vec = None
        for step in range(max_length):
            input = beam.current_predictions.view(1, -1) #(1, B x parallel_paths)
            scores, (trg_h_t, trg_c_t), attn, context_vec = self._decode_and_generate_one_step(input, (dec_states[0], dec_states[1]), memory_bank, memory_mask, context_vec, extra_feed_, src_vocabs, src_map, beam_size, collapse=True, batch_offset=beam._batch_offset)
            dec_states = (trg_h_t, trg_c_t)
            log_probs = scores.log()

            beam.advance(log_probs, attn)
            any_beam_is_finished = beam.is_finished.any()
            if any_beam_is_finished:
                beam.update_finished()
                if beam.done:
                    break

            select_indices = beam.current_origin

            if any_beam_is_finished:
                # Reorder states.
                memory_bank = memory_bank.index_select(1, select_indices)
                memory_mask = memory_mask.index_select(0, select_indices)
                src_map = src_map.index_select(1, select_indices)
                extra_feed_ = extra_feed_.index_select(1, select_indices) if extra_feed_ is not None else None
                dec_states = (
                    dec_states[0].index_select(1, select_indices),
                    dec_states[1].index_select(1, select_indices)
                )

        allHyp, allScores = beam.predictions, beam.scores #allHyp: list[batch, nbest, toks].  allScores: list[batch, nbest]
        return allHyp, allScores


    def get_stuffs_to_embed(self, batch):
        """
        Extract:
        - text (batch, num_lines, *)
        - code (batch, num_lines, *)
        if compiler
            - msg (batch, *)
            - err_lineno (batch)
        """
        text_stuff = []
        code_stuff = []
        gold_code_line_stuff = []
        if self.c_model.type == "code_compiler_text":
            _code_slen = 0
            _msg_slen = 0
            msg_stuff = []
            err_linenos = []
            for ex in batch:
                text_stuff_sub = []
                code_stuff_sub = []
                for line in ex.code_lines:
                    text_stuff_sub.append(line.text_idxs)
                    code_stuff_sub.append(line.code_idxs)
                    _cur_code_slen = len(line.code_idxs) + 2 #BOS, EOS
                    _code_slen = _cur_code_slen if _cur_code_slen > _code_slen else _code_slen
                text_stuff.append(text_stuff_sub)
                code_stuff.append(code_stuff_sub)
                lidx = np.argmax(np.array(ex.edit_linenos))
                gold_code_line_stuff.append(ex.gold_code_lines[lidx].code_idxs)

                msg_stuff.append(ex.err_line.msg_idxs)
                err_linenos.append(ex.err_line.lineno)
                _cur_msg_slen = len(ex.err_line.msg_idxs) + 2 #BOS, EOS
                _msg_slen = _cur_msg_slen if _cur_msg_slen > _msg_slen else _msg_slen
            return text_stuff, (code_stuff, _code_slen), (msg_stuff, _msg_slen), err_linenos, gold_code_line_stuff

        elif self.c_model.type == "code_compiler":
            _code_slen = 0
            _msg_slen = 0
            msg_stuff = []
            err_linenos = []
            for ex in batch:
                code_stuff_sub = []
                for line in ex.code_lines:
                    code_stuff_sub.append(line.code_idxs)
                    _cur_code_slen = len(line.code_idxs) + 2 #BOS, EOS
                    _code_slen = _cur_code_slen if _cur_code_slen > _code_slen else _code_slen
                code_stuff.append(code_stuff_sub)
                lidx = np.argmax(np.array(ex.edit_linenos))
                gold_code_line_stuff.append(ex.gold_code_lines[lidx].code_idxs)

                msg_stuff.append(ex.err_line.msg_idxs)
                err_linenos.append(ex.err_line.lineno)
                _cur_msg_slen = len(ex.err_line.msg_idxs) + 2 #BOS, EOS
                _msg_slen = _cur_msg_slen if _cur_msg_slen > _msg_slen else _msg_slen
            return (code_stuff, _code_slen), (msg_stuff, _msg_slen), err_linenos, gold_code_line_stuff #CHANGE-FOR-EDITOR

        elif self.c_model.type == "code_only":
            _code_slen = 0
            err_linenos = []
            for ex in batch:
                code_stuff_sub = []
                for line in ex.code_lines:
                    code_stuff_sub.append(line.code_idxs)
                    _cur_code_slen = len(line.code_idxs) + 2 #BOS, EOS
                    _code_slen = _cur_code_slen if _cur_code_slen > _code_slen else _code_slen
                code_stuff.append(code_stuff_sub)
                lidx = np.argmax(np.array(ex.edit_linenos))
                gold_code_line_stuff.append(ex.gold_code_lines[lidx].code_idxs)
            return (code_stuff, _code_slen), gold_code_line_stuff

        else:
            raise NotImplementedError


    def embed_stuff_for_wembs(self, stuff, true_slen):
        """
        Embed the sequences.
        """
        def pad_sequence_with_length(sequences, true_slen, batch_first=False, padding_value=0):
            max_size = sequences[0].size()
            trailing_dims = max_size[1:]
            max_len = max([s.size(0) for s in sequences])
            max_len = max_len if true_slen==0 else true_slen
            if batch_first:
                out_dims = (len(sequences), max_len) + trailing_dims
            else:
                out_dims = (max_len, len(sequences)) + trailing_dims
            out_tensor = sequences[0].data.new(*out_dims).fill_(padding_value)
            for i, tensor in enumerate(sequences):
                length = tensor.size(0)
                # use index notation to prevent duplicate references to the tensor
                if batch_first:
                    out_tensor[i, :length, ...] = tensor
                else:
                    out_tensor[:length, i, ...] = tensor
            return out_tensor

        _b_size = len(stuff)
        if isinstance(stuff[0][0], list): #2d
            all_seq_indices = []
            for batch in stuff:
                for seq in batch:
                    token_indices = [BOS_INDEX] + seq + [EOS_INDEX]
                    token_indices = torch.tensor(token_indices)
                    all_seq_indices.append(token_indices)
            padded_token_indices = try_gpu(pad_sequence_with_length(all_seq_indices, true_slen)) #(seq_len, batch * num_lines)
        else:
            all_seq_indices = []
            for seq in stuff:
                token_indices = [BOS_INDEX] + seq + [EOS_INDEX]
                token_indices = torch.tensor(token_indices)
                all_seq_indices.append(token_indices)
            padded_token_indices = try_gpu(pad_sequence_with_length(all_seq_indices, true_slen))  #(seq_len, batch)

        embedded_tokens = self.tok_embedder(padded_token_indices)
        embedded_tokens = self.dropout(embedded_tokens) #(seq_len, `batch`, embed_dim)

        _seqlen_, _b_, _dim_ = embedded_tokens.size()
        padded_token_indices = padded_token_indices.transpose(0,1) #(`batch`, seqlen)
        padded_token_indices = padded_token_indices.view(_b_size, -1, _seqlen_).squeeze(1) #(batch, seqlen) or (batch, num_lines, seqlen)

        embedded_tokens = embedded_tokens.transpose(0,1) #(`batch`, seqlen, embed_dim)
        embedded_tokens = embedded_tokens.view(_b_size, -1, _seqlen_, _dim_).squeeze(1) #(batch, seqlen, dim) or (batch, num_lines, seqlen, dim)

        return padded_token_indices.contiguous(), embedded_tokens.contiguous()


    def embed_stuff_for_lstm(self, inp_indices, inp_wembs, seq_embedder):
        """
        Embed the sequences.
        """
        if len(inp_wembs.size()) == 4:
            _2d_flag = True
            _b_size, _num_lines, true_slen, _wembdim = inp_wembs.size()
            inp_wembs = inp_wembs.view(-1, true_slen, _wembdim) #(batch * num_lines, true_slen, dim)
            inp_indices = inp_indices.view(-1, true_slen) #(batch * num_lines, true_slen)
        else:
            _2d_flag = False
            _b_size, true_slen, _wembdim = inp_wembs.size()
            _num_lines = None

        inp_mask = (inp_indices != PAD_INDEX) #(`batch`, true_slen), where `batch` means batch * num_lines if 2d
        inp_length = inp_mask.sum(dim=1) #(`batch`,)
        inp_wembs = inp_wembs.transpose(0,1) #(true_slen, `batch`, dim)

        #LSTM
        seq_input, hx, rev_order, mask = prepare_rnn_seq(inp_wembs, inp_length, hx=None, masks=None, batch_first=False)
        seq_output, hn = seq_embedder(seq_input)
        lstm_output, (h_n, c_n) = recover_rnn_seq(seq_output, rev_order, hx=hn, batch_first=False) #lstm_output:(slen, `batch`, 2*lstm_dim). h_n:(lstm_layers * 2, `batch`, lstm_dim)
        _num_seqs = lstm_output.size(1)
        lstm_output_fullypadded = try_gpu(torch.zeros((true_slen, _num_seqs, lstm_output.size(2)))).float()
        lstm_output_fullypadded[:lstm_output.size(0)] = lstm_output #(true_slen, `batch`, 2*lstm_dim)

        # Arrange dimenstions
        lstm_out_h = h_n.transpose(0, 1).reshape(_num_seqs, -1) #(`batch`, lstm_layers * 2 * lstm_dim)
        lstm_out_c = c_n.transpose(0, 1).reshape(_num_seqs, -1)
        lstm_output = lstm_output_fullypadded.transpose(0, 1).reshape(_num_seqs, true_slen, -1) #(`batch`, true_slen, lstm_dim *2)

        if _2d_flag:
            lstm_out_h = lstm_out_h.view(_b_size, _num_lines, -1) #(batch, num_lines, lstm_layers * 2 * lstm_dim)
            lstm_out_c = lstm_out_c.view(_b_size, _num_lines, -1) #(batch, num_lines, lstm_layers * 2 * lstm_dim)
            lstm_output = lstm_output.view(_b_size, _num_lines, true_slen, -1) #(batch, num_lines, true_slen, lstm_dim *2)

        return lstm_out_h, lstm_out_c, lstm_output



    def positional_encoding(self, err_linenos, num_lines):
        """
        Return the positional embedding tensor.

        Args:
            err_linenos: (batch,)
                List of ints indicating the error line.
                Each value x satisfies 0 <= x < num_lines.
            num_lines: int
        Returns:
            (batch, num_lines, pos_embed_dim)
        """
        batch_size = len(err_linenos)
        err_linenos = torch.tensor(err_linenos)

        # offsets: (batch, num_lines)
        # yay broadcasting magic
        offsets = torch.arange(num_lines).unsqueeze(0) - err_linenos.unsqueeze(1)
        offsets = try_gpu(offsets.float())

        # Build arguments for sine and cosine
        coeffs = try_gpu(torch.arange(self.pos_embed_dim / 2.))
        coeffs = torch.pow(10000., - coeffs / self.pos_embed_dim)
        # (batch, num_lines, pos_embed_dim / 2)
        arguments = offsets.unsqueeze(2) * coeffs
        result = torch.cat([torch.sin(arguments), torch.cos(arguments)], dim=2)
        return result


    def get_loss_localization(self, logit, label, batch):
        """
        Args:
            logit: (batch, num_lines)
                Output from forward(batch)
            label: (batch, num_lines)
            batch: list[Example]
        Returns:
            a scalar tensor
        """
        loss = cross_entropy_after_probsum(logit, label.float())
        return loss

    def get_loss_edit(self, dec_output, padded_gold_code_line, batch, force_copy_loss=None):
        """
        Args:
            dec_output: prob scores (seqlen, batch, dynamic_vocab_size)
                Output from forward(batch)
            padded_gold_code_line: (seqlen, batch) #pad value is 0
            batch: list[Example]
        Returns:
            a scalar tensor
        """
        true_tlen, b_size, dynamic_vocab_size = dec_output.size()
        scores = dec_output.view(-1, dynamic_vocab_size) #(tlen * batch, dynamic_vocab_size)
        target = padded_gold_code_line.view(-1) #(tlen * batch, )

        # prepare align: (tlen * batch, )
        align = [] #list(batch, tlen)
        for ex in batch:
            align.append(torch.tensor(ex.align).long())

        _align = try_gpu(pad_sequence(align, padding_value=UNK_INDEX)) #(_tlen, batch)
        _tlen = _align.size(0)
        align = try_gpu(torch.zeros((true_tlen, b_size)).long().fill_(UNK_INDEX))
        align[:_tlen] = _align
        align = align.view(-1) #(tlen * batch, )
        align = Variable(align)

        loss = self.copy_generator_loss(scores, align, target, b_size, force_copy_loss)

        return loss #(batch,)


    def get_pred_localization(self, logit, batch, train_mode=True):
        """
        Args:
            logit: (batch, num_lines)
                Output from forward(batch)
            batch: list[Example]
        Returns:
            predictions
        """
        return torch.argmax(logit, dim=1)

    def get_pred_edit(self, dec_output, batch, train_mode=True, retAllHyp=False):
        """
        Args:
            dec_output: (seqlen, batch, vocab_size)
                Output from forward(batch)
            batch: list[Example]
        Returns:
            predictions
        """
        if train_mode:
            return torch.argmax(dec_output, dim=2, keepdim=False).transpose(0,1) #(batch, seqlen)
        else:
            if isinstance(dec_output, tuple): #get result from beam search
                allHyp, allScores = dec_output
                if not retAllHyp:
                    return [torch.tensor(hyps[0]) for hyps in allHyp] #hyps[0]: list of w_id
                else:
                    return [allHyp, allScores]
            else: #greedy decode
                return torch.argmax(dec_output, dim=2, keepdim=False).transpose(0,1) #(batch, seqlen)
