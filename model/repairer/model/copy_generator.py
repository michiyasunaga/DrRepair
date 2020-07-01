import torch
import torch.nn as nn

from repairer.utils import try_gpu, BOS_INDEX, EOS_INDEX, PAD_INDEX, UNK_INDEX


def collapse_copy_scores(scores, batch, tgt_vocab_x, src_vocabs,
                         batch_dim=0, batch_offset=None):
    """
    Given scores from an expanded dictionary
    corresponeding to a batch, sums together copies,
    with a dictionary word when it is ambiguous.

    Args:
        scores: (batch_size, beam_size, dynamic_vocab_size)
    """
    offset = len(tgt_vocab_x)
    for b in range(scores.size(batch_dim)):
        blank = []
        fill = []
        batch_id = batch_offset[b] if batch_offset is not None else b
        # index = batch.indices.data[batch_id]
        # src_vocab = src_vocabs[index]
        src_vocab = src_vocabs[batch_id]
        for i in range(4, len(src_vocab)):
            sw = src_vocab[i]
            if sw in tgt_vocab_x:
                ti = tgt_vocab_x[sw]
                blank.append(offset + i)
                fill.append(ti)
        if blank:
            blank = try_gpu(torch.Tensor(blank).long()) #type_as(batch.indices.data)
            fill = try_gpu(torch.Tensor(fill).long()) #type_as(batch.indices.data)
            score = scores[:, b] if batch_dim == 1 else scores[b] #(beam_size, dynamic_vocab_size)
            score.index_add_(1, fill, score.index_select(1, blank))
            score.index_fill_(1, blank, 1e-10)
    return scores


class CopyGenerator(nn.Module):
    """An implementation of pointer-generator networks
    :cite:`DBLP:journals/corr/SeeLM17`.

    These networks consider copying words
    directly from the source sequence.

    The copy generator is an extended version of the standard
    generator that computes three values.

    * :math:`p_{softmax}` the standard softmax over `tgt_dict`
    * :math:`p(z)` the probability of copying a word from
      the source
    * :math:`p_{copy}` the probility of copying a particular word.
      taken from the attention distribution directly.

    The model returns a distribution over the extend dictionary,
    computed as

    :math:`p(w) = p(z=1)  p_{copy}(w)  +  p(z=0)  p_{softmax}(w)`


    Args:
       input_size (int): size of input representation (hiddim)
       output_size (int): size of output vocabulary
       pad_idx (int)
    """

    def __init__(self, input_size, output_size, pad_idx=PAD_INDEX):
        super(CopyGenerator, self).__init__()
        self.linear = nn.Linear(input_size, output_size)
        self.linear_copy = nn.Linear(input_size, 1)
        self.pad_idx = pad_idx

    def forward(self, hidden, attn, src_map):
        """
        Compute a distribution over the target dictionary
        extended by the dynamic dictionary implied by copying
        source words.

        Args:
           hidden (FloatTensor): hidden outputs ``(tlen x batch or beamxbatch, hiddim)``
           attn (FloatTensor): attn for each ``(tlen x batch or beamxbatch, slen)``
           src_map (FloatTensor):
               A sparse indicator matrix mapping each source word to
               its index in the vocab
               ``(src_len, batch, svocab)``
        Return:
           scores: (tlen x batch or beamxbatch, vocab_size + svocab)
        """

        # CHECKS
        # batch_by_tlen, _ = hidden.size()
        _, slen = attn.size()
        slen_, batch, svocab = src_map.size()
        # aeq(batch_by_tlen, batch_by_tlen_)
        # print ("slen", slen)
        # print ("slen_", slen_)
        assert slen == slen_

        # Original probabilities.
        logits = self.linear(hidden)
        # logits[:, self.pad_idx] = -float('inf')
        prob = torch.softmax(logits, 1) #(tlen x batch, vocab_size)


        # Probability of copying p(z=1) batch.
        p_copy = torch.sigmoid(self.linear_copy(hidden)) #(batch, 1)
        # Probability of not copying: p_{word}(w) * (1 - p(z))
        out_prob = torch.mul(prob, 1 - p_copy) ####(tlen x batch, vocab_size)####

        mul_attn = torch.mul(attn, p_copy) #(tlen x batch, slen)
        copy_prob = torch.bmm(
            mul_attn.view(-1, batch, slen_).transpose(0, 1), #(batch, tlen, slen)
            src_map.transpose(0, 1) #(batch, src_len, svocab)
        ).transpose(0, 1) #(tlen, batch, svocab)
        copy_prob = copy_prob.contiguous().view(-1, svocab) ####(tlen x batch, svocab)####

        scores = torch.cat([out_prob, copy_prob], 1) #(tlen x batch, vocab_size + svocab)
        return scores



class CopyGeneratorLoss(nn.Module):
    """Copy generator criterion."""
    def __init__(self, vocab_size, force_copy=True, unk_index=UNK_INDEX, ignore_index=PAD_INDEX, eps=1e-20):
        super(CopyGeneratorLoss, self).__init__()
        self.force_copy = force_copy
        self.eps = eps
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index #important!
        self.unk_index = unk_index

    def forward(self, scores, align, target, batch_size, force_copy=None):
        """
        Args:
            scores (FloatTensor): ``(tgt_len x batch_size, vocab_size + svocab)``
            align (LongTensor): ``(tgt_len x batch_size)``
            target (LongTensor): ``(tgt_len x batch_size)``
        """

        # probabilities assigned by the model to the gold targets
        vocab_probs = scores.gather(1, target.unsqueeze(1)).squeeze(1) #(tgt_len x batch_size)
        vocab_probs += self.eps  # to avoid -inf logs


        # probability of tokens copied from source
        copy_ix = align.unsqueeze(1) + self.vocab_size #(tgt_len x batch_size, 1)
        copy_tok_probs = scores.gather(1, copy_ix).squeeze(1) #(tgt_len x batch_size)
        # Set scores for unk to 0 and add eps
        copy_tok_probs[align == self.unk_index] = 0

        copy_tok_probs += self.eps  # to avoid -inf logs

        # find the indices in which you do not use the copy mechanism
        non_copy = align == self.unk_index

        if force_copy is None: force_copy = self.force_copy
        if not force_copy:
            non_copy = non_copy | (target != self.unk_index)

        probs = torch.where(
            non_copy, copy_tok_probs + vocab_probs, copy_tok_probs
        ) #(tgt_len x batch_size)


        loss = -probs.log()
        # Drop padding.
        loss[target == self.ignore_index] = 0

        loss = loss.view(-1, batch_size).transpose(0,1) #(batch, t_len)
        denom = (target != self.ignore_index).float().view(-1, batch_size).transpose(0,1).sum(dim=1) #(batch)

        return loss.sum(dim=1) / denom #(batch, )
