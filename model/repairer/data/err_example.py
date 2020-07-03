from collections import namedtuple, defaultdict
import numpy as np

from repairer.utils import PAD, PAD_INDEX, UNK, UNK_INDEX, BOS, BOS_INDEX, EOS, EOS_INDEX

import sys
sys.path.append("../utils")
from c_tokenizer_mod import C_Tokenizer
c_tokenizer = C_Tokenizer()


CodeLine = namedtuple('CodeLine', [
    'lineno',       # int: line number
    'text',         # list[str]: text tokens (empty for DUMMY lines), no BOS/EOS
    'code',         # list[str]: code tokens, no EOS/BOS
    'indent',       # int: indent level
    'text_idxs',    # list[int]: text token indices (empty list = to be populated)
    'code_idxs',    # list[int]: code token indices (empty list = to be populated)
])
ErrLine = namedtuple('ErrLine', [
    'lineno',       # int: line number as reported by g++
    'msg',          # list[str]: error message tokens
    'msg_idxs',     # list[int]: error token indices (empty list = to be populated)
])


def fix_strings(inp, only_char=False):
    if not only_char:
        res = ""
        temp_string = ""
        inside = False
        for i in range(len(inp)):
            if not inside:
                res += inp[i]
                if inp[i] == "\"":
                    inside = True
                continue
            if inside:
                if inp[i] == "\"":
                    inside = False
                    if len(temp_string) > 2 and temp_string[0] == " " and temp_string[-1] == " ":
                        res += temp_string[1:-1]
                    else:
                        res += temp_string
                    res += "\""
                    temp_string = ""
                else:
                    temp_string += inp[i]
        inp = res
    res = ""
    temp_string = ""
    inside = False
    for i in range(len(inp)):
        if not inside:
            res += inp[i]
            if inp[i] == "\'":
                inside = True
            continue
        if inside:
            if inp[i] == "\'":
                inside = False
                if len(temp_string) > 2 and temp_string[0] == " " and temp_string[-1] == " ":
                    res += temp_string[1:-1]
                else:
                    res += temp_string
                res += "\'"
                temp_string = ""
            else:
                temp_string += inp[i]
    return res



class ErrExample(object):

    def __init__(self, code_lines, err_line, info=None, gold_linenos=None, edit_linenos=None, gold_code_lines=None, config_top=None):
        """
        Args:
            code_lines (list[CodeLine]): the texts and codes
            err_line (ErrLine): error as reported by g++
            info (Optional[Any]): additional data like probid and subid
            gold_lineno (Optional[int]): correct error line (for training data)
        """
        self.info = info
        self.code_lines = code_lines
        self.err_line = err_line
        self.gold_linenos = gold_linenos #indicator vector
        self.edit_linenos = edit_linenos #indicator vector
        self.gold_code_lines = gold_code_lines
        self.config_top = config_top
        assert config_top is not None

    def add_idxs(self, vocab_x):
        """
        Add the token indices (the *_idxs fields)
        to all CodeLine and ErrLine if they are currently empty.
        """
        for code_line in self.code_lines + self.gold_code_lines:
            if not code_line.text_idxs and code_line.text:
                code_line.text_idxs.extend(
                    vocab_x.get(x, UNK_INDEX) for x in code_line.text
                )
            if not code_line.code_idxs and code_line.code:
                code_line.code_idxs.extend(
                    vocab_x.get(x, UNK_INDEX) for x in code_line.code
                )
        if not self.err_line.msg_idxs and self.err_line.msg:
            self.err_line.msg_idxs.extend(
                vocab_x.get(x, UNK_INDEX) for x in self.err_line.msg
            )

        ### Prepare graph
        if self.config_top.model.graph == 0:
            self.graph_mask = None
            self.graph_A = None
        else:
            #  assume we use msg and code
            var_vocab_msg = defaultdict(int)  #identifiers & any diagnostic arguments
            var_vocab_code = defaultdict(int) #identifiers & any diagnostic arguments
            # from msg
            msg_len = len(self.err_line.msg)
            for j, tok in enumerate(self.err_line.msg):
                if tok == "‘":
                    j_copy = j+1
                    while j_copy < msg_len and self.err_line.msg[j_copy] != "’": #diagnostic argument
                        var_candidate = self.err_line.msg[j_copy]
                        var_vocab_msg[var_candidate] += 1
                        j_copy += 1

            # from code
            if self.config_top.data.name == "spoc-style":
                for _lidx in range(len(self.gold_linenos)):
                    src_seq = self.code_lines[_lidx].code
                    src_seq_str = fix_strings(" ".join(src_seq), only_char=True)
                    toks, kinds = c_tokenizer.tokenize(src_seq_str)
                    for (tok, kind) in zip(toks, kinds):
                        if kind == "name" and len(tok) <= 8:
                            var_vocab_code[tok] += 1
            else: #deepfix style i.e. with _<string>_, _<char>_, _<number>_
                to_avoid = c_tokenizer._keywords.union(c_tokenizer._calls).union(c_tokenizer._types).union(c_tokenizer._ops).union(c_tokenizer._includes)
                to_avoid = to_avoid.union(["_<string>_", "_<char>_", "_<number>_", "#include"])
                for _lidx in range(len(self.gold_linenos)):
                    src_seq = self.code_lines[_lidx].code
                    for tok in src_seq:
                        if tok not in to_avoid:
                            var_vocab_code[tok] += 1


            # only keep symbols that appear multiple times
            final_var_vocab = defaultdict(list)
            src_seqs = [self.err_line.msg] + [self.code_lines[_lidx].code for _lidx in range(len(self.gold_linenos))]
            var_vocab_msg.update(var_vocab_code)
            var_vocab = var_vocab_msg
            for seq_id, src_seq in enumerate(src_seqs):
                for tok_id, tok in enumerate(src_seq):
                    if tok in var_vocab:
                        final_var_vocab[tok].append(True)
            final_var_vocab = set([var for var in final_var_vocab if len(final_var_vocab[var]) > 1])
            if len(final_var_vocab) == 0:
                self.graph_mask = None
                self.graph_A = None
            else:
                # graph_mask
                var2node = defaultdict(list)
                node2var = {}
                node_id = 0
                graph_mask = [] #(num_seqs, *)
                for seq_id, src_seq in enumerate(src_seqs):
                    _graph_mask = []
                    _graph_mask.append(0) #<BOS>
                    for tok_id, tok in enumerate(src_seq):
                        if tok in final_var_vocab:
                            var2node[tok].append(node_id)
                            node2var[node_id] = tok
                            node_id += 1
                            _graph_mask.append(1)
                        else:
                            _graph_mask.append(0)
                    _graph_mask.append(0) #<EOS>
                    graph_mask.append(_graph_mask)
                self.graph_mask = graph_mask

                # graph_A
                num_nodes = node_id
                graph_A = np.zeros((num_nodes, num_nodes))
                for node_id in range(num_nodes):
                    var = node2var[node_id]
                    for node_id2 in var2node[var]:
                        graph_A[node_id, node_id2] = 1
                self.graph_A = graph_A


        ### For ptr-gen (editor)
        if self.config_top.model.get_('no_ptr_gen_process'): return

        assert sum(self.edit_linenos) == 1
        lidx = np.argmax(np.array(self.edit_linenos))
        tgt_seq = self.gold_code_lines[lidx].code
        if self.config_top.model.type in ["code_compiler", "code_only"]:
            if not self.config_top.model.decoder_attend_all:
                src_seq = self.code_lines[lidx].code
                src_vocab = defaultdict(int)
                for tok in src_seq:
                    src_vocab[tok] += 1
                self.src_vocab = [PAD, UNK, BOS, EOS] + list(src_vocab.keys())
                self.src_vocab_x = {x: i for (i, x) in enumerate(self.src_vocab)}
                self.src_map = [self.src_vocab_x[tok] for tok in [BOS]+src_seq+[EOS]]
            else:
                src_seqs = [self.code_lines[_lidx].code for _lidx in range(len(self.gold_linenos))]
                if self.config_top.model.type == "code_compiler":
                    src_seqs = [self.err_line.msg] + src_seqs
                src_vocab = defaultdict(int)
                for src_seq in src_seqs:
                    for tok in src_seq:
                        src_vocab[tok] += 1
                self.src_vocab = [PAD, UNK, BOS, EOS] + list(src_vocab.keys())
                self.src_vocab_x = {x: i for (i, x) in enumerate(self.src_vocab)}
                src_map = [] #(num_seqs=1+num_lines, seqlen)
                for src_seq in src_seqs:
                    src_map.append([self.src_vocab_x[tok] for tok in [BOS]+src_seq+[EOS]])
                self.src_map = src_map
        elif self.config_top.model.type == "code_compiler_text":
            if not self.config_top.model.decoder_attend_all:
                src_seq = self.code_lines[lidx].text
                src_vocab = defaultdict(int)
                for tok in src_seq:
                    src_vocab[tok] += 1
                self.src_vocab = [PAD, UNK, BOS, EOS] + list(src_vocab.keys())
                self.src_vocab_x = {x: i for (i, x) in enumerate(self.src_vocab)}
                self.src_map = [self.src_vocab_x[tok] for tok in [BOS]+src_seq+[EOS]]
            else:
                src_seqs = [self.err_line.msg] + [self.code_lines[_lidx].code for _lidx in range(len(self.gold_linenos))] + [self.code_lines[lidx].text]
                src_vocab = defaultdict(int)
                for src_seq in src_seqs:
                    for tok in src_seq:
                        src_vocab[tok] += 1
                self.src_vocab = [PAD, UNK, BOS, EOS] + list(src_vocab.keys())
                self.src_vocab_x = {x: i for (i, x) in enumerate(self.src_vocab)}
                src_map = [] #(num_seqs=1+2*num_lines, seqlen)
                for src_seq in src_seqs:
                    src_map.append([self.src_vocab_x[tok] for tok in [BOS]+src_seq+[EOS]])
                self.src_map = src_map
        else:
            raise NotImplementedError


        align = [UNK_INDEX] #correspond to BOS
        for tok in tgt_seq:
            if tok in self.src_vocab_x:
                align.append(self.src_vocab_x[tok])
            else:
                align.append(UNK_INDEX)
        align.append(UNK_INDEX) #correspond to EOS
        self.align = align




    def serialize(self):
        """
        Create an object that can be serialized to JSON
        """
        return {
            'code_lines': [
                {
                    'text': ' '.join(x.text),
                    'code': ' '.join(x.code),
                    'indent': x.indent,
                } for x in self.code_lines
            ],
            'err_line': {
                'lineno': self.err_line.lineno,
                'msg': ' '.join(self.err_line.msg),
            },
            'info': self.info,
            'gold_linenos': self.gold_linenos,
            'gold_code_lines': [
                {
                    'text': ' '.join(x.text),
                    'code': ' '.join(x.code),
                    'indent': x.indent,
                } for x in self.gold_code_lines
            ],
        }

    @classmethod
    def deserialize(cls, data, config_top, vocab_x=None):
        """
        Create an Example from loaded JSON.
        If vocab_x is provided, also call add_idxs.
        """
        code_lines = []
        for x in data['code_lines']:
            if isinstance(x, list):
                text, code, indent = x
            else:
                text, code, indent = x['text'], x['code'], x['indent']
            code_lines.append(CodeLine(
                lineno = len(code_lines),
                text = text.split(),
                code = code.split(),
                indent = indent,
                text_idxs = [],
                code_idxs = [],
            ))
        err_line = ErrLine(
            lineno = data['err_line']['lineno'],
            msg = data['err_line']['msg'].split(),
            msg_idxs = [],
        )
        gold_code_lines = []
        for x in data['gold_code_lines']:
            if isinstance(x, list):
                text, code, indent = x
            else:
                text, code, indent = x['text'], x['code'], x['indent']
            gold_code_lines.append(CodeLine(
                lineno = len(code_lines),
                text = text.split(),
                code = code.split(),
                indent = indent,
                text_idxs = [],
                code_idxs = [],
            ))
        gold_linenos = data.get('gold_linenos')
        edit_linenos_tmp = [0] * len(gold_linenos)
        edit_linenos_tmp[np.argmax(np.array(gold_linenos))] = 1
        ex = cls(code_lines, err_line, data.get('info'), gold_linenos, edit_linenos_tmp, gold_code_lines, config_top)
        if vocab_x is not None:
            ex.add_idxs(vocab_x)
        return ex
