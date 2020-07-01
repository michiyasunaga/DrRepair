from collections import Counter, namedtuple, defaultdict
import glob, os
import json
import torch

import numpy as np
np.random.seed(73619)

from repairer.data.base import Dataset
from repairer.data.err_example import ErrExample, CodeLine, ErrLine
from repairer.utils import (
    batch_iter, PAD, PAD_INDEX, UNK, UNK_INDEX, BOS, BOS_INDEX, EOS, EOS_INDEX,
    tokenize_err_msg,
)

import itertools


def eval_print_helper(ex, pred_label, logit, fout):
    """
    prepare what to print for one example
      ex:          ErrExample (or ErrExampleTestcase)
      logit:       list of float
    """
    logit_localize, logit_edit = logit
    pred_localize, pred_edit = pred_label

    program_id = '{:04d}-{}-{}'.format(ex.info['index'], ex.info['probid'], ex.info['subid'])
    print (program_id, file=fout)

    max_text_len = 50
    max_code_len = 50
    def prepare_lines_print(text_indent, text_str_noindt, _max_len, wrap_indent=3): #initial text_indent, text_str_noindt
        text_str = text_indent*"  " + text_str_noindt
        text_to_print = []
        if len(text_str) <= _max_len:
            text_to_print.append(text_str)
        else:
            text_str_print = text_str[:_max_len]
            text_to_print.append(text_str_print)
            text_str_noindt = text_str[_max_len:]
            text_indent += wrap_indent
            text_str = text_indent*"  " + text_str_noindt
            while len(text_str) > _max_len:
                text_str_print = text_str[:_max_len]
                text_to_print.append(text_str_print)
                text_str_noindt = text_str[_max_len:]
                text_str = text_indent*"  " + text_str_noindt
            text_to_print.append(text_str)
        return text_to_print



    gold_label = {i: None for i, indi in enumerate(ex.gold_linenos) if indi==1} #CHANGE-FOR-MULTI
    pred1, pred2, pred3 = list((-np.array(logit_localize)).argsort()[:3])

    pred_code = pred_edit #assume pred_label is already a string

    assert pred_localize == np.argmax(np.array(ex.edit_linenos)) ##20200110 - for dev, we do edit on the localized line

    assert pred1 == pred_localize
    for code_line in ex.code_lines:
        lineno = code_line.lineno
        text_indent = code_line.indent
        text_str_noindt = " ".join(code_line.text) #no indent
        code_indent = code_line.indent
        code_str_noindt = " ".join(code_line.code) #no indent
        is_error = ""
        if lineno in gold_label:
            is_error = "Err. Gold: {}".format(" ".join(ex.gold_code_lines[lineno].code))
        is_pred = ""
        if lineno == pred1: is_pred = "Pred 1**"
        if lineno == pred2: is_pred = "Pred 2*"
        if lineno == pred3: is_pred = "Pred 3"
        text_to_print = prepare_lines_print(text_indent, text_str_noindt, max_text_len)
        code_to_print = prepare_lines_print(code_indent, code_str_noindt, max_code_len)
        is_error_to_print = prepare_lines_print(0, is_error, max_code_len, wrap_indent=8)
        if lineno == pred1:
            is_error_to_print += prepare_lines_print(0, "Edit: {}".format(pred_code), 50, wrap_indent=8)
        _text = text_to_print.pop(0)
        _code = code_to_print.pop(0)
        _is_err = is_error_to_print.pop(0)
        print ("{:>3}  {:<{width1}}  {:<{width2}}   {:<8}  {}".format(str(lineno), _text, _code, is_pred, _is_err, width1=max_text_len, width2=max_code_len), file=fout)
        for _text, _code, _is_err in itertools.zip_longest(text_to_print, code_to_print, is_error_to_print):
            if _text is None: _text = ""
            if _code is None: _code = ""
            if _is_err is None: _is_err = ""
            print ("{:>3}  {:<{width1}}  {:<{width2}}   {:<8}  {}".format("", _text, _code, "", _is_err, width1=max_text_len, width2=max_code_len), file=fout)

    print ("localization (pred1) correct?: %d" %(pred_localize in gold_label), file=fout)
    gold_code = " ".join(ex.gold_code_lines[pred_localize].code)
    print ("edit correct?: %d" %(pred_code==gold_code), file=fout)


    print ("compiler err msg: %s" %(" ".join(ex.err_line.msg)), file=fout)
    print ("compiler err line#: %d" %(ex.err_line.lineno), file=fout)

    print ("", file=fout)
    print ("", file=fout)
    fout.flush()



class ErrDataset(Dataset):
    """
    Dataset of compilation errors.

    The vocab for both text and code should be pre-computed.
    """

    def __init__(self, config, meta):
        super().__init__(config, meta)
        self.task = config.data.task
        self.config_top = config
        assert self.task == "err-compiler"
        if 'test_batch_size' in config.train:
            self.test_batch_size = config.train.test_batch_size
        else:
            self.test_batch_size = self.batch_size
        # Delegate init_iter and get_iter to other classes
        # based on the dataset format.
        self._data = {}
        for name in config.data.splits:
            fmt = config.data.splits[name].format
            if fmt == 'substitute':
                data = SubstituteErrData(name, config.data.splits[name], self.config_top)
            elif fmt == 'vanilla':
                data = VanillaErrData(name, config.data.splits[name], self.config_top)
            else:
                raise ValueError('Unknown format for {}: {}'.format(name, fmt))
            self._data[name] = data
        # Either import vocab from meta, or load vocab from file
        if hasattr(meta, 'vocab'):
            print('Loading vocab from meta')
            self.vocab = meta.vocab[:]
        else:
            print('Initialize vocab ...')
            self.vocab = [PAD, UNK, BOS, EOS]
            with open(config.data.vocab.path) as fin:
                self.vocab += json.load(fin).keys()
            meta.vocab = self.vocab[:]
        print('Vocab size: {}'.format(len(self.vocab)))
        self.vocab_x = {x: i for (i, x) in enumerate(self.vocab)}
        meta.vocab_x = self.vocab_x

    def init_iter(self, name):
        """
        Initialize the iterator for the specified data split.
        """
        self._data[name].init_iter(self)

    def get_iter(self, name):
        """
        Get the iterator over the specified data split.
        """
        return self._data[name].get_iter()


    def evaluate(self, batch, logit, prediction, stats, data_task, fout=None):
        """
        Evaluate the predictions and write the results to stats.
        """
        logit_localize, logit_edit1, logit_edit2  = logit
        pred_localize, pred_edit1, pred_edit2 = prediction
        for i, ex in enumerate(batch):
            #err-localize
            pred_label = pred_localize[i]
            pred_label = pred_label.item() #one number
            gold_label = [j for j, indi in enumerate(ex.gold_linenos) if indi==1]
            localize_success = (pred_label in gold_label)
            stats.accuracy_localize += localize_success
            pred_label_localize = pred_label

            #err-edit 1 (if use given edit line)
            pred_label = pred_edit1[i]
            pred_toks = []
            for idx in pred_label:
                if idx < len(self.vocab):
                    tok = self.vocab[idx]
                else:
                    tok = ex.src_vocab[idx - len(self.vocab)]
                if tok in [EOS]: break
                if tok in [PAD, BOS]: continue
                pred_toks.append(tok)
            pred_label = " ".join(pred_toks) #pred_code
            lidx = np.argmax(np.array(ex.edit_linenos))
            gold_label = " ".join(ex.gold_code_lines[lidx].code)
            if gold_label == pred_label:
                stats.accuracy_edit1 += 1
            pred_label_edit = pred_label

            #err-edit 2 (if use localized line)
            if pred_edit2 is not None:
                pred_label = pred_edit2[i]
                pred_toks = []
                for idx in pred_label:
                    if idx < len(self.vocab):
                        tok = self.vocab[idx]
                    else:
                        tok = ex.src_vocab[idx - len(self.vocab)]
                    if tok in [EOS]: break
                    if tok in [PAD, BOS]: continue
                    pred_toks.append(tok)
                pred_label = " ".join(pred_toks) #pred_code
                lidx = np.argmax(np.array(ex.edit_linenos))
                gold_label = " ".join(ex.gold_code_lines[lidx].code)
                if gold_label == pred_label:
                    stats.accuracy_edit2 += 1
                    if localize_success:
                        stats.accuracy_repair += 1
                pred_label_edit = pred_label

            if fout:
                _pred_label = [pred_label_localize, pred_label_edit]
                _logit = [logit_localize[i].tolist(), logit_edit1[i].tolist()]
                eval_print_helper(ex, _pred_label, _logit, fout)


    def s_parse_request(self, q):
        comment = q["comment"] if "comment" in q else ""
        return [ErrExample.deserialize(q, self.config_top, self.vocab_x)], comment

    def s_parse_request_for_loss(self, q):
        preds = q["preds"]
        code_lines = q["code_lines"]
        pred_lineno = np.argmax(q["gold_linenos"])
        qs = []
        for pred_code in preds:
            gold_code_lines = code_lines[:]
            item = list(gold_code_lines[pred_lineno])
            item[1] = pred_code  #gold_code_lines[pred_lineno]: (text, code, indent)
            gold_code_lines[pred_lineno] = item
            q = {
                'info': q["info"],
                'code_lines': code_lines,
                'err_line': q["err_line"],
                "gold_linenos": q["gold_linenos"],
                "gold_code_lines": gold_code_lines,
            }
            qs.append(q)
        return [ErrExample.deserialize(q, self.config_top, self.vocab_x) for q in qs]

    def s_generate_response(self, q, batch, logit, prediction, comment=None):
        logit_localize = logit.tolist() if logit is not None else None

        pred_localize, pred_edit = prediction
        pred_localize = pred_localize.tolist() if pred_localize is not None else None

        nbest_preds = None
        if pred_edit is not None:
            ex = batch[0]
            prediction, scores = pred_edit
            if isinstance(prediction[0], torch.Tensor): #only top1
                pred_labels = [list(prediction[0].cpu().numpy())]
            else:
                pred_labels = prediction[0]
            nbest_preds = []
            for pred_label in pred_labels: # the nbest preds
                pred_toks = []
                for idx in pred_label:
                    if idx < len(self.vocab):
                        tok = self.vocab[idx]
                    else:
                        tok = ex.src_vocab[idx - len(self.vocab)]
                    if tok in [EOS]: break
                    if tok in [PAD, BOS]: continue
                    pred_toks.append(tok)
                pred_label = " ".join(pred_toks) #pred_code
                nbest_preds.append(pred_label)

        return {
            'logit_localize': logit_localize,
            'pred_localize': pred_localize, #should be [some_num] b/c B_size=1
            'pred_edit': nbest_preds, #list(pred_code)
        }


################################################


class SubstituteErrData(object):
    def __init__(self, name, config, config_top):
        ## `config` is config_top.data
        self.name = name
        self.config_top = config_top

        ## If SPoC, avoid programs in dev and test
        self.uniqueid_to_avoid = defaultdict(int)
        if config.paths[0].endswith("s1"):
            print ("remove programs in spoc dev & test")
            _dev_dir = config.paths[0].replace("s1", "s5") #e.g. data/err-data-compiler--orig-spoc/s5/*
            _dev_files = glob.glob(_dev_dir)
            assert len(_dev_files) > 0
            for json_fbname in _dev_files:
                if json_fbname.endswith(".json"):
                    _uniqueid = "-".join(json_fbname.rstrip(".json").split("-")[1:]) #e.g. 214A-48303922
                    self.uniqueid_to_avoid[_uniqueid] += 1
            _test_dir = "../raw_data/spoc_data/spoc/test"
            for test_fbname in os.listdir(_test_dir):
                with open(os.path.join(_test_dir, test_fbname)) as in_f:
                    lines = in_f.readlines()[1:] #skip header
                    for line in lines:
                        _probid, _subid = line.split("\t")[4:6]
                        _uniqueid = "%s-%s" % (_probid, _subid)
                        self.uniqueid_to_avoid[_uniqueid] += 1

        ## To avoid big file
        _big_files_info_path = "../data/programs_with_line_of_length_over_80toks.txt"
        self.uniqueid_big_file = defaultdict(int)
        with open(_big_files_info_path) as in_f:
            lines = in_f.readlines()
            assert len(lines) > 0
            for line in lines:
                _uniqueid = "-".join(line.strip().rstrip(".json").split("-")[1:])
                self.uniqueid_big_file[_uniqueid] += 1

        self.filenames = []
        self.orig_err_count = 0
        self.extra_err_count = 0
        for pattern in config.paths:
            filenames = glob.glob(pattern)
            print('Read {} {} filenames from pattern {}'.format(
                len(filenames), name, pattern,
            ))
            for fname in filenames:
                if self.overlap_w_test(fname):
                    continue
                if self.is_too_big(fname):
                    continue
                self.filenames.append(fname)
                if self.is_orig_err(fname):
                    self.orig_err_count += 1
                elif self.is_extra_err(fname):
                    self.extra_err_count += 1
                else:
                    raise NotImplementedError

        self.filenames.sort()
        self.shuffle = config.get_('shuffle', None)

        if self.orig_err_count == 0:
            self.extra_inclusion_prob = 1
        elif self.extra_err_count == 0:
            self.extra_inclusion_prob = 0
        else:
            self.extra_inclusion_prob = min(1, 0.1* float(self.orig_err_count) / (self.extra_err_count+0.00001))
        print ("orig_err_count", self.orig_err_count, "extra_err_count", self.extra_err_count, self.extra_inclusion_prob)


    def is_orig_err(self, filename):
        if "err-data-compiler--orig" in filename: return True
        if "err-data-compiler--auto-corrupt--orig-" in filename: return True
        return False
    def is_extra_err(self, filename):
        return "err-data-compiler--auto-corrupt--codeforce" in filename
    def overlap_w_test(self, filename):
        bfname = os.path.basename(filename)
        _uniqueid = "-".join(bfname.rstrip(".json").split("-")[1:]) #e.g. 214A-48303922
        if _uniqueid in self.uniqueid_to_avoid:
            return True
        else:
            return False
    def is_too_big(self, filename):
        bfname = os.path.basename(filename)
        _uniqueid = "-".join(bfname.rstrip(".json").split("-")[1:]) #e.g. 214A-48303922
        return  _uniqueid in self.uniqueid_big_file


    def init_iter(self, dataset):
        if self.shuffle:
            np.random.shuffle(self.filenames)
        self.itr = (
            batch
            for fname in self.filenames if self.is_orig_err(fname) or np.random.uniform() < self.extra_inclusion_prob
            for batch in SubstituteFileIter(self.name, fname, dataset, self.config_top) if len(batch) > 0
        )

    def get_iter(self):
        return self.itr


class SubstituteFileIter(object):
    """
    Read a JSON file
    """

    def __init__(self, name, filename, dataset, config_top):
        self.config_top = config_top
        self.name = name
        with open(filename) as fin:
            data = json.load(fin)
        # info: dict with the following keys:
        #   index, hitid, workerid, probid, subid
        # Renamed to "info" to prevent confusion with the Metadata class
        self.info = data['meta']
        # print ("{}-{}".format(self.info["probid"], self.info["subid"]))
        # lines: list[dict] where dict has keys
        #   line, text, code, indent
        self.lines = data['lines']
        self.code_lines = []
        self.dummy_lidxs = []
        self.non_dummy_lidxs = []
        for x in self.lines:
            self.code_lines.append(
                CodeLine(
                    lineno = x['line'],
                    text = (x['text'].split() if x['text'] != 'DUMMY' else []),
                    code = x['code'].split() if self.config_top.data.name != "deepfix-style" else x['code_anonymized'].split(),
                    indent = x['indent'],
                    text_idxs = [],
                    code_idxs = [],
                ))
            if x['text'] == 'DUMMY':
                self.dummy_lidxs.append(x['line'])
            else:
                self.non_dummy_lidxs.append(x['line'])
        self.dummy_lidxs = set(self.dummy_lidxs)

        # errors: list[dict] where dict has keys
        #   mod_line, mod_code, err_line, err_msg
        self.errors = data['errors']

        self.dataset = dataset
        if len(self.errors) > 100:
            self.sample_prob = min(float(100) / len(self.errors), 1)
        else:
            self.sample_prob = 1
        self.next_index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.next_index >= len(self.errors):
            raise StopIteration
        examples = []
        batch_size = (
            self.dataset.batch_size if self.name == 'train'
            else self.dataset.test_batch_size
        )

        while (
            len(examples) < batch_size and
            self.next_index < len(self.errors)
        ):
            if self.sample_prob < 1 and np.random.uniform() > self.sample_prob:
                self.next_index += 1; continue

            error = self.errors[self.next_index]
            mod_line_list = error['mod_line'][:]
            mod_code_list = error['mod_code'][:] if self.config_top.data.name != "deepfix-style" else error['mod_code_anonymized'][:]

            _msg = tokenize_err_msg(error['err_msg'])
            if len(_msg) > 80:
                self.next_index += 1
                continue
            err_line = ErrLine(
                lineno = error['err_line'],
                msg = _msg,
                msg_idxs = [],
            )

            ## mutate lines
            mod_code_lines = self.code_lines[:]
            ml_indicators = [0] * len(mod_code_lines)
            for (idx, ml) in enumerate(mod_line_list):
                ml_indicators[ml] = 1
                mod_code_lines[ml] = CodeLine(
                    lineno = ml,
                    text = self.code_lines[ml].text,
                    code = mod_code_list[idx].split(),
                    indent = self.code_lines[ml].indent,
                    text_idxs = self.code_lines[ml].text_idxs,
                    code_idxs = [],
                )

            ## Randomly pick some lines with already correct code and add it
            lidx = np.random.choice(self.non_dummy_lidxs)
            if lidx not in mod_line_list:
                mod_line_list.append(lidx)
                code_ = self.code_lines[lidx].code
                assert isinstance(code_, list)
                code_ = " ".join(code_)
                mod_code_list.append(code_)

            #line for edit
            ml = np.random.choice(mod_line_list)
            edit_indicators = [0] * len(mod_code_lines)
            edit_indicators[ml] = 1
            ex = ErrExample(mod_code_lines, err_line, self.info, ml_indicators, edit_indicators, self.code_lines, self.config_top)
            ex.add_idxs(self.dataset.vocab_x)
            examples.append(ex)

            self.next_index += 1

        return examples


################################################


class VanillaErrData(object):

    def __init__(self, name, config, config_top=None):
        self.name = name
        self.config_top = config_top
        self.path = config.path
        self.data = None
        self.shuffle = config.get_('shuffle', None) #train

    def init_iter(self, dataset):
        if self.data is None:
            self.data = []
            with open(self.path) as fin:
                for line in fin:
                    ex = json.loads(line)
                    ex = ErrExample.deserialize(ex, self.config_top, dataset.vocab_x)

                    gold_linenos_list = [i for (i, indic) in enumerate(ex.gold_linenos) if indic==1]
                    gold_linenos_list = np.array(gold_linenos_list)
                    np.random.shuffle(gold_linenos_list)

                    gold_linenos_onehot = [0] * len(ex.gold_linenos)
                    gold_linenos_onehot[gold_linenos_list[0]] = 1
                    ex.edit_linenos = gold_linenos_onehot
                    ex.add_idxs(dataset.vocab_x)
                    self.data.append(ex)

        if self.shuffle:
            np.random.shuffle(self.data)
        print('Loaded {} {} examples'.format(len(self.data), self.name))
        assert len(self.data) > 0, 'No {} data'.format(self.name)
        batch_size = (
            dataset.batch_size if self.name == 'train'
            else dataset.test_batch_size
        )
        self.itr = batch_iter(self.data, batch_size)

    def get_iter(self):
        return self.itr
