#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil, re, argparse, json, random, glob
from collections import defaultdict, Counter, OrderedDict


sys.path.append("../utils")
from code_process import tokenize_code, tokenize_err_msg


class Processor(object):
    def __init__(self, args):
        self.args = args
        self.text_vocab = Counter()
        self.code_vocab = Counter()
        self.msg_vocab = Counter()
        self.combined_vocab = Counter()

    def process(self, data):
        text_tokens = set()
        code_tokens = set()
        msg_tokens = set()
        for line in data['lines']:
            if line['text'] != 'DUMMY':
                text_tokens.update(line['text'].split())
            code_tokens.update(line['code'].split())
        for err in data['errors']:
            for mod_code_ in err['mod_code']:
                code_tokens.update(mod_code_.split())
            if 'err_msg' in err:
                msg_tokens.update(tokenize_err_msg(err['err_msg']))

        self.text_vocab.update(text_tokens)
        self.code_vocab.update(code_tokens)
        self.msg_vocab.update(msg_tokens)
        self.combined_vocab.update(text_tokens | code_tokens | msg_tokens)


    def process_spoc_test(self):
        _test_dir = "../raw_data/spoc_data/spoc/test"
        for test_fbname in os.listdir(_test_dir):
            print ("Reading {}".format(os.path.join(_test_dir, test_fbname)))
            with open(os.path.join(_test_dir, test_fbname)) as in_f:
                lines = in_f.readlines()[1:] #skip header
                for line in lines:
                    # _probid, _subid = line.split("\t")[4:6]
                    # _uniqueid = "%s-%s" % (_probid, _subid)
                    _code = line.split("\t")[1]
                    # print (_code)
                    _code_toks = tokenize_code(_code)
                    # print (_code_toks)
                    self.code_vocab.update(_code_toks)
                    self.combined_vocab.update(_code_toks)


    def dump(self):
        with open(os.path.join(self.args.outdir, 'text.vocab'), 'w') as fout:
            json.dump(OrderedDict(self.text_vocab.most_common()), fout,indent=2)
        with open(os.path.join(self.args.outdir, 'code.vocab'), 'w') as fout:
            json.dump(OrderedDict(self.code_vocab.most_common()), fout,indent=2)
        with open(os.path.join(self.args.outdir, 'msg.vocab'), 'w') as fout:
            json.dump(OrderedDict(self.msg_vocab.most_common()), fout,indent=2)
        with open(os.path.join(self.args.outdir, 'combined.vocab'), 'w') as fout:
            json.dump(OrderedDict(self.combined_vocab.most_common()), fout,indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('indir1')
    parser.add_argument('outdir')
    args = parser.parse_args()

    assert os.path.isdir(args.indir1)
    assert os.path.isdir(args.outdir)

    filenames = sorted(glob.glob(os.path.join(args.indir1, '*/*.json')))
    processor = Processor(args)

    processor.process_spoc_test()

    for i, filename in enumerate(filenames):
        if (i + 1) % 100 == 0:
            print('({} / {}) Processing {}'.format(i + 1, len(filenames), filename))
        with open(filename) as fin:
            data = json.load(fin)
        processor.process(data)

    processor.dump()


if __name__ == '__main__':
    main()
