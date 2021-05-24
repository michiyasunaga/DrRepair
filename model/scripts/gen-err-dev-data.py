#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys, os, shutil, re, argparse, json, random
from collections import defaultdict, Counter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from repairer.data.err_dataset import SubstituteErrData
from repairer.configs import ConfigDict


class DummyDataset(object):
    test_batch_size = 1 #DEBUGed
    batch_size = 1
    vocab_x = {}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('indir')
    parser.add_argument('outfile')
    parser.add_argument('type')
    args = parser.parse_args()

    if args.type == "pretrain-spoc":
        config = ConfigDict({'paths': [args.indir +'/*3A/*9.json']}) #random. 2588 files
        config_top = ConfigDict({'model': ConfigDict({'graph': 0, 'no_ptr_gen_process': 'true'}), 'data': ConfigDict({'name': 'spoc-style'})})

    elif args.type == "spoc":
        config = ConfigDict({'paths': [args.indir +'/*']})
        config_top = ConfigDict({'model': ConfigDict({'graph': 0, 'no_ptr_gen_process': 'true'}), 'data': ConfigDict({'name': 'spoc-style'})})

    elif args.type == "deepfix":
        config = ConfigDict({'paths': [args.indir +'/*/*.json']})
        config_top = ConfigDict({'model': ConfigDict({'graph': 0, 'no_ptr_gen_process': 'true'}), 'data': ConfigDict({'name': 'deepfix-style'})})

    else:
        raise NotImplementedError


    data = SubstituteErrData('dev', config, config_top)
    data.init_iter(DummyDataset())
    itr = data.get_iter()
    with open(args.outfile, 'w') as fout:
        count = 0
        for x in itr:
            if x == []: continue
            x = json.dumps(x[0].serialize(), separators=(',', ':'), ensure_ascii=False)
            print(x, file=fout)
            count += 1
            if count % 1000 == 0:
                print ("{} done".format(count))


if __name__ == '__main__':
    main()
