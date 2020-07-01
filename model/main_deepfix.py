#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Main entry for running experiments.

The results will be saved to ./out/
(relative to this file)
"""
import argparse
import json
import os
import sys

from repairer.configs import (
    load_config,
    merge_configs,
    ConfigDict,
)
from repairer.experiments import Experiment
from repairer.outputter import Outputter


BASEDIR = os.path.abspath(os.path.join(
    os.path.dirname(__file__), 'out_deepfix',
))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--load-prefix',
            help='Load a model from this file')
    parser.add_argument('-c', '--config-string',
            help='Additional config (as JSON)')
    parser.add_argument('-o', '--outdir',
            help='Force the output directory')
    parser.add_argument('-s', '--seed', type=int, default=42,
            help='Set seed')
    parser.add_argument('-p', '--port', type=int, default=8080,
            help='Port for the "server" action')
    parser.add_argument('-C', '--force-cpu', action='store_true',
            help='Load a GPU-trained model on CPU')
    parser.add_argument('action', choices=['train', 'test', 'server'])
    parser.add_argument('configs', nargs='+',
            help='Config JSON or YAML files')
    args = parser.parse_args()

    config = {}
    for path in args.configs:
        new_config = load_config(path)
        merge_configs(config, new_config)

    if args.config_string:
        new_config = json.loads(args.config_string)
        merge_configs(config, new_config)

    print(json.dumps(config, indent=2))
    config = ConfigDict(config)

    outputter = Outputter(config, basedir=BASEDIR, force_outdir=args.outdir)
    experiment = Experiment(
        config, outputter, args.load_prefix, args.seed, args.force_cpu
    )

    if args.action == 'train':
        experiment.train()
    elif args.action == 'test':
        experiment.test()
    elif args.action == 'server':
        experiment.serve(args.port)
    else:
        raise ValueError('Unknown action: {}'.format(args.action))


if __name__ == '__main__':
    main()
