# Outputter manages the outdir directory
import os
import logging

try:
    from tensorboardX import SummaryWriter
except ImportError:
    logging.warning('tensorboardX is not supported')
    SummaryWriter = None

import torch

from repairer.configs import dump_config


class DummySummaryWriter(object):
    def __init__(self, *args, **kwargs):
        pass
    def add_scalar(self, *args, **kwargs):
        print('TENSORBOARD: add_scalar({}, {})'.format(args, kwargs))

if SummaryWriter is None:
    SummaryWriter = DummySummaryWriter


class Outputter(object):
    """
    Outputter manages the output directory.

    Directory structure:
    - config.json
    - [step].model: torch state dict
    - [step].meta: human-readable metadata
    - tensorboard/*
    """

    def __init__(self, config, basedir, force_outdir=None):
        self.config = config
        self.outdir = self._get_outdir(basedir, force_outdir)
        print('Output directory: {}'.format(self.outdir))
        # Dump the config to config.json
        dump_config(config, os.path.join(self.outdir, 'config.json'))
        # Tensorboard logger
        self.tb_logger = SummaryWriter(os.path.join(self.outdir, 'tensorboard'))

    def _get_outdir(self, basedir, force_outdir=None):
        assert os.path.isdir(basedir), \
                'basedir is not a directory: {}'.format(basedir)
        if force_outdir:
            outdir = os.path.join(
                basedir,
                force_outdir,
            )
            assert os.path.isdir(outdir), \
                    'forced outdir is not a directory: {}'.format(outdir)
        else:
            execs = [
                int(filename.split('.')[0])
                for filename in os.listdir(basedir)
                if filename.split('.')[0].isdigit()
            ]
            outdir = os.path.join(
                basedir,
                str(0 if not execs else max(execs) + 1) + '.exec',
            )
            os.makedirs(outdir)
        return outdir

    def get_path(self, filename):
        return os.path.join(self.outdir, filename)

    def save_model(self, step, model, meta):
        print('Saving model to checkpoint {}'.format(step))
        meta_path = os.path.join(self.outdir, '{}.meta'.format(step))
        meta.save(meta_path)
        state_dict = model.state_dict()
        path = os.path.join(self.outdir, '{}.model'.format(step))
        torch.save(state_dict, path)
