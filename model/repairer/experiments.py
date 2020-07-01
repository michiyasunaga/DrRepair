import random, re

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
from tqdm import tqdm

from repairer.data import create_dataset
from repairer.metadata import Metadata
from repairer.model import create_model
from repairer.utils import Stats, try_gpu

import socket
print(socket.gethostname())


class Experiment(object):

    def __init__(self, config, outputter, load_prefix=None, seed=None, force_cpu=False):
        self.config = config
        self.add_text = False #used when adapting a model without text (pseudocode) to one with text
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
        self.outputter = outputter
        self.meta = Metadata(config)
        if load_prefix:
            if load_prefix.endswith("-add_text"):
                self.add_text = True
                load_prefix = load_prefix.rstrip("-add_text")
            self.load_metadata(load_prefix)
        self.dataset = create_dataset(self.config, self.meta)
        self.create_model()
        if load_prefix:
            if self.add_text:
                self.load_model_add_text(load_prefix)
            else:
                self.load_model(load_prefix, force_cpu=force_cpu)
        else:
            self.model.initialize(self.config, self.meta)

    def close(self):
        pass

    def create_model(self):
        config = self.config
        self.model = create_model(config, self.meta)
        self.model = try_gpu(self.model)
        self.optimizer = optim.Adam(self.model.parameters(),
                lr=config.train.learning_rate,
                weight_decay=config.train.l2_reg)

    def load_metadata(self, prefix):
        print('Loading metadata from {}.meta'.format(prefix))
        self.meta.load(prefix + '.meta')

    def load_model(self, prefix, force_cpu=False):
        print('Loading model from {}.model'.format(prefix))
        if force_cpu:
            state_dict = torch.load(prefix + '.model', map_location='cpu')
        else:
            state_dict = torch.load(prefix + '.model')
        self.model.load_state_dict(state_dict)

    def load_model_add_text(self, prefix):
        print('Loading model from {}.model (add_text)'.format(prefix))
        loaded_state_dict = torch.load(prefix + '.model')
        model_state = self.model.state_dict()
        state_name_to_keep = set([
                r"tok_embedder.weight",
                r"combo_mlp_pos_enc\.[.]*",
                r"code_[.]*",
                r"msg_[.]*",
                r"line_seq_embedder\.[.]*",
                r"final_mlp\.[.]*",
                r"bridge_[.]*",
                r"decoder.embedding\.[.]*",
                r"decoder.lstm\.[.]*",
                r"copy_generator.linear\.[.]*"
            ])
        for name, param in loaded_state_dict.items():
            keep = False
            for pttn_to_keep in state_name_to_keep:
                if re.match(pttn_to_keep, name):
                    param = param.data
                    model_state[name].copy_(param)
                    keep = True
                    break
            if keep:
                print ('params: keeping', name)
            else:
                print ('params: not keeping', name)


    ################################
    # Train loop

    def train(self):
        config = self.config

        # Initial save
        self.outputter.save_model(self.meta.step, self.model, self.meta)

        max_steps = config.timing.max_steps
        progress_bar = tqdm(total=max_steps, desc='TRAIN', mininterval=20)
        progress_bar.update(self.meta.step)

        train_iter = None
        train_stats = Stats()

        while self.meta.step < max_steps:
            self.meta.step += 1
            progress_bar.update()

            train_batch = None if train_iter is None else next(train_iter, None)
            if train_batch is None:
                self.dataset.init_iter('train')
                train_iter = self.dataset.get_iter('train')
                train_batch = next(train_iter)
                assert train_batch is not None, 'No training data'

            stats = self.process_batch(train_batch, train=True)
            train_stats.add(stats)

            # Log the aggregate statistics
            if self.meta.step % config.timing.log_freq == 0:
                print('TRAIN @ {}: {}'.format(self.meta.step, train_stats))
                train_stats.log(self.outputter.tb_logger, self.meta.step, 'pn_train_')
                train_stats = Stats()

            # Save the model
            if self.meta.step % config.timing.save_freq == 0 or self.meta.step == max_steps-1:
                self.outputter.save_model(self.meta.step, self.model, self.meta)

            # Evaluate
            if self.meta.step % config.timing.eval_freq == 0:
                dev_stats = Stats()
                self.dataset.init_iter('dev')
                fout_filename = 'pred.dev.{}'.format(self.meta.step)
                fout = open(self.outputter.get_path(fout_filename), 'w')
                for dev_batch in tqdm(self.dataset.get_iter('dev'), desc='DEV', mininterval=60):
                    stats = self.process_batch(dev_batch, train=False, fout=fout)
                    dev_stats.add(stats)
                if fout: fout.close()
                print('DEV @ {}: {}'.format(self.meta.step, dev_stats))
                dev_stats.log(self.outputter.tb_logger, self.meta.step, 'pn_dev_')

        progress_bar.close()

    def test(self):
        test_stats = Stats()
        self.dataset.init_iter('test')
        fout_filename = 'pred.test.{}'.format(self.meta.step)
        with open(self.outputter.get_path(fout_filename), 'w') as fout:
            for test_batch in tqdm(self.dataset.get_iter('test'), desc='TEST', mininterval=60):
                stats = self.process_batch(test_batch, train=False, fout=fout)
                test_stats.add(stats)
        print('TEST @ {}: {}'.format(self.meta.step, test_stats))
        test_stats.log(self.outputter.tb_logger, self.meta.step, 'pn_test_')

    ################################
    # Processing a batch

    def process_batch(self, batch, train=False, fout=None):
        """
        Process a batch of examples.

        Args:
            batch (list[???])
            train (bool): Whether it is training or testing
            fout (file): Dump predictions to this file
        Returns:
            a Stats containing the model's statistics
        """
        stats = Stats()

        if train:
            self.optimizer.zero_grad()
            self.model.train()
            # Forward pass
            all_enc_stuff = self.model.forward_encode(batch)
            logit_localize, label_localize = self.model.forward_localize(batch, all_enc_stuff)
            logit_edit, label_edit = self.model.forward_edit(batch, all_enc_stuff, train_mode=0.5)

            loss_localize = self.model.get_loss_localization(logit_localize, label_localize, batch)
            loss_edit     = self.model.get_loss_edit(logit_edit, label_edit, batch)

            loss_localize = loss_localize.sum() / len(batch)
            loss_edit = loss_edit.sum() / len(batch)
            mean_loss = 0.5 * (loss_localize + loss_edit)

            stats.n = len(batch)
            stats.n_batches = 1
            stats.loss_localize = float(loss_localize)
            stats.loss_edit = float(loss_edit)
            # Evaluate
            pred_localize = self.model.get_pred_localization(logit_localize, batch)
            pred_edit = self.model.get_pred_edit(logit_edit, batch)
            self.dataset.evaluate(batch, [logit_localize, logit_edit, None], [pred_localize, pred_edit, None], stats, self.config.data.task, fout)
            # Gradient
            if mean_loss.requires_grad:
                mean_loss.backward()
                stats.grad_norm = clip_grad_norm_(
                    self.model.parameters(),
                    self.config.train.gradient_clip
                )
                self.optimizer.step()
        else:
            self.model.eval()
            with torch.no_grad():

                assert len(batch) == 1
                all_enc_stuff = self.model.forward_encode(batch)
                #localize
                logit_localize, label_localize = self.model.forward_localize(batch, all_enc_stuff)
                pred_localize = self.model.get_pred_localization(logit_localize, batch)
                pred_lineno = pred_localize[0].item() #one scalar

                #edit
                logit_edit1, label_edit1 = self.model.forward_edit(batch, all_enc_stuff, train_mode=False, beam_size=10) #follow the edit_lineno given
                logit_edit2, label_edit2 = self.model.forward_edit(batch, all_enc_stuff, train_mode=False, beam_size=10, edit_lineno_specified=[pred_lineno]) #follow the edit_lineno predicted
                pred_edit1 = self.model.get_pred_edit(logit_edit1, batch, train_mode=False)
                pred_edit2 = self.model.get_pred_edit(logit_edit2, batch, train_mode=False)

                stats.n = len(batch)
                stats.n_batches = 1
                stats.loss_localize = 0.
                stats.loss_edit = 0.
                logit_edit1 = torch.zeros([stats.n, 1])
                logit_edit2 = torch.zeros([stats.n, 1])

                # Evaluate
                self.dataset.evaluate(batch, [logit_localize, logit_edit1, logit_edit2], [pred_localize, pred_edit1, pred_edit2], stats, self.config.data.task, fout)

        return stats

    ################################
    # Server mode

    def serve(self, port):
        from repairer.server import start_server
        self.model.eval()
        start_server(self, host="0.0.0.0", port=port)
