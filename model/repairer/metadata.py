# Metadata contains miscellaneous information for training
# and for constructing the model
# (e.g., the vocab should be saved here).
# Metadata should be lightweight and serializable.

import pickle


DEFAULT_SAVE_KEYS = ['step', 'vocab']


class Metadata(object):

    ################################
    # Generic operations

    def __init__(self, config):
        self.save_keys = set(DEFAULT_SAVE_KEYS)
        self.step = 0
        self.vocab_x = None

    def save(self, filename):
        to_save = {k: getattr(self, k) for k in self.save_keys}
        with open(filename, 'wb') as fout:
            pickle.dump(to_save, fout)

    def load(self, filename):
        with open(filename, 'rb') as fin:
            loaded = pickle.load(fin)
        for k, v in loaded.items():
            setattr(self, k, v)

    ################################
    # Key-specific operations
