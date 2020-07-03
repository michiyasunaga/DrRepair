class Dataset(object):

    def __init__(self, config, meta):
        self.config = config
        self.batch_size = config.train.batch_size

    def init_iter(self, name):
        """
        Initialize the iterator for the specified data split.
        """
        raise NotImplementedError

    def get_iter(self, name):
        """
        Get the iterator over the specified data split.
        """
        raise NotImplementedError

    def evaluate(self, batch, logit, prediction, stats, fout=None):
        """
        Evaluate the predictions and write the results to stats.
        """
        raise NotImplementedError
