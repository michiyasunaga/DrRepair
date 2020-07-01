def create_dataset(config, meta):
    if config.data.task == "err-compiler":
        from repairer.data.err_dataset import ErrDataset
        return ErrDataset(config, meta)
    raise ValueError('Unknown data task: {}'.format(config.data.task))
