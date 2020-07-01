import copy
import json
import logging

try:
    import yaml
except ImportError:
    logging.warning('yaml is not supported')


def load_config(filename):
    if filename.endswith('.json'):
        with open(filename) as fin:
            return json.load(fin)
    elif filename.endswith('.yaml') or filename.endswith('.yml'):
        with open(filename) as fin:
            return yaml.load(fin)
    else:
        raise ValueError('Unknown file type: {}'.format(filename))


def dump_config(config, filename):
    if isinstance(config, ConfigDict):
        config = config.to_vanilla_()
    print('Writing config to {}'.format(filename))
    with open(filename, 'w') as fout:
        json.dump(config, fout, indent=2)
        fout.write('\n')


def merge_configs(base, new, wildcard_key='XXX'):
    """
    Merge the new config (dict) into the base config (dict).
    This modifies base but not new.

    Rules:
    - Look at each key k in the new config.
    - If base[k] does not exist, set base[k] = new[k]
    - If base[k] exists:
        - If base[k] and new[k] are both dicts, do recursive merge.
        - If new[k] is null, remove key k from base.
        - Otherwise, set base[k] = new[k].

    Special Rule:
    - If k is wildcard_key, merge new[k] with base[k'] for all k'
    """
    for key in new:
        base_keys = list(base) if key == wildcard_key else [key]
        for base_key in base_keys:
            if base_key not in base:
                base[base_key] = copy.deepcopy(new[key])
            elif isinstance(base[base_key], dict) and isinstance(new[key], dict):
                merge_configs(base[base_key], new[key], wildcard_key)
            elif new[key] == None:
                del base[base_key]
            else:
                base[base_key] = copy.deepcopy(new[key])


class ConfigDict(object):
    """
    Allow the config to be accessed with dot notation:
    config['epochs'] --> config.epochs
    """

    def __init__(self, data):
        assert isinstance(data, dict)
        self._data = {}
        for key, value in data.items():
            if isinstance(value, dict):
                value = ConfigDict(value)
            elif isinstance(value, list):
                value = ConfigList(value)
            self._data[key] = value

    def __getitem__(self, key):
        return self._data[key]

    def __getattr__(self, key):
        return self._data[key]

    def __iter__(self):
        for key in self._data:
            yield key

    def get_(self, key, value=None):
        return self._data.get(key, value)

    def to_vanilla_(self):
        data = {}
        for key, value in self._data.items():
            if isinstance(value, (ConfigDict, ConfigList)):
                value = value.to_vanilla_()
            data[key] = value
        return data


class ConfigList(object):

    def __init__(self, data):
        assert isinstance(data, list)
        self._data = []
        for value in data:
            if isinstance(value, dict):
                value = ConfigDict(value)
            elif isinstance(value, list):
                value = ConfigList(value)
            self._data.append(value)

    def __getitem__(self, index):
        return self._data[index]

    def __iter__(self):
        for value in self._data:
            yield value

    def to_vanilla_(self):
        data = []
        for value in self._data:
            if isinstance(value, (ConfigDict, ConfigList)):
                value = value.to_vanilla_()
            data.append(value)
        return data
