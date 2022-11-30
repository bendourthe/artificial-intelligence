# --------------------------------------------------------------------------- #
# IMPORTS
# --------------------------------------------------------------------------- #

import yaml


__all__ = ["get_config", "update_config", "to_yaml", "get_easy_dict"]


# --------------------------------------------------------------------------- #
# CLASS DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
class EasyDict(object):

    def __init__(self, opt):

        self.opt = opt

    def __getattribute__(self, name):
        if name == 'opt' or name.startswith('_') or name not in self.opt:
            return object.__getattribute__(self, name)
        else:
            return self.opt[name]

    def __setattr__(self, name, value):
        if name == 'opt':
            object.__setattr__(self, name, value)
        else:
            self.opt[name] = value

    def __getitem__(self, name):
        return self.opt[name]

    def __setitem__(self, name, value):
        self.opt[name] = value

    def __contains__(self, item):
        return self.opt.__repr__()

    def __repr__(self):
        return self.opt.__repr__()

    def keys(self):
        return self.opt.keys()

    def values(self):
        return self.opt.values()

    def items(self):
        return self.opt.items()

    def __len__(self):
        return len(self.opt)


# --------------------------------------------------------------------------- #
# METHODS DEFINITION
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
def get_easy_dict(d):
    new_dict = d

    for k, v in d.items():
        if isinstance(v, dict):
            new_dict[k] = get_easy_dict(v)
        else:
            new_dict[k] = v

    return EasyDict(new_dict)


# --------------------------------------------------------------------------- #
def resolve_expression(config):
    if type(config) is dict:
        new_config = {}
        for k, v in config.items():
            if type(v) is str and v.startswith('!!python'):
                v = eval(v[8:])
            elif type(v) is dict:
                v = resolve_expression(v)
            new_config[k] = v
        config = new_config

    return config


# --------------------------------------------------------------------------- #
def get_config(config_file, config_names=[]):
    with open(config_file) as f:
        config = resolve_expression(yaml.load(f, Loader=yaml.FullLoader))

    if type(config_names) == str:
        return get_easy_dict(config[config_names])

    while len(config_names) != 0:
        config_name = config_names.pop(0)
        if config_name not in config:
            raise ValueError(f'Invalid config name: {config_name}')
        config = config[config_name]

    return get_easy_dict(config)


# --------------------------------------------------------------------------- #
def update_config(config, args):
    if args is None:
        return
    if hasattr(args, '__dict__'):
        args = args.__dict__
    for arg, val in args.items():
        if arg in config and val is not None:
            config[arg] = val

    for _, val in config.items():
        if isinstance(val, dict) or isinstance(val, EasyDict):
            update_config(val, args)


# --------------------------------------------------------------------------- #
def resolve_tuple(d):
    ret = {}
    for k, v in d.items():
        if isinstance(v, tuple):
            ret[k] = list(v)
        elif isinstance(v, dict):
            ret[k] = resolve_tuple(v)
        else:
            ret[k] = v

    return ret


# --------------------------------------------------------------------------- #
def to_yaml(path, d, list_one_line=True):
    d = resolve_tuple(d)
    if list_one_line:
        data_str = yaml.dump(d)
    else:
        data_str = yaml.dump(d, default_flow_style=False)
    with open(path, 'w') as f:
        f.write(data_str)
