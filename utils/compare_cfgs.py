import yaml
from yacs.config import CfgNode as CN

def convert_to_dict(cfg_node, key_list=[]):
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    """ Convert a config node to dictionary """
    if not isinstance(cfg_node, CN):
        if type(cfg_node) not in _VALID_TYPES:
            print("Key {} with value {} is not a valid type; valid types: {}".format(
                ".".join(key_list), type(cfg_node), _VALID_TYPES), )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


def check_cfgs_same(cfg1, cfg2):
    return True
    def print_cfg(cfg, prefix=''):
        for key, value in cfg.items():
            if isinstance(value, CN):
                print_cfg(value, prefix=f'{prefix}{key}.')
            else:
                print(f'{prefix}{key}: {value}')

    # Assume cfg is a CfgNode object
    print_cfg(cfg)