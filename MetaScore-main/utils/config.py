import yaml
from easydict import EasyDict

def load_config(config_path):
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    return EasyDict(config_dict)