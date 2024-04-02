import yaml


def load_config(config_path='config.yaml'):
    """Load the configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config
