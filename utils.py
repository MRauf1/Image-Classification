import yaml


def read_config(config_path):
    with open(config_path) as file:
        data = yaml.load(file, Loader = yaml.FullLoader)
        return data
