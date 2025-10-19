import yaml


def load_yaml(path):
    with open(path, 'r') as file:
        try:
            yaml_file = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)

    return yaml_file


def update_yaml(configs, path):

    with open(path, 'w') as file:
        yaml.dump(configs, file)