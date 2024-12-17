import os
import yaml

def config_reader(config_file, data_section):
    """
    Simple config reader
    """
    base_dir = os.path.dirname(__file__)
    config_path = os.path.join(base_dir, config_file)  

    with open(config_path, 'r', encoding='utf-8') as file:
        config = yaml.safe_load(file)
        return config[data_section]
    
