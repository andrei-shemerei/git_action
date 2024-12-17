"""

Collection of configuration variables

"""
from utils import config_reader
CONFIG_FILE=r"config.yaml"


SHOP_DUPLICATE_SET = config_reader(CONFIG_FILE, r"SHOP_DUPLICATE_SET")
CORRECT_CITY_NAME = config_reader(CONFIG_FILE, r"CORRECT_CITY_NAME")

FEATURE_SET = config_reader(CONFIG_FILE, r"FEATURE_SET")
