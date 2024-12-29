from . import global_var
from .utils import config_reader
from .etl_process import ETL
from .feature_extraction import main_pipeline
from .validation_schema import Validator

__version__ = '0.0.9'


# Убедимся, что все глобальные переменные инициализируются автоматически
__all__ = [
    "global_var",
    "etl_process",
    "feature_extraction",
    "hyperopt",
    "validation_schema",
    "config_reader",
]

def init():
    """
    Функция инициализации глобальных переменных.
    """
    # Проверка на инициализацию переменных
    if not global_var.SHOP_DUPLICATE_SET or not global_var.CORRECT_CITY_NAME:
        raise ValueError("Глобальные переменные не были загружены корректно!")


init()