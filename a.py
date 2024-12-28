import sys
sys.path.append(r"C:\Users\Asus\AppData\Local\Programs\Python\Python311\Lib\site-packages")
from feature_sales_prediction_engine import ETL




path_to_data = r'/data'  # Specify the path to your data
etl = ETL(path_to_data)
data = etl.process()
