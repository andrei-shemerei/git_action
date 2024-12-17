import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error  # For calculation RMSE

class validator:
    """
    A class to perform validation of a machine learning model using time-based data splits.

    This class uses a sliding window approach for model validation, allowing users to assess
    the model's performance on different folds of the dataset.

    Example usage:
    model = RandomForestRegressor()
    validator = Validator(model)
    report = validator.validation(data, start_size_window=1, end_size_of_window=3, number_of_fold=2)

    """
    
    def __init__(self, model):
        self.model = model
    
    def validation(self, data, start_size_window, end_size_of_window, number_of_fold):

        def get_target(data):
             return data['item_cnt_month']

        def get_features(data):
            return data.drop(['item_cnt_month'], axis=1)

        report = []
        
        for i in range(number_of_fold):
      
            size_window = start_size_window + 1 + i * abs(start_size_window - end_size_of_window) / number_of_fold
            print(size_window)
            train_data = data[data.date_block_num < size_window]
            test_data = data[data.date_block_num == size_window]
                           
            X_train = get_features(train_data)
            Y_train = get_target(train_data)

            X_test = get_features(test_data)
            Y_test = get_target(test_data)


            print(i, 'load data')
            self.model.fit(
                        X_train, 
                        Y_train, 
                        eval_set=[(X_train, Y_train), (X_test, Y_test)],
                        verbose=True
            )
            print(i,'fit')
            Y_pred_train = self.model.predict(X_train)
            Y_pred_test = self.model.predict(X_test)
            print(i,'predict')
            train_rmse = mean_squared_error(Y_train, Y_pred_train, squared=False)
            test_rmse = mean_squared_error(Y_test, Y_pred_test, squared=False)
           
            report.append({
                'iteration': i,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse
                })


        report_df = pd.DataFrame(report)
        
        return report_df