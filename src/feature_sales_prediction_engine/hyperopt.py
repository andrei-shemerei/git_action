from xgboost import XGBRegressor 
from sklearn.model_selection import RandomizedSearchCV

from .global_var import PARAM_SET



class Hyperopt:
    """
    A class to perform hyperparameter tuning for XGBoost using Hyperopt.

    Methods:
    --------
    tune(X_train, Y_train, X_valid, Y_valid):
        Tunes hyperparameters of the XGBoost model using Hyperopt.
    
    Example usage:
    --------------
    hyperopt = Hyperopt()
    best_xgb_model, best_params = hyperopt.tune(X_train, Y_train, X_valid, Y_valid)
    """

    def __init__(self, params=PARAM_SET):
 
        self.random_search = RandomizedSearchCV(
            estimator=XGBRegressor(), 
            param_distributions=params, 
            n_iter=50,              
            scoring='neg_root_mean_squared_error',
            cv=3,                   
            verbose=1,              
            n_jobs=1,
            random_state=20
        )

    def tune(self, X_train, Y_train, X_valid, Y_valid):
        """
        Perform hyperparameter tuning for XGBoost using Hyperopt.

        Parameters:
        -----------
        X_train : pd.DataFrame
            Training feature matrix.
        Y_train : pd.Series
            Training target vector.
        X_valid : pd.DataFrame
            Validation feature matrix.
        Y_valid : pd.Series
            Validation target vector.

        Returns:
        --------
        The best madel and params
        """
        self.random_search.fit(X_train, Y_train,
                               eval_set=[(X_valid, Y_valid)],
                               verbose=True)
        
        best_xgb_model = self.random_search.best_estimator_
        best_params = self.random_search.best_params_

        return best_xgb_model, best_params

    

        




