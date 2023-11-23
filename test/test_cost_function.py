# -*- coding: utf-8 -*-
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
# Import 
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, log_loss
from src.utils.cost_functions import CostFunctions

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------

if __name__ == "__main__":
    y_true = np.array([3.0, 4.5, 2.5, 6.0, 5.0])
    y_pred = np.array([2.8, 4.2, 2.7, 5.8, 4.9])
    
    mse_result = CostFunctions.mse(y_true, y_pred)
    mae_result = CostFunctions.mae(y_true, y_pred) 
    rmse_result = CostFunctions.rmse(y_true, y_pred)
    mape_result = CostFunctions.mape(y_true, y_pred)
    
    mse_sklearn = mean_squared_error(y_true, y_pred)
    mae_sklearn = mean_absolute_error(y_true, y_pred)    
    
    print(f"Custom MSE: {mse_result:.2f}")
    print(f"Custom MAE: {mae_result:.2f}")
    print(f"Scikit-learn MSE: {mse_sklearn:.2f}")
    print(f"Scikit-learn MAE: {mae_sklearn:.2f}")
    print(f"RMSE: {rmse_result:.2f}")
    print(f"MAPE: {mape_result:.2f}%")
    
    
    predicted_probabilities = np.array([0.7, 0.8, 0.9, 0.6, 0.2])
    actual_labels = np.array([1, 1, 1, 0, 1])  # Binary labels (0 or 1)
    
    logloss_result = CostFunctions.log_loss(predicted_probabilities, predicted_probabilities)
    logloss_sklearn = log_loss(actual_labels, predicted_probabilities)       



    
    print(f"Custom log loss: {logloss_result:.2f}")
    print(f"Scikit-learn log loss: {logloss_sklearn:.2f}%")
    
    
   
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------  