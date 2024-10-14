#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 14:06:53 2024

@author: lucamesserschmidt
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load all datasets
file_paths = {
    'DataPutAme': '~/Documents/DataPutAme.xlsx',
    'DataPutEuro': '~/Documents/DataPutEuro.xlsx',
    'DataCallEuro': '~/Documents/DataCallEuro.xlsx',
    'DataCallAme': '~/Documents/DataCallAme.xlsx'
}

# Initialize a dictionary to store performance results
results = {file: pd.DataFrame(columns=['Model', 'RMSE_Train', 'RMSE_Test', 'MAE_Train', 'MAE_Test', 'R2_Train', 'R2_Test', 'Overfit_Check', 'Best_Params']) for file in file_paths.keys()}

# Function to evaluate the models
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # Predictions on training data
    train_predictions = model.predict(X_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, train_predictions))
    mae_train = mean_absolute_error(y_train, train_predictions)
    r2_train = r2_score(y_train, train_predictions)

    # Predictions on testing data
    test_predictions = model.predict(X_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, test_predictions))
    mae_test = mean_absolute_error(y_test, test_predictions)
    r2_test = r2_score(y_test, test_predictions)

    return rmse_train, rmse_test, mae_train, mae_test, r2_train, r2_test

# Function to plot learning curves
def plot_learning_curve(estimator, X, y, title):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10), scoring='neg_mean_squared_error'
    )
    
    train_scores_mean = -np.mean(train_scores, axis=1)
    test_scores_mean = -np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.title(title)
    plt.xlabel("Training Examples")
    plt.ylabel("RMSE")
    
    plt.plot(train_sizes, np.sqrt(train_scores_mean), label="Training Error", color="r")
    plt.plot(train_sizes, np.sqrt(test_scores_mean), label="Cross-validation Error", color="g")
    plt.legend(loc="best")
    plt.grid()
    plt.show()

# Function to perform grid search cross-validation
def tune_hyperparameters(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_, grid_search.best_params_

# Loop through each file and train models independently
for file_name, file_path in file_paths.items():
    # Load dataset
    data = pd.read_excel(file_path)
    
    # Assuming the first 5 columns are inputs and the last column is the target ('Settle Price')
    X = data.iloc[:, :-1].values  # First 5 columns as features
    y = data.iloc[:, -1].values   # Last column as target

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # List of models with hyperparameters
    models = {
        'SVR': (SVR(), {'C': [0.1, 1, 10], 'epsilon': [0.01, 0.1, 0.5], 'kernel': ['rbf']}),
        'RandomForest': (RandomForestRegressor(random_state=42), {'n_estimators': [1, 2, 4 ,8 ,16, 32, 64, 100, 200], 'max_depth': [3, 5, 7], 'min_samples_split': [2, 5], 'criterion' : ['squared_error']}),
        'XGBoost': (XGBRegressor(random_state=42), {'n_estimators': [100, 200, 300, 400, 500, 600, 700 ,800, 900, 1000], 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3], 'max_depth': [3, 4, 5, 6, 7, 8, 9, 10]}),
        'LGBM': (LGBMRegressor(random_state=42), {'min_data_in_leaf':[10, 20, 50, 100, 200, 300] ,'n_estimators': [100, 200, 300, 400, 500, 600 ,700 ,800, 900, 1000], 'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3], 'num_leaves': [32, 64, 128, 256], 'max_depth':[3, 4, 5, 6, 7, 8, 9 , 10]}),
        'MLP': (MLPRegressor(random_state=42), {'hidden_layer_sizes': [(4,), (4, 3,), (4,3,2,), (32,), (128,), (128, 64,), (128,64,32,), (128, 64, 32, 16,), (100,100,),(50,50,),(150,150,),(100,),(50,),(150,),(100,100,100,), (50,50,50,),(150,150,150,), (50,50,50,50,),(150,150,150,150,),(100,100,100,100,)], 'activation': ['relu'], 'solver': ['adam', 'lbfgs', 'sgd']})
    }

    # Train each model and evaluate
    for model_name, (model, param_grid) in models.items():
        # Tune hyperparameters using grid search
        best_model, best_params = tune_hyperparameters(model, param_grid, X_train_scaled, y_train)
        
        # Train the best model
        best_model.fit(X_train_scaled, y_train)
        
        # Evaluate the model on both training and testing data
        rmse_train, rmse_test, mae_train, mae_test, r2_train, r2_test = evaluate_model(
            best_model, X_train_scaled, y_train, X_test_scaled, y_test)
        
        # Store the results
        new_results = pd.DataFrame({
            'Model': [model_name],
            'RMSE_Train': [rmse_train],
            'RMSE_Test': [rmse_test],
            'MAE_Train': [mae_train],
            'MAE_Test': [mae_test],
            'R2_Train': [r2_train],
            'R2_Test': [r2_test],

            'Best_Params': [str(best_params)]
        })

        
        # Append the new results using pd.concat
        results[file_name] = pd.concat([results[file_name], new_results], ignore_index=True)
        
        # Plot learning curve
        plot_learning_curve(best_model, X_train_scaled, y_train, f"{file_name} - {model_name}")

# Save the results to Excel files
for file_name, result_table in results.items():
    result_table.to_excel(f'{file_name}_results_with_best_params.xlsx', index=False)