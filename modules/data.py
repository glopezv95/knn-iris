from ucimlrepo import fetch_ucirepo
import pandas as pd
import streamlit as st
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
import numpy as np

# Fetch data for UCI Machine Learning Repository
iris = fetch_ucirepo(id = 53)
iris.data.keys()
# Generate pandas DataFrame
default_df = pd.DataFrame(
    data = iris.data.features,
    columns = iris.data.headers)

default_df['species'] = iris.data.targets
default_df = default_df.drop('class', axis = 1)
default_df.columns = default_df.columns.str.replace(' ', '_')

def col_score(data:pd.DataFrame, X:list, y:str, k: int = 6):
    col_score_bar = st.progress(0, 'Computing data...')
    col_score_dict = {}
    
    for item in X:
        
        col_score_bar.progress(len(col_score_dict)/len(X),
                               text = f'Computing data... {(len(col_score_dict)/len(X)) * 100}%')
        
        X_test, X_train, y_test, y_train = train_test_split(
            data[item].values.reshape(-1, 1),
            np.ravel(data[y].values.reshape(-1, 1)),
            train_size = .2,
            stratify = data[y].values.reshape(-1, 1),
            random_state = 17)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        knn = KNeighborsClassifier(n_neighbors = k, n_jobs = -1)
        knn.fit(X_train_scaled, y_train)
        col_score_dict[item] = round(knn.score(X_test_scaled, y_test), 2)
        
    final_dict = {key: value for key, value in sorted(col_score_dict.items(),
                                                      key = lambda item: item[1],
                                                      reverse = True)}
    
    col_score_bar.empty()
    
    final_df = {'variable':list(final_dict.keys()),
                'accuracy':list(final_dict.values())}
    
    return final_df

def k_score(data: pd.DataFrame, X: list, y: str, k_max: int):
    
    k_score_dict = {'k':[], 'accuracy':[]}
    
    if len(X) == 1:
            X_reshaped = data[X].values.reshape(-1, 1)
    else:
        X_reshaped = data[X].values
        
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped,
        data[y].values.reshape(-1, 1),
        train_size = .2,
        random_state = 17,
        stratify = data[y].values.reshape(-1, 1))
    
    
    scaler = StandardScaler()
    scaler.fit_transform(X_train)
    scaler.transform(X_test)
    
    for k_value in np.arange(1, k_max + 1):
        
        knn = KNeighborsClassifier(k_value)
        knn.fit(X_train, y_train)
        score = knn.score(X_test, y_test)
        
        k_score_dict['k'].append(k_value)
        k_score_dict['accuracy'].append(score)
    
    return k_score_dict
            
# def knn_global_metric(data: pd.DataFrame, X: list, y: str, k: int,
#                       key_metric: str):
    
#     X_train, X_test, y_train, y_test = train_test_split(
#         data[X].values,
#         data[y].values.reshape(-1, 1),
#         test_size = .2,
#         stratify = data[y].values.reshape(-1, 1))
    
#     knn = KNeighborsClassifier(n_neighbors = k)
#     scaler = StandardScaler()

#     X_train = scaler.fit_transform(X_train)
#     X_test = scaler.transform(X_test)

#     knn.fit(X_train, np.ravel(y_train))
#     y_pred = knn.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     precision = precision_score(y_test, y_pred, average = 'weighted')
#     recall = recall_score(y_test, y_pred, average = 'weighted')
    
#     key_metric_allowed_values = ['prec', 'recall', 'none']
    
#     if key_metric not in key_metric_allowed_values:
#         raise ValueError(
#             f'Invalid "key_metric" value. Allowed values are {key_metric_allowed_values}')
    
#     if key_metric == 'prec':
#         mean_metric = (.2 * accuracy + .5 * precision + .3 * recall)
        
#     elif key_metric == 'recall':
#         mean_metric = (.2 * accuracy + .3 * precision + .5 * recall)
        
#     elif key_metric == 'none':
#         mean_metric = (.2 * accuracy + .4 * precision + .4 * recall)
        
#     return mean_metric

# k_mean_values_dict = {}

# knn_global_metric(default_df, list(default_df.columns.drop('Cover_Type')), 'Cover_Type', 9, 'str')