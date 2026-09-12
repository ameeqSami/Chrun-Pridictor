from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np


class clean_cls(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.dummy_columns = None
        self.feature_columns = None    

    def _clean(self, X):
        X = X.copy()
        if 'customerID' in X.columns:
            X = X.drop(columns=['customerID'], errors='ignore')
     
        if 'TotalCharges' in X.columns:
            X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce').fillna(0)

        for col in X.columns:
            if set(X[col].dropna().unique()).issubset({'Yes', 'No'}):
                X[col] = X[col].map({'Yes': 1, 'No': 0})
        
        if 'gender' in X.columns and X['gender'].dtype == 'object':
            X['gender'] = X['gender'].map({'Male': 1, 'Female': 0})

        if self.dummy_columns:
            X = pd.get_dummies(X, columns=self.dummy_columns, drop_first=True, dtype=int)

        return X


    def fit(self, X, y=None):
        self.X = X.copy()
        if 'customerID' in self.X.columns:
            self.X = self.X.drop(columns=['customerID'], errors='ignore')

        if 'TotalCharges' in self.X.columns:
            self.X['TotalCharges'] = pd.to_numeric(self.X['TotalCharges'], errors='coerce').fillna(0)

        self.dummy_columns = list()
        for col in self.X.columns:
            if self.X[col].nunique() > 2 and self.X[col].dtype == 'object': 
                self.dummy_columns.append(col)
        
        cleaned_X = self._clean(self.X)
        self.feature_columns = cleaned_X.columns.tolist()
        return self


    def transform(self, X):
        cleaned_X = self._clean(X)

        if self.feature_columns is not None:
            cleaned_X = cleaned_X.reindex(columns=self.feature_columns, fill_value=0)
        return cleaned_X

    def get_feature_names_out(self, input_features=None):
        return self.feature_columns
    
