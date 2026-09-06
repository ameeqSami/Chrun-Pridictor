from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
    
class CorrelationThresholdFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 0.10):
        self.threshold = threshold
        self.selected_features = None

    def fit(self, x, y):
        df = pd.DataFrame(x).reset_index(drop=True)
        y_ser = pd.Series(np.asarray(y).ravel()).reset_index(drop=True)
        corrs = df.corrwith(y_ser.reset_index(drop=True))
        self.selected_features = corrs[corrs.abs() >= self.threshold].index    
        return self

    def transform(self, x):
        df = pd.DataFrame(x)
        return x[self.selected_features].values