# from sklearn.base import BaseEstimator, TransformerMixin
# import pandas as pd
# import numpy as np
    
# class CorrelationThresholdFilter(BaseEstimator, TransformerMixin):
#     def __init__(self, threshold = 0.10):
#         self.threshold = threshold
#         self.selected_features = None

#     def fit(self, x, y):
#         df = pd.DataFrame(x).reset_index(drop=True)
#         y_ser = pd.Series(np.asarray(y).ravel()).reset_index(drop=True)
#         corrs = df.corrwith(y_ser.reset_index(drop=True))
#         print(corrs.abs())
#         self.selected_features = corrs[corrs.abs() >= self.threshold].index    
#         return self

#     def transform(self, x):
#         df = pd.DataFrame(x)
#         return x[self.selected_features].values


from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np
    
class CorrelationThresholdFilter(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 0.10):
        self.threshold = threshold
        self.selected_features = None

    def fit(self, x, y):
        X = x.copy().reset_index(drop=True)
        Y = y.copy().reset_index(drop=True)
        y_ser = pd.Series(np.asarray(Y).ravel())
        corrs = X.corrwith(y_ser)
        self.selected_features = corrs[corrs.abs() >= self.threshold].index
        
        if len(self.selected_features) == 0:
            max_corr = corrs.abs().max()
            raise ValueError(
                f"CorrelationThresholdFilter Error: All features were dropped!\n"
                f"   - Your threshold: {self.threshold}\n"
                f"   - Max correlation in your data: {max_corr:.4f}\n"
                f"   Fix: Lower your threshold below {max_corr:.4f}."
            )
        return self

    def transform(self, x):
        return x[self.selected_features]

    def get_feature_names_out(self, input_features=None):
        return self.selected_features