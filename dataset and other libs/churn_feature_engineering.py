import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ChurnCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans raw Telco Churn data:
    - Drops 'customerID'
    - Coerces 'TotalCharges' to float and handles 0 tenure cases
    - Normalizes 'No internet service' / 'No phone service' to 'No' to eliminate
      6 duplicate redundant collinear columns
    - Maps binary Yes/No and gender
    - Performs one-hot encoding for multi-class categoricals
    - Guarantees feature alignment across fit and transform
    """
    def __init__(self):
        self.feature_columns = None
        self.dummy_columns = None

    def fit(self, X, y=None):
        X_clean = self._clean(X)
        self.dummy_columns = [c for c in X_clean.columns if X_clean[c].dtype == 'object']
        X_encoded = pd.get_dummies(X_clean, columns=self.dummy_columns, drop_first=True, dtype=int)
        self.feature_columns = list(X_encoded.columns)
        return self

    def _clean(self, X):
        df = X.copy()
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'], errors='ignore')

        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

        # Normalize redundant categories before encoding
        internet_sub_services = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        for col in internet_sub_services:
            if col in df.columns:
                df[col] = df[col].replace({'No internet service': 'No'})

        if 'MultipleLines' in df.columns:
            df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})

        # Map binary Yes/No
        for col in df.columns:
            if set(df[col].dropna().unique()).issubset({'Yes', 'No'}):
                df[col] = df[col].map({'Yes': 1, 'No': 0})

        # Map gender
        if 'gender' in df.columns and df['gender'].dtype == 'object':
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

        return df

    def transform(self, X):
        X_clean = self._clean(X)
        X_encoded = pd.get_dummies(X_clean, columns=self.dummy_columns, drop_first=True, dtype=int)
        if self.feature_columns is not None:
            X_encoded = X_encoded.reindex(columns=self.feature_columns, fill_value=0)
        return X_encoded

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_columns)


class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates high-impact domain features for churn prediction:
    1. Billing & charges discrepancy (catches rate hikes and billing anomalies)
    2. Average monthly charges over tenure
    3. Ratio of monthly to total charges
    4. Total active bundled services count
    5. High-risk contract & payment method combination
    6. Tenure cohort flags (new customers vs long-term loyal customers)
    """
    def __init__(self):
        self.feature_names = None

    def fit(self, X, y=None):
        out = self._engineer(X)
        self.feature_names = list(out.columns)
        return self

    def _engineer(self, X):
        df = X.copy()

        # 1. Billing and charge dynamics
        if 'tenure' in df.columns and 'MonthlyCharges' in df.columns and 'TotalCharges' in df.columns:
            expected_total = df['tenure'] * df['MonthlyCharges']
            df['ChargeDiff'] = df['TotalCharges'] - expected_total
            df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)
            df['Monthly_to_Total_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

        # 2. Tenure cohorts
        if 'tenure' in df.columns:
            df['Is_New_Customer'] = (df['tenure'] <= 6).astype(int)
            df['Is_Long_Customer'] = (df['tenure'] >= 48).astype(int)

        # 3. Add-on services count (stickiness measure)
        service_cols = [
            c for c in ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                        'TechSupport', 'StreamingTV', 'StreamingMovies']
            if c in df.columns
        ]
        if service_cols:
            df['TotalServices'] = df[service_cols].sum(axis=1)

        # 4. High-risk flag: Month-to-month + Electronic check
        m2m_cols = [c for c in df.columns if 'Month-to-month' in c]
        echeck_cols = [c for c in df.columns if 'Electronic check' in c]
        if m2m_cols and echeck_cols:
            df['HighRisk_Contract_Payment'] = (
                (df[m2m_cols[0]] == 1) & (df[echeck_cols[0]] == 1)
            ).astype(int)

        return df

    def transform(self, X):
        out = self._engineer(X)
        if self.feature_names is not None:
            out = out.reindex(columns=self.feature_names, fill_value=0)
        return out

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names)
