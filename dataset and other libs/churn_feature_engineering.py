"""
churn_feature_engineering.py
=============================
Provides two sklearn-compatible transformers for the Telco Churn pipeline:

1. **ChurnCleaner** – raw data cleaning (drop IDs, coerce types, normalise
   redundant service categories, encode binary and multi-category columns).

2. **ChurnFeatureEngineer** – domain-driven feature construction (charge
   dynamics, tenure cohorts, service count, high-risk flags).

Both classes follow the sklearn ``BaseEstimator + TransformerMixin`` API so
they can be dropped into any ``sklearn.pipeline.Pipeline``.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


# =========================================================================
# ChurnCleaner
# =========================================================================

class ChurnCleaner(BaseEstimator, TransformerMixin):
    """
    Cleans raw Telco Churn data:

    - Drops ``customerID`` (non-predictive unique identifier)
    - Coerces ``TotalCharges`` to float; fills blank entries with 0
      (rows with tenure=0 have an empty string for TotalCharges)
    - Normalises ``No internet service`` / ``No phone service`` to ``No``
      to eliminate 6 duplicate/redundant collinear columns
    - Maps binary Yes/No columns to 1/0 and gender to Male=1, Female=0
    - One-hot encodes remaining multi-class categoricals (Contract,
      PaymentMethod, InternetService)
    - Guarantees feature alignment between fit and transform phases via
      ``reindex`` so inference always sees the exact same columns as training

    Attributes
    ----------
    feature_columns : list or None
        Full ordered list of output column names captured during ``fit``.
    dummy_columns : list or None
        Categorical columns (object dtype, >2 unique values) that will be
        one-hot encoded.
    """

    def __init__(self):
        self.feature_columns = None  # Learned during fit; ensures column alignment
        self.dummy_columns = None    # Categorical columns to one-hot encode

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _clean(self, X):
        """
        Core cleaning logic applied during both fit and transform.

        Parameters
        ----------
        X : pd.DataFrame
            Raw customer feature matrix.

        Returns
        -------
        pd.DataFrame
            Cleaned and encoded dataframe.
        """
        df = X.copy()  # Avoid mutating the original dataframe

        # --- Drop the non-predictive customer identifier ---
        if 'customerID' in df.columns:
            df = df.drop(columns=['customerID'], errors='ignore')

        # --- Coerce TotalCharges: blank strings → NaN → 0 ---
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(
                df['TotalCharges'], errors='coerce'
            ).fillna(0)

        # --- Normalise redundant service sub-categories ---
        # Customers without internet cannot have streaming/security services.
        # Telco dataset encodes this as "No internet service" which is
        # effectively identical to "No" — collapse them to reduce collinearity.
        internet_sub_services = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        for col in internet_sub_services:
            if col in df.columns:
                df[col] = df[col].replace({'No internet service': 'No'})

        # Similarly for phone service sub-feature
        if 'MultipleLines' in df.columns:
            df['MultipleLines'] = df['MultipleLines'].replace({'No phone service': 'No'})

        # --- Binary-encode all remaining Yes/No columns to 1/0 ---
        for col in df.columns:
            if set(df[col].dropna().unique()).issubset({'Yes', 'No'}):
                df[col] = df[col].map({'Yes': 1, 'No': 0})

        # --- Encode gender: Male=1, Female=0 ---
        if 'gender' in df.columns and df['gender'].dtype == 'object':
            df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})

        # --- One-hot encode multi-category columns ---
        # drop_first=True avoids the dummy-variable trap (multicollinearity)
        return df

    def fit(self, X, y=None):
        """
        Learn which columns need one-hot encoding and capture the final
        column schema produced on the training set.

        Parameters
        ----------
        X : pd.DataFrame
            Training feature matrix.
        y : ignored

        Returns
        -------
        self
        """
        X_clean = self._clean(X)

        # Detect object columns that still remain after binary encoding
        self.dummy_columns = [
            c for c in X_clean.columns if X_clean[c].dtype == 'object'
        ]

        # Apply one-hot encoding and capture the complete output column list
        X_encoded = pd.get_dummies(
            X_clean, columns=self.dummy_columns, drop_first=True, dtype=int
        )
        self.feature_columns = list(X_encoded.columns)
        return self

    def transform(self, X):
        """
        Apply cleaning, normalisation, and encoding; then align columns.

        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix (train or test/inference).

        Returns
        -------
        pd.DataFrame
            Fully cleaned and column-aligned dataframe.
        """
        X_clean = self._clean(X)

        # One-hot encode using the same columns identified during fit
        X_encoded = pd.get_dummies(
            X_clean, columns=self.dummy_columns, drop_first=True, dtype=int
        )

        # Reindex to guarantee column alignment with training schema:
        # - Extra columns (unseen categories) are dropped
        # - Missing columns (rare categories absent in this batch) are filled with 0
        if self.feature_columns is not None:
            X_encoded = X_encoded.reindex(columns=self.feature_columns, fill_value=0)

        return X_encoded

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names (sklearn API convention).

        Returns
        -------
        np.ndarray
            Array of column name strings.
        """
        return np.array(self.feature_columns)


# =========================================================================
# ChurnFeatureEngineer
# =========================================================================

class ChurnFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Creates high-impact domain features for churn prediction:

    1. **ChargeDiff** – Billing & charges discrepancy:
       ``TotalCharges - tenure * MonthlyCharges``
       Positive values may indicate rate hikes or billing anomalies.

    2. **AvgMonthlyCharges** – Average monthly spend over tenure:
       ``TotalCharges / (tenure + 1)``
       Smoothed with +1 to avoid division-by-zero for new customers.

    3. **Monthly_to_Total_Ratio** – Ratio of current monthly charge to
       total historical spend:
       ``MonthlyCharges / (TotalCharges + 1)``
       High values suggest a recent price increase.

    4. **TotalServices** – Count of active bundled add-on services
       (OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport,
       StreamingTV, StreamingMovies). Higher counts indicate stickiness.

    5. **HighRisk_Contract_Payment** – Binary flag (1/0) for the highest-
       churn contract/payment combination: month-to-month contract **and**
       electronic check payment.

    6. **Is_New_Customer** – Tenure ≤ 6 months flag (early-life churn risk).

    7. **Is_Long_Customer** – Tenure ≥ 48 months flag (loyal customer proxy).

    Attributes
    ----------
    feature_names : list or None
        Output column names captured during ``fit()``.
    """

    def __init__(self):
        self.feature_names = None  # Populated by fit(); ensures column alignment

    # ------------------------------------------------------------------
    # Internal helper
    # ------------------------------------------------------------------

    def _engineer(self, X):
        """
        Compute all engineered features and append them to the input frame.

        Parameters
        ----------
        X : pd.DataFrame
            Cleaned and encoded feature matrix (output of ChurnCleaner).

        Returns
        -------
        pd.DataFrame
            Original columns **plus** all new engineered features.
        """
        df = X.copy()  # Work on a copy to avoid side-effects

        # --- 1. Billing and charge dynamics ---
        if (
            'tenure' in df.columns
            and 'MonthlyCharges' in df.columns
            and 'TotalCharges' in df.columns
        ):
            # Expected total based on current rate × tenure (months billed so far)
            expected_total = df['tenure'] * df['MonthlyCharges']

            # Positive diff → customer has been charged more than expected
            df['ChargeDiff'] = df['TotalCharges'] - expected_total

            # Smoothed average monthly spend; avoids /0 for tenure=0 customers
            df['AvgMonthlyCharges'] = df['TotalCharges'] / (df['tenure'] + 1)

            # High ratio means recent charges are disproportionately large
            df['Monthly_to_Total_Ratio'] = df['MonthlyCharges'] / (df['TotalCharges'] + 1)

        # --- 2. Tenure cohort flags ---
        if 'tenure' in df.columns:
            # New customers (first 6 months) have the highest churn risk
            df['Is_New_Customer'] = (df['tenure'] <= 6).astype(int)
            # Long-term customers (4+ years) are generally sticky
            df['Is_Long_Customer'] = (df['tenure'] >= 48).astype(int)

        # --- 3. Total add-on services count (measures customer "stickiness") ---
        service_cols = [
            c for c in [
                'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies'
            ]
            if c in df.columns
        ]
        if service_cols:
            # Each service col is already binary-encoded (1=Yes, 0=No)
            df['TotalServices'] = df[service_cols].sum(axis=1)

        # --- 4. High-risk flag: Month-to-month + Electronic check ---
        # This combination has historically the highest churn rate
        m2m_cols = [c for c in df.columns if 'Month-to-month' in c]
        echeck_cols = [c for c in df.columns if 'Electronic check' in c]
        if m2m_cols and echeck_cols:
            df['HighRisk_Contract_Payment'] = (
                (df[m2m_cols[0]] == 1) & (df[echeck_cols[0]] == 1)
            ).astype(int)

        return df

    # ------------------------------------------------------------------
    # Sklearn API
    # ------------------------------------------------------------------

    def fit(self, X, y=None):
        """
        Compute engineered features on the training set and remember
        output column order.

        Parameters
        ----------
        X : pd.DataFrame
            Cleaned feature matrix.
        y : ignored

        Returns
        -------
        self
        """
        out = self._engineer(X)
        self.feature_names = list(out.columns)  # Snapshot column order
        return self

    def transform(self, X):
        """
        Compute engineered features and align to training column schema.

        Parameters
        ----------
        X : pd.DataFrame
            Cleaned feature matrix.

        Returns
        -------
        pd.DataFrame
            Feature matrix with all engineered columns appended.
        """
        out = self._engineer(X)

        # Reindex to ensure consistent feature set (handles missing columns gracefully)
        if self.feature_names is not None:
            out = out.reindex(columns=self.feature_names, fill_value=0)

        return out

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names (sklearn API convention).

        Returns
        -------
        np.ndarray
            Array of column name strings.
        """
        return np.array(self.feature_names)
