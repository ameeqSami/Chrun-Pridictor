from sklearn.base import BaseEstimator, ClassifierMixin, clone

class pridict_thresh(BaseEstimator, ClassifierMixin):
    def __init__(self, model, thresh=0.5):
        self.thresh = thresh
        self.model = model
        self._estimator_type = "classifier"

    def fit(self, X, y, **kwargs):
        self.model.fit(X, y, **kwargs)
        self.classes_ = self.model.classes_
        self.is_fitted_ = True  # lets sklearn's check_is_fitted detect fitted state
        return self

    def __sklearn_is_fitted__(self):
        return getattr(self, 'is_fitted_', False)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def predict(self, X):
        y_prob = self.model.predict_proba(X)[:, 1]
        y_pred = (y_prob > self.thresh).astype(int)
        return y_pred
        
        
        