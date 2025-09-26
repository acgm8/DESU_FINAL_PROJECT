from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
import numpy as np

# Feature Selection using PLS
class PLSFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_features=5, feature_names=None):
        self.n_components = n_components
        self.n_features = n_features
        self.feature_names = feature_names
        self.selected_features_ = None

    def fit(self, X, y):
        pls = PLSRegression(n_components=self.n_components)
        pls.fit(X, y)
        coefs = pls.coef_.ravel()

        # Select top n_features by absolute coefficient value
        top_idx = np.argsort(np.abs(coefs))[-self.n_features:]
        self.selected_features_ = np.sort(top_idx)

        return self

    def transform(self, X):
        return X[:, self.selected_features_]

    def get_feature_names(self):
        if self.feature_names is not None:
            return [self.feature_names[i] for i in self.selected_features_]
        else:
            return self.selected_features_