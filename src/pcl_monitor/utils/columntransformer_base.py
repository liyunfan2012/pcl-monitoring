from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted
import pandas as pd
import numpy as np
from functools import partial

from ._logger import LoggerGenerator

class ColumnTransformerBase(BaseEstimator, TransformerMixin):
    """
    Base class for column-wise transformers with logging, NA handling, and basic lifecycle methods.
    """

    def __init__(self):
        self.LOGGER = LoggerGenerator().generate(self.__class__.__name__)
        self.transform_type = self.__class__.__name__[17:]

    def check_is_fitted(self, attributes=None):
        check_is_fitted(self, attributes)

    @property
    def log(self):
        if hasattr(self, 'columns'):
            return {self.transform_type: self.columns}
        return {}

    def clear_log(self):
        self.LOGGER.handlers.clear()

    def _fit(self, X, y=None):
        """To be implemented in subclass"""
        return self

    def _transform(self, X):
        """To be implemented in subclass"""
        return X

    def fit(self, X, y=None):
        self.LOGGER.info(f"Starting fit, X of shape: {X.shape}")
        self._fit(X, y)
        if hasattr(self, 'columns'):
            self.LOGGER.info(f"{self.columns} detected, done.")
        return self

    def transform(self, X):
        self.LOGGER.info(f"Starting transform, X of shape: {X.shape}")
        X_t = self._transform(X)
        self.LOGGER.info(f"X_t shape: {X_t.shape}, done")
        return X_t

    def fit_transform(self, X, y=None, **fit_params):
        """
        Fit to data, then transform it.
        """
        return self.fit(X, y).transform(X)
