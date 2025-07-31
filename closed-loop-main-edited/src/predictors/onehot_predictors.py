from __future__ import annotations
from sklearn.linear_model import Ridge, Lasso

from typing import Sequence
import numpy as np

from utils import seqs_to_onehot, seqs_to_georgiev
from predictors.base_predictors import BaseRegressionPredictor, BaseGPPredictor


class OnehotRidgePredictor(BaseRegressionPredictor):
    """Simple one hot encoding + ridge regression."""

    def __init__(self, dataset_name, reg_coef=1.0, **kwargs):
        super().__init__(dataset_name, reg_coef, Ridge, **kwargs)

    def seq2feat(self, seqs: Sequence[str]) -> np.ndarray:
        return seqs_to_onehot(seqs)


class OnehotLassoPredictor(BaseRegressionPredictor):
    """Simple one hot encoding + lasso regression."""

    def __init__(self, dataset_name, reg_coef=1.0, **kwargs):
        super().__init__(dataset_name, reg_coef, Lasso, **kwargs)

    def seq2feat(self, seqs: Sequence[str]) -> np.ndarray:
        return seqs_to_onehot(seqs)


class OnehotGPPredictor(BaseGPPredictor):
    """One-hot encoding for Gaussian Process."""

    def seq2feat(self, seqs: Sequence[str]) -> np.ndarray:
        return seqs_to_onehot(seqs)


class GeorgievRidgePredictor(BaseRegressionPredictor):
    """Georgiev encoding + ridge regression."""

    def __init__(self, dataset_name, reg_coef=1.0, **kwargs):
        super().__init__(dataset_name, reg_coef, Ridge, **kwargs)

    def seq2feat(self, seqs: Sequence[str]) -> np.ndarray:
        return seqs_to_georgiev(seqs)
