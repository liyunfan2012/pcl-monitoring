from ...utils.columntransformer_base import ColumnTransformerBase
from functools import partial
from scipy.special import expit


class ColTransCPD(ColumnTransformerBase):
    """
    Example column transformer that fills NA values with a specified value.
    """

    def __init__(self, model_dict=None):
        super().__init__()
        self.model_dict = model_dict

    @staticmethod
    def scoring(r, model_dict=None):
        score = 0
        for k, v in model_dict.items():
            score += v * r[k]
        return expit(score)

    def _transform(self, X):
        X_t = X.copy()
        f = partial(self.scoring, model_dict=self.model_dict)
        X_t['cpd'] = X_t.apply(f, axis=1)

        return X_t


class ColTransUPD(ColumnTransformerBase):

    def __init__(self,
                 closure_rate=0.0,
                 eps=0.00000001):
        super().__init__()
        self.closure_rate = closure_rate
        self.eps = eps

    def _transform(self, X):
        cr, eps = self.closure_rate, self.eps
        X_t = X.copy()
        survival_prev = X_t.groupby("ID")["cpd"].transform(
            lambda s: ((1 - s - cr).clip(lower=eps, upper=1 - eps))
            .shift(fill_value=1).cumprod()
        )
        X_t["upd"] = survival_prev * X_t["cpd"]
        X_t["upd"] = survival_prev * X_t["cpd"]
        X_t["cumpd"] = X_t.groupby("ID")["upd"].cumsum()

        return X_t
