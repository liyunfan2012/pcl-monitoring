from columntransformers import ColTransCPD
from sklearn.pipeline import Pipeline
from ...utils.columntransformer_base import ColumnTransformerBase


class ScoringEngineCard(ColumnTransformerBase):
    """
    """
    def __init__(self,
                 model_dict=None):
        super().__init__()
        if model_dict is None:
            model_dict = {}
        self.model_dict = model_dict
        self.pipeline = Pipeline([
            ('ColTransCPD', ColTransCPD(model_dict=model_dict))
        ])


def _transform(self, X):
    X_t = self.pipeline.transform(X)
    return X_t
