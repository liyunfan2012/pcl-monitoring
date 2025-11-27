from .columntransformers import ColTransCPD, ColTransUPD
from ...utils.columntransformer_base import ColumnTransformerBase
from sklearn.pipeline import Pipeline


class ScoringEngineCard(ColumnTransformerBase):
    """
    """
    def __init__(self,
                 model_dict=None,
                 closure_rate=0.0005):
        super().__init__()
        if model_dict is None:
            model_dict = {}
        self.model_dict = model_dict
        self.pipeline = Pipeline([
            ('ColTransCPD', ColTransCPD(model_dict=model_dict)),
            ('ColTransUPD', ColTransUPD(closure_rate=closure_rate))
        ])

    def _fit(self, X, y=None):
        # THIS IS THE MISSING PIECE
        self.pipeline.fit(X, y)
        return self

    def _transform(self, X):
        X_t = self.pipeline.transform(X)
        return X_t

    def clear_log(self):
        self.LOGGER.handlers.clear()
        for step in self.pipeline.steps:
            step[1].clear_log()
