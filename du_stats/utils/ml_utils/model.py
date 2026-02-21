class DUStatsModel:

    def __init__(self, preprocessor:object, model: object):
        self.preprocessor = preprocessor
        self.model = model
    def predict(self, X):
        X_transformed = self.preprocessor.transform(X)
        return model.predict(X_transformed)