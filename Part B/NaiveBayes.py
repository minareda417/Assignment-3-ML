import numpy as np
import pandas as pd
from utils import evaluate

class NaiveBayes:
    def __init__(self, x: pd.DataFrame, y: pd.DataFrame, unique_values: pd.DataFrame):
        self.x = x
        self.y = y if y.name == 'income' else y.rename('income')
        self.unique_values = unique_values
        (self.n_rows, self.n_columns) = self.x.shape
        self.priors = None
        self.likelihoods = None
        self.columns = self.x.columns

    def _calculate_priors(self, alpha:float):
        labels_count = self.y.value_counts().sort_index()
        self.priors = (labels_count + alpha) / (self.n_rows + 2*alpha)

    def _calculate_likelihoods(self, alpha: float):
        likelihoods = {}
        y = self.y
        z = pd.concat([self.x, y], axis=1)
        class_counts = self.y.value_counts().sort_index()
        for col in self.columns:
            counts = z.groupby("income")[col].value_counts().unstack(fill_value=0)
            num_unique = self.unique_values[col]
            all_values = pd.Index(range(num_unique))
            counts = counts.reindex(columns=all_values, fill_value=0)
            smoothed0 = (counts.iloc[0] + alpha) / (class_counts[0] + alpha * num_unique)
            smoothed1 = (counts.iloc[1] + alpha) / (class_counts[1] + alpha * num_unique)
            likelihoods[col] = np.array([smoothed0, smoothed1])
        self.likelihoods = likelihoods

    def fit(self, alpha: float):
        self._calculate_priors(alpha)
        self._calculate_likelihoods(alpha)

    def _calculate_posterior(self, x:pd.DataFrame):
        posteriors = np.zeros((x.shape[0], 2), dtype=np.float64)
        for idx, (_, row) in enumerate(x.iterrows()):
            for i in range(2):
                posterior = self.priors[i]
                for col in self.columns:
                    posterior *= self.likelihoods[col][i][row[col]]
                posteriors[idx][i] = posterior
        return posteriors

    def calculate_probabilities(self, x:pd.DataFrame):
        posteriors = self._calculate_posterior(x)
        totals = posteriors.sum(axis=1, keepdims=True)
        probabilities = posteriors/totals
        return probabilities

    def predict(self, x: pd.DataFrame, y_true: pd.DataFrame):
        posteriors = self._calculate_posterior(x)
        y_pred = np.argmax(posteriors, axis = 1)
        return evaluate(y_true, y_pred)

    def fit_and_predict(self, x: pd.DataFrame, y_true: pd.DataFrame, alpha:float):
        self.fit(alpha)
        return self.predict(x, y_true)