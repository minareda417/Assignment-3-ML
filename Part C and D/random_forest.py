import numpy as np
import pandas as pd

from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, n_trees: int, max_features: int, max_depth: int = 4, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_trees = n_trees
        self.max_features = max_features
        self.trees = []
        self.seed = 42

    def _create_bootstrap_sample(self, x: pd.DataFrame, y: pd.DataFrame):
        n_samples = x.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return x[indices], y[indices]

    def fit(self, x: pd.DataFrame, y: pd.DataFrame):
        for _ in range(self.n_trees):
            x_sample, y_sample = self._create_bootstrap_sample(x, y)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split, max_features=self.max_features)
            tree.fit(x_sample, y_sample)
            self.trees.append(tree)

    def _most_common_label(self, labels:np.array):
        return np.argmax(np.bincount(labels))

    def predict(self, x: pd.DataFrame):
        tree_predictions = np.array([tree.predict(x) for tree in self.trees])
        majority_votes = np.apply_along_axis(self._most_common_label, axis=0, arr=tree_predictions)
        return majority_votes

    def calculate_probabilities(self, x: pd.DataFrame):
        tree_probabilities = np.array([tree.calculate_probabilities(x) for tree in self.trees])
        avg_probabilities = np.mean(tree_probabilities, axis=0)
        return avg_probabilities




