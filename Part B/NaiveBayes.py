import numpy as np
import pandas as pd
from utils import evaluate
import matplotlib.pyplot as plt

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

    def _fit(self, alpha: float):
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

    def plot_probabilities(self):
        probabilities = self.calculate_probabilities(self.x)
        plt.figure(figsize=(10,6))
        fig, axes = plt.subplots(1,2, figsize=(14,6))
        axes[0].hist(probabilities[:,0], bins=20, color='skyblue', edgecolor='black')
        axes[0].set_title('Histogram of P(income <=50K | features)')
        axes[0].set_xlabel('Probability')
        axes[0].set_ylabel('Frequency')
        axes[1].hist(probabilities[:,1], bins=20, color='salmon', edgecolor='black')
        axes[1].set_title('Histogram of P(income >50K | features)')
        axes[1].set_xlabel('Probability')
        axes[1].set_ylabel('Frequency')
        plt.tight_layout()
        plt.show()

    def _predict(self, x: pd.DataFrame, y_true: pd.DataFrame):
        posteriors = self._calculate_posterior(x)
        y_pred = np.argmax(posteriors, axis = 1)
        return evaluate(y_true, y_pred)

    def fit_and_predict(self, x: pd.DataFrame, y_true: pd.DataFrame, alpha:float):
        self._fit(alpha)
        return self._predict(x, y_true)


    def _plot_likelihoods(self, label_mapping: dict):
        fig, axes = plt.subplots(self.n_columns, 1, figsize=(12, 5 * self.n_columns))
        for idx, col in enumerate(self.columns):
            likelihoods_col = self.likelihoods[col]
            labels = [label_mapping[col][i] for i in range(len(likelihoods_col[0]))]
            x_pos = np.arange(len(labels))
            width = 0.35
            axes[idx].bar(x_pos - width / 2, likelihoods_col[0], width, label='Income <=50K')
            axes[idx].bar(x_pos + width / 2, likelihoods_col[1], width, label='Income >50K')
            axes[idx].set_xlabel('Feature Values')
            axes[idx].set_ylabel('Likelihood P(feature|class)')
            axes[idx].set_title(f'Likelihood Distribution for {col.upper()}')
            axes[idx].set_xticks(x_pos)
            axes[idx].set_xticklabels(labels, rotation=45, ha='right')
            axes[idx].legend()
            axes[idx].grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.show()


    def analyze_likelihoods(self, label_mapping: dict, top_n: int = 3):
        feature_impacts = []
        for col in self.columns:
            likelihoods_col = self.likelihoods[col]
            diff = np.abs(likelihoods_col[0] - likelihoods_col[1])
            top_indices = np.argsort(diff)[-top_n:][::-1]
            print(f"\n{'─' * 80}")
            print(f"Feature: {col.upper()}")
            print(f"{'─' * 80}")
            for rank, idx in enumerate(top_indices, 1):
                label_name = label_mapping[col][idx]
                prob_low_income = likelihoods_col[0][idx]
                prob_high_income = likelihoods_col[1][idx]
                difference = diff[idx]
                favors = "<=50K" if prob_low_income > prob_high_income else ">50K"
                print(f"\n  Rank {rank}: '{label_name}'")
                print(f"    P(label | income <=50K) = {prob_low_income:.4f}")
                print(f"    P(label | income >50K)  = {prob_high_income:.4f}")
                print(f"    Difference = {difference:.4f}")
                print(f"    Favors: {favors}")
            max_diff = diff[top_indices[0]]
            feature_impacts.append((col, max_diff))
        feature_impacts.sort(key=lambda x: x[1], reverse=True)
        print(f"\n{'=' * 80}")
        print("FEATURE IMPORTANCE RANKING (by maximum label difference)")
        print(f"{'=' * 80}")
        for rank, (feature, impact) in enumerate(feature_impacts, 1):
            print(f"  {rank}. {feature}: {impact:.4f}")
        print(f"{'=' * 80}\n")
        self._plot_likelihoods(label_mapping)
