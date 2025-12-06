import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency


def preprocess_data(df: pd.DataFrame):
    df.drop(["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"], axis=1,
            inplace=True)
    label_mappings = {}
    for col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_mappings[col] = dict(zip(le.transform(le.classes_), le.classes_))
    return label_mappings


def analyze_target_distribution(df: pd.DataFrame):
    class_counts = df['income'].value_counts(normalize = True)
    print(class_counts)
    sns.countplot(x='income', data=df)
    plt.title('Class distribution')
    plt.show()

def feature_target_relationship(df:pd.DataFrame):
    fig, axes = plt.subplots(len(df.columns)-1, 2, figsize=(8, 5*(len(df.columns)-1)))
    for i, column_name in enumerate(df.columns):
        if column_name == 'income':
            continue
        sns.countplot(x = column_name, data=df, ax=axes[i- (1 if i>df.columns.get_loc('income') else 0)][0])
        axes[i- (1 if i>df.columns.get_loc('income') else 0)][0].set_title(f'{column_name} Distribution')

        sns.countplot(x = column_name, hue= 'income', data=df, ax=axes[i- (1 if i>df.columns.get_loc('income') else 0)][1])
        axes[i- (1 if i>df.columns.get_loc('income') else 0)][1].set_title(f'{column_name} vs Target')
    plt.tight_layout()
    plt.show()

def correlation_heatmap(df: pd.DataFrame):
    for col in df.columns:
        if col != 'income':
            contingency_table = pd.crosstab(df[col], df['income'])
            chi2, p, dof, ex = chi2_contingency(contingency_table)
            print(f'Chi-squared test for {col} and income: chi2={chi2}, p-value={p}')
    plt.figure(figsize=(10, 8))
    corr = df.corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.show()

def split_x_and_y(df: pd.DataFrame):
    y = df['income']
    x = df.drop('income', axis=1)
    return x, y

def plot_feature_analysis(df: pd.DataFrame, column_name: str, target_col: str = 'income'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Class distribution
    class_counts = df[target_col].value_counts()
    axes[0].bar(class_counts.index, class_counts.values, color=['steelblue', 'coral'])
    axes[0].set_xlabel('Class')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Class Distribution')
    axes[0].set_xticks(class_counts.index)

    # Plot 2: Feature-target relationship
    feature_target = df.groupby([column_name, target_col]).size().unstack(fill_value=0)
    feature_target.plot(kind='bar', ax=axes[1], color=['steelblue', 'coral'])
    axes[1].set_xlabel(column_name)
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'{column_name} vs Target')
    axes[1].legend(title='Class', labels=['Class 0', 'Class 1'])
    axes[1].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

def split_data(x: pd.DataFrame, y: pd.DataFrame, train_size: float, val_size: float, test_size: float, seed = 42):
    assert train_size+val_size+test_size == 1
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val,
                                                      test_size=val_size / (train_size + val_size), random_state=42,
                                                      stratify=y_train_val)
    return x_train, x_val, x_test, y_train, y_val, y_test

def evaluate(y_true: np.array, y_pred: np.array):
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    print(f"accuracy = {acc}\nprecision = {precision}\nrecall = {recall}\nf1 score = {f1}")
    conf_matrix = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=["<=50K", ">50K"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()
    return acc, precision, recall, f1

def print_best_alphas(alphas: list[float], accs:list[float], precisions:list[float], recalls:list[float], f1s:list[float]):
    metrics = {
        'Accuracy': list(zip(alphas, accs)),
        'Precisions': list(zip(alphas, precisions)),
        'Recall': list(zip(alphas, recalls)),
        'F1': list(zip(alphas, f1s))
    }

    for metric, scores in metrics.items():
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)
        print(f"\n{metric} (best to worst):")
        for alpha, score in sorted_scores:
            print(f"  α={alpha}: {score}")

def compare_alphas(accs:list[float], precisions:list[float], recalls:list[float], f1s:list[float]):
    alphas = [0.1, 0.5, 1, 2, 5]
    print_best_alphas(alphas, accs, precisions, recalls, f1s)
    plt.plot(alphas, accs, label="accuracy")
    plt.plot(alphas, precisions, label="precision")
    plt.plot(alphas, recalls, label="recall")
    plt.plot(alphas, f1s, label="f1")
    plt.legend()
    plt.title("performances")
    plt.xlabel("alpha")
    plt.ylabel("score")
    plt.show()
