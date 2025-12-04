import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import matplotlib.pyplot as plt


def preprocess_data(df: pd.DataFrame):
    df.drop(["age", "fnlwgt", "education.num", "capital.gain", "capital.loss", "hours.per.week"], axis=1, inplace=True)
    le = LabelEncoder()
    for col in df.columns:
        df[col] = le.fit_transform(df[col])
    y = df['income']
    x = df.drop('income', axis=1)
    return x, y


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
