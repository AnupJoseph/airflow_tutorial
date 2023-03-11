from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import pandas as pd
import numpy as np


def download_dataset_fn():
    iris = load_iris()
    iris_df = pd.DataFrame(
        data=np.c_[iris["data"], iris["target"]],
        columns=iris["feature_names"] + ["target"],
    )
    iris_df.to_csv("iris_dataset.csv")


def data_processing_fn():
    final = pd.read_csv("iris_dataset.csv", index_col=0)
    cols = [
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)",
    ]
    final[cols] = final[cols].fillna(final[cols].mean())
    final.to_csv("clean_iris_dataset.csv")


def ml_training_RandomForest_fn(**kwargs):
    final = pd.read_csv("clean_iris_dataset.csv", index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(
        final.iloc[:, 0:4], final.iloc[:, -1], test_size=0.3
    )
    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"Accuracy of the model:{accuracy_score(y_test, y_pred)}")
    acc = accuracy_score(y_test, y_pred)

    kwargs["ti"].xcom_push(key="model_accuracy", value=acc)


def ml_training_Logistic_fn(**kwargs):
    final = pd.read_csv("clean_iris_dataset.csv", index_col=0)
    X_train, X_test, y_train, y_test = train_test_split(
        final.iloc[:, 0:4], final.iloc[:, -1], test_size=0.3
    )
    clf = LogisticRegression(multi_class="ovr")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    print(f"Accuracy of the model:{accuracy_score(y_test, y_pred)}")
    acc = accuracy_score(y_test, y_pred)

    kwargs["ti"].xcom_push(key="model_accuracy", value=acc)


def identify_best_model_fn(**kwargs):
    ti = kwargs["ti"]
    fetched_accs = ti.xcom_pull(
        key="model_accuracy",
        task_ids=["ml_training_RandomForest", "ml_training_Logistic"],
    )
