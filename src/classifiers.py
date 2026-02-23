from sklearn.naive_bayes import MultinomialNB  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore
from sklearn.metrics import (  # type: ignore
    accuracy_score, precision_score, recall_score, f1_score)


def evaluate_predictions(y_true, y_pred):
    """
    Calculates core classification metrics using weighted averaging
    to account for potential class imbalances in sentiment labels.
    """
    return {
        'Accuracy': accuracy_score(y_true, y_pred),
        'Precision': precision_score(
            y_true, y_pred, average='weighted', zero_division=0),
        'Recall': recall_score(
            y_true, y_pred, average='weighted', zero_division=0),
        'F1-Score': f1_score(
            y_true, y_pred, average='weighted', zero_division=0)
    }


def train_naive_bayes(X_train, y_train, X_test, y_test):
    """
    Trains a Multinomial Naive Bayes classifier and evaluates its performance.
    """
    nb_model = MultinomialNB()
    nb_model.fit(X_train, y_train)

    predictions = nb_model.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)

    return nb_model, metrics, predictions


def train_logistic_regression(X_train, y_train, X_test, y_test, c_value=1.0):
    """
    Trains a Logistic Regression classifier and evaluates its performance.

    Args:
        c_value (float): Inverse of regularization strength.
    """
    lr_model = LogisticRegression(C=c_value, max_iter=1000, random_state=42)
    lr_model.fit(X_train, y_train)

    predictions = lr_model.predict(X_test)
    metrics = evaluate_predictions(y_test, predictions)

    return lr_model, metrics, predictions
