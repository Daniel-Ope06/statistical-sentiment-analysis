import pandas as pd
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


def get_misclassified_examples(
        y_true, y_pred, raw_texts, label_encoder, num_examples=5):
    """
    Identifies and extracts examples where the model's prediction failed.
    """
    errors_df = pd.DataFrame({
        'Text': raw_texts,
        'True_Label': label_encoder.inverse_transform(y_true),
        'Predicted_Label': label_encoder.inverse_transform(y_pred)
    })

    errors_df = errors_df[
        errors_df['True_Label'] != errors_df['Predicted_Label']]

    return errors_df.head(num_examples)
