import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Embedding, Bidirectional, LSTM, Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
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


def train_bilstm(X_train, y_train, X_val, y_val, vocab_size, max_len, num_classes=3):  # noqa: E501
    """
    Builds and trains a Bidirectional LSTM with a trainable embedding layer.
    Applies Dropout and Early Stopping for regularization.

    Args:
        vocab_size (int): Maximum number of unique words kept in the vocabulary.
        max_len (int): The exact length every sequence is padded/truncated to.
        num_classes (int): The number of output sentiment categories.
    """  # noqa: E501

    print("\n--- Training BiLSTM ---")

    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len),
        Bidirectional(LSTM(64, return_sequences=False)),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    early_stop = EarlyStopping(
        monitor='val_loss', patience=3, restore_best_weights=True)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=10,
        batch_size=64,
        callbacks=[early_stop]
    )

    # Predict and Evaluate
    print("\nEvaluating BiLSTM on Validation Set...")
    val_probs = model.predict(X_val)
    # Convert probabilities to class labels
    val_predictions = np.argmax(val_probs, axis=1)
    metrics = evaluate_predictions(y_val, val_predictions)

    return model, history, metrics, val_predictions
