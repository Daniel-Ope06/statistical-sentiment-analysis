import torch
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential  # type: ignore
from tensorflow.keras.layers import (  # type: ignore
    Embedding, Bidirectional, LSTM, Dense, Dropout)
from tensorflow.keras.callbacks import EarlyStopping  # type: ignore
from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments  # noqa: E501
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


def train_bilstm(X_train, y_train, X_val, y_val, vocab_size, num_classes=3):
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
        Embedding(input_dim=vocab_size, output_dim=128),
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


# PyTorch requires a Dataset object to handle batching memory efficiently
class SentimentDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.labels)


class KerasMockHistory:
    def __init__(self, logs):
        self.history = {
            'loss': [log_entry['loss'] for log_entry in logs if 'loss' in log_entry],  # noqa: E501
            'val_loss': [log_entry['eval_loss'] for log_entry in logs if 'eval_loss' in log_entry],  # noqa: E501
            'accuracy': [log_entry['eval_accuracy'] for log_entry in logs if 'eval_accuracy' in log_entry],  # noqa: E501
            'val_accuracy': [log_entry['eval_accuracy'] for log_entry in logs if 'eval_accuracy' in log_entry]  # noqa: E501
        }


def train_distilbert(train_encodings, y_train, val_encodings, y_val, num_classes=3):  # noqa: E501
    """
    Loads and fine-tunes a pretrained DistilBERT sequence classifier.
    Requires a very low learning rate to prevent catastrophic forgetting.

    Args:
        num_classes (int): The number of output sentiment categories.
    """
    print("\n--- Training DistilBERT ---")

    # Convert data into PyTorch Datasets
    train_dataset = SentimentDataset(train_encodings, y_train)
    val_dataset = SentimentDataset(val_encodings, y_val)

    model = DistilBertForSequenceClassification.from_pretrained(
        'distilbert-base-uncased',
        num_labels=num_classes
    )

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=3e-5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True
    )

    # How trainer calculates accuracy during training
    def compute_metrics(pred):
        labels = pred.label_ids
        preds = np.argmax(pred.predictions, axis=1)
        return {'accuracy': accuracy_score(labels, preds)}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Predict and Evaluate
    print("\nEvaluating DistilBERT on Validation Set...")
    val_raw_output = trainer.predict(val_dataset)
    val_predictions = np.argmax(val_raw_output.predictions, axis=1)
    metrics = evaluate_predictions(y_val, val_predictions)
    history = KerasMockHistory(trainer.state.log_history)

    return model, history, metrics, val_predictions
