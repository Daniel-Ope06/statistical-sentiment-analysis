import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder  # type: ignore

from src2.preprocessing import preprocess_sentiment
from src2.feature_extraction import (
    extract_bilstm_features,
    extract_distilbert_features
)
from src2.model_training import (
    train_bilstm,
    train_distilbert,
    evaluate_predictions,
    get_misclassified_examples
)


def main():
    """
    Main execution pipeline for Deep Learning sentiment analysis.
    Orchestrates data loading, preprocessing, feature extraction,
    and model evaluation.
    """
    # Task 0: DATA LOADING
    df_train = pd.read_csv('data2/train.csv', encoding='utf-8-sig')
    df_val = pd.read_csv('data2/val.csv', encoding='utf-8-sig')
    df_test = pd.read_csv('data2/test.csv', encoding='latin-1')

    # Isolate columns and drop nulls
    df_train = df_train[['text', 'sentiment']].dropna(
        subset=['text', 'sentiment'])
    df_val = df_val[['text', 'sentiment']].dropna(subset=['text', 'sentiment'])
    df_test = df_test[['text', 'sentiment']].dropna(
        subset=['text', 'sentiment'])

    # Enforce string data type for text columns
    df_train['text'] = df_train['text'].astype(str)
    df_val['text'] = df_val['text'].astype(str)
    df_test['text'] = df_test['text'].astype(str)

    print(f"Train size: {len(df_train)} | Val size: {len(df_val)} | Test size: {len(df_test)}")  # noqa: E501
    print("----------------------------------\n")

    # Task 1: PREPROCESSING
    df_train['clean_text'] = df_train['text'].apply(preprocess_sentiment)
    df_val['clean_text'] = df_val['text'].apply(preprocess_sentiment)
    df_test['clean_text'] = df_test['text'].apply(preprocess_sentiment)

    print("\nSample Processed Data:")
    pd.set_option('display.max_colwidth', None)  # prevent truncating text
    print(df_train[['text', 'clean_text', 'sentiment']].head())
    print("----------------------------------\n")

    # Task 2: FEATURE EXTRACTION
    print("\n--- Feature Extraction Summary ---")
    VOCAB_LIMIT = 20000
    MAX_LEN = 50

    # Model 1: Extract BiLSTM Features (Word-level tokenization)
    bilstm_tokenizer, X_train_bilstm, X_val_bilstm, X_test_bilstm = extract_bilstm_features(  # noqa: E501
        df_train['clean_text'], df_val['clean_text'], df_test['clean_text'],
        max_vocab=VOCAB_LIMIT, max_len=MAX_LEN
    )
    print(f"[BiLSTM] Train Tensors Shape: {X_train_bilstm.shape}")

    # Model 2: Extract DistilBERT Features (Subword tokenization)
    bert_tokenizer, train_encodings, val_encodings, test_encodings = extract_distilbert_features(  # noqa: E501
        df_train['clean_text'], df_val['clean_text'], df_test['clean_text'],
        max_len=MAX_LEN
    )
    print(f"[DistilBERT] Train Tensors Shape: {train_encodings['input_ids'].shape}")  # noqa: E501
    print("----------------------------------\n")

    # Task 3: MODEL TRAINING AND EVALUATION
    # Label Encoding (e.g., negative -> 0, neutral -> 1, positive -> 2)
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['sentiment'])
    y_val = le.transform(df_val['sentiment'])
    y_test = le.transform(df_test['sentiment'])
    NUM_CLASSES = len(le.classes_)

    print("\n--- Task 3: Model Training ---")
    # Combination 1: BiLSTM + Word-Level
    bilstm_model, bilstm_history, bilstm_val_metrics, _ = train_bilstm(
        X_train_bilstm, y_train, X_val_bilstm, y_val,
        vocab_size=VOCAB_LIMIT, max_len=MAX_LEN, num_classes=NUM_CLASSES
    )

    # Combination 2: DistilBERT + Subword
    distilbert_model, bert_history, bert_val_metrics, _ = train_distilbert(
        train_encodings, y_train, val_encodings, y_val,
        num_classes=NUM_CLASSES
    )

    print("\n--- Model Evaluation Results ---")

    # Predict and evaluate BiLSTM on strictly unseen test data
    bilstm_test_probs = bilstm_model.predict(X_test_bilstm)
    bilstm_test_preds = np.argmax(bilstm_test_probs, axis=1)
    bilstm_test_metrics = evaluate_predictions(y_test, bilstm_test_preds)

    # Predict and evaluate DistilBERT on strictly unseen test data
    bert_test_output = distilbert_model.predict(dict(test_encodings))
    bert_test_preds = np.argmax(bert_test_output.logits, axis=1)
    bert_test_metrics = evaluate_predictions(y_test, bert_test_preds)

    # Aggregate metrics into a DataFrame for clean reporting
    results_data = [
        {"Model": "BiLSTM (Trainable Word-Level)", **bilstm_test_metrics},
        {"Model": "DistilBERT (Pretrained Subword)", **bert_test_metrics}
    ]
    results_df = pd.DataFrame(results_data)
    print(results_df.to_string(index=False))
    print("----------------------------------\n")

    # Task 4: ERROR ANALYSIS
    print("\n--- Task 4: Error Analysis (DistilBERT) ---")

    # Retrieve 10 examples where the top-performing model's prediction failed
    misclassified_df = get_misclassified_examples(
        y_true=y_test,
        y_pred=bert_test_preds,
        raw_texts=df_test['text'],
        label_encoder=le,
        num_examples=10
    )

    print(misclassified_df.to_string(index=False, justify='left'))
    print("----------------------------------\n")


if __name__ == "__main__":
    main()
