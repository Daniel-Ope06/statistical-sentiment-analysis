import pandas as pd
from src.preprocessing import preprocess_sentiment
from src.feature_extraction import (  # type: ignore
    extract_bow_features, extract_tfidf_features)


def main():
    """
    Main execution pipeline for statistical sentiment analysis.
    Orchestrates data loading, preprocessing, feature extraction,
    and model evaluation.
    """
    # Task 0: DATA LOADING
    # TODO: Remove nrows=500 before final training and submission.
    df_train = pd.read_csv('data/train.csv', encoding='latin-1', nrows=500)
    df_test = pd.read_csv('data/test.csv', encoding='latin-1', nrows=500)

    # Isolate columns and drop nulls
    df_train = df_train[['text', 'sentiment']].dropna(subset=['text'])
    df_test = df_test[['text', 'sentiment']].dropna(subset=['text'])

    # Enforce string data type for text column
    df_train['text'] = df_train['text'].astype(str)
    df_test['text'] = df_test['text'].astype(str)

    print("\n--- Raw Data Sample ---")
    print(df_train.head())

    # Task 1: PREPROCESSING
    df_train['clean_text'] = df_train['text'].apply(preprocess_sentiment)
    df_test['clean_text'] = df_test['text'].apply(preprocess_sentiment)

    print("\n--- Processed Data Sample ---")
    print(df_train[['sentiment', 'text', 'clean_text']].head())

    # Task 2: FEATURE EXTRACTION
    print("\n--- Feature Extraction Summary ---")
    VOCAB_LIMIT = 5000

    # Extract Bag-of-Words
    X_train_bow, X_test_bow, bow_vec = extract_bow_features(
        df_train['clean_text'], df_test['clean_text'], max_vocab=VOCAB_LIMIT
    )
    print(
        f"[BoW] Train Matrix: {X_train_bow.shape[0]} documents x {
            X_train_bow.shape[1]} features")
    print(
        f"[BoW] Test Matrix:  {X_test_bow.shape[0]} documents x {
            X_test_bow.shape[1]} features")

    # Extract TF-IDF
    X_train_tfidf, X_test_tfidf, tfidf_vec = extract_tfidf_features(
        df_train['clean_text'], df_test['clean_text'], max_vocab=VOCAB_LIMIT
    )
    print(
        f"[TF-IDF] Train Matrix: {X_train_tfidf.shape[0]} documents x {
            X_train_tfidf.shape[1]} features")
    print(
        f"[TF-IDF] Test Matrix:  {X_test_tfidf.shape[0]} documents x {
            X_test_tfidf.shape[1]} features")
    print("----------------------------------\n")


if __name__ == "__main__":
    main()
