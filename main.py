import pandas as pd
from sklearn.preprocessing import LabelEncoder  # type: ignore

from src.preprocessing import preprocess_sentiment
from src.feature_extraction import (
    extract_bow_features,
    extract_tfidf_features)
from src.classifiers import (
    train_naive_bayes,
    train_logistic_regression,
    get_misclassified_examples)


def main():
    """
    Main execution pipeline for statistical sentiment analysis.
    Orchestrates data loading, preprocessing, feature extraction,
    and model evaluation.
    """
    # Task 0: DATA LOADING
    df_train = pd.read_csv('data/train.csv', encoding='latin-1')
    df_test = pd.read_csv('data/test.csv', encoding='latin-1')

    # Isolate columns and drop nulls
    df_train = df_train[['text', 'sentiment']].dropna(subset=['text'])
    df_test = df_test[['text', 'sentiment']].dropna(subset=['text'])

    # Enforce string data type for text column
    df_train['text'] = df_train['text'].astype(str)
    df_test['text'] = df_test['text'].astype(str)

    print("\n--- Raw Data Sample ---")
    pd.set_option('display.max_colwidth', None)  # prevent truncating text
    print(df_train.head())
    print("----------------------------------\n")

    # Task 1: PREPROCESSING
    df_train['clean_text'] = df_train['text'].apply(preprocess_sentiment)
    df_test['clean_text'] = df_test['text'].apply(preprocess_sentiment)

    print("\n--- Processed Data Sample ---")
    print(df_train[['sentiment', 'text', 'clean_text']].head())
    print("----------------------------------\n")

    # Task 2: FEATURE EXTRACTION
    print("\n--- Feature Extraction Summary ---")
    VOCAB_LIMIT = 10000

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

    # Task 3: MODEL TRAINING AND EVALUATION
    C_VALUE = 10.0

    # Label Encoding
    le = LabelEncoder()
    y_train = le.fit_transform(df_train['sentiment'])
    y_test = le.transform(df_test['sentiment'])

    # Combination 1: Naive Bayes + BoW
    nb_bow_model, nb_bow_metrics, nb_bow_preds = train_naive_bayes(
        X_train_bow, y_train, X_test_bow, y_test
    )

    # Combination 2: Naive Bayes + TF-IDF
    nb_tfidf_model, nb_tfidf_metrics, nb_tfidf_preds = train_naive_bayes(
        X_train_tfidf, y_train, X_test_tfidf, y_test
    )

    # Combination 3: Logistic Regression + BoW
    lr_bow_model, lr_bow_metrics, lr_bow_preds = train_logistic_regression(
        X_train_bow, y_train, X_test_bow, y_test, c_value=C_VALUE
    )

    # Combination 4: Logistic Regression + TF-IDF
    lr_tfidf_model, lr_tfidf_metrics, lr_tfidf_preds = train_logistic_regression(  # noqa: E501
        X_train_tfidf, y_train, X_test_tfidf, y_test, c_value=C_VALUE
    )

    print("\n--- Model Evaluation Results ---")

    # Aggregate metrics into a list of dictionaries for pandas
    results_data = [
        {"Model": "Naive Bayes (BoW)", **nb_bow_metrics},
        {"Model": "Naive Bayes (TF-IDF)", **nb_tfidf_metrics},
        {"Model": f"Log Reg (BoW, C={C_VALUE})", **lr_bow_metrics},
        {"Model": f"Log Reg (TF-IDF, C={C_VALUE})", **lr_tfidf_metrics}
    ]

    # Create and display the DataFrame
    results_df = pd.DataFrame(results_data)
    print(results_df.to_string(index=False))
    print("----------------------------------\n")

    # Task 4: ERROR ANALYSIS
    print("\n--- Error Analysis (Log Reg BoW, C=10.0) ---")

    # Retrieve 10 examples where the model's prediction failed
    misclassified_df = get_misclassified_examples(
        y_true=y_test,
        y_pred=lr_bow_preds,
        raw_texts=df_test['text'],
        label_encoder=le,
        num_examples=10
    )

    print(misclassified_df.to_string(index=False, justify='left'))
    print("----------------------------------\n")


if __name__ == "__main__":
    main()
