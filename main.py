import pandas as pd
from src.preprocessing import preprocess_sentiment


def main():
    """
    Main execution pipeline for statistical sentiment analysis.
    Orchestrates data loading, preprocessing, feature extraction,
    and model evaluation.
    """
    # TODO: Remove nrows=500 before final training and submission
    df_train = pd.read_csv('data/train.csv', encoding='latin-1', nrows=500)

    # Isolate target label and input feature
    df_train = df_train[['text', 'sentiment']]

    # Drop null values to ensure preprocessing pipeline stability
    df_train = df_train.dropna(subset=['text'])

    # Enforce string data type for text feature
    df_train['text'] = df_train['text'].astype(str)

    print("--- Raw Data Sample ---")
    print(df_train.head())

    print("\nExecuting preprocessing pipeline...")
    df_train['clean_text'] = df_train['text'].apply(preprocess_sentiment)

    print("\n--- Processed Data Sample ---")
    print(df_train[['sentiment', 'text', 'clean_text']].head())


if __name__ == "__main__":
    main()
