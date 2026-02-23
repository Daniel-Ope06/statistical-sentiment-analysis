from sklearn.feature_extraction.text import (  # type: ignore
    CountVectorizer, TfidfVectorizer)


def extract_bow_features(train_texts, test_texts, max_vocab=5000):
    """
    Extracts Bag-of-Words (BoW) numerical features from preprocessed text.
    Fits strictly on training data to prevent data leakage.

    Args:
        train_texts: Cleaned training strings.
        test_texts: Cleaned testing strings.
        max_vocab: Maximum vocabulary size.

    Returns:
        tuple: (X_train_bow, X_test_bow, bow_vectorizer)
    """
    bow_vectorizer = CountVectorizer(max_features=max_vocab)

    X_train_bow = bow_vectorizer.fit_transform(train_texts)
    X_test_bow = bow_vectorizer.transform(test_texts)

    return X_train_bow, X_test_bow, bow_vectorizer


def extract_tfidf_features(train_texts, test_texts, max_vocab=5000):
    """
    Extracts TF-IDF numerical features from preprocessed text.
    Fits strictly on training data to prevent data leakage.

    Args:
        train_texts: Cleaned training strings.
        test_texts: Cleaned testing strings.
        max_vocab: Maximum vocabulary size.

    Returns:
        tuple: (X_train_tfidf, X_test_tfidf, tfidf_vectorizer)
    """
    tfidf_vectorizer = TfidfVectorizer(max_features=max_vocab)

    X_train_tfidf = tfidf_vectorizer.fit_transform(train_texts)
    X_test_tfidf = tfidf_vectorizer.transform(test_texts)

    return X_train_tfidf, X_test_tfidf, tfidf_vectorizer
