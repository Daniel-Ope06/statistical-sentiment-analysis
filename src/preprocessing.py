import re
import spacy

# Load spaCy model with parser and NER disabled to optimize processing speed
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


# Customize default stop words to retain sentiment-modifying terms
custom_stop_words = nlp.Defaults.stop_words.copy()
exceptions = {
    "not", "no", "never", "none", "cannot", "but", "however", "down", "up"
}
for word in exceptions:
    if word in custom_stop_words:
        custom_stop_words.remove(word)


def preprocess_sentiment(text):
    """
    Preprocesses raw text data specifically for sentiment analysis.
    Removes platform-specific noise, preserves critical punctuation
    and profanity as text tokens, applies lemmatization,
    filters custom stop words, and implements sequential negation handling.
    """
    # Convert to lowercase
    text = text.lower()

    # Remove platform-specific noise (URLs, user mentions, hashtag symbols)
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+', '', text)
    text = re.sub(r'\#', '', text)

    # Preserve sentiment-heavy characters by converting them to text tokens
    text = re.sub(r'\*{2,}', ' censoredswear ', text)
    text = re.sub(r'!+', ' exclamationmark ', text)
    text = re.sub(r'\?+', ' questionmark ', text)

    # Tokenize text
    doc = nlp(text)
    tokens = []

    # Initialize negation tracking flag
    negate_next = False

    for token in doc:
        # Reset negation flag on punctuation and proceed to next token
        if token.is_punct:
            negate_next = False
            continue

        # Filter out whitespace, numbers, and non-alphabetic characters
        if token.is_space or token.like_num or not token.is_alpha:
            continue

        # Extract lemma
        lemma = token.lemma_.strip()

        # Filter tokens using the customized stop word list
        if lemma in custom_stop_words:
            continue

        # Flag the subsequent token for negation prefixing
        if lemma in ["not", "no", "never", "cannot"]:
            negate_next = True
            # Append immediately to preserve if it is the final token
            tokens.append(lemma)
            continue

        # Apply negation prefix by combining with the previous negation term
        if negate_next:
            if tokens:  # Safety check to ensure the token list is not empty
                prev_negation = tokens.pop()
                lemma = f"{prev_negation}_{lemma}"
            negate_next = False

        # Remove exceedingly short tokens unless modified by negation
        if len(lemma) < 2 and "_" not in lemma:
            continue

        tokens.append(lemma)

    return " ".join(tokens)
