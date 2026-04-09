from transformers import DistilBertTokenizer
from tensorflow.keras.preprocessing.text import Tokenizer  # type: ignore
from tensorflow.keras.preprocessing.sequence import (  # type: ignore
    pad_sequences)


# Word-Level Tokenization
def extract_bilstm_features(train_texts, val_texts, test_texts, max_vocab=20000, max_len=50):  # noqa: E501
    """
    Fits a word-level tokenizer for the BiLSTM, converts text to sequences,
    and applies post-padding/truncation to ensure uniform tensor shapes.

    Args:
        train_texts (list or pd.Series): Raw text strings for training.
        val_texts (list or pd.Series): Raw text strings for validation.
        test_texts (list or pd.Series): Raw text strings for testing.
        max_vocab (int): Maximum number of unique words to keep in the vocabulary.
        max_len (int): The exact length every sequence will be padded/truncated to.

    Returns:
        bilstm_tokenizer (Tokenizer): The fitted Keras Tokenizer object (contains word index).
        train_padded (numpy.ndarray): 2D array of padded integer sequences for training.
        val_padded (numpy.ndarray): 2D array of padded integer sequences for validation.
        test_padded (numpy.ndarray): 2D array of padded integer sequences for testing.
    """  # noqa: E501

    # Initialize Word-Level Tokenizer with OOV token to preserve structure
    bilstm_tokenizer = Tokenizer(num_words=max_vocab, oov_token="<OOV>")
    bilstm_tokenizer.fit_on_texts(train_texts)

    # Convert strings to integer sequences
    train_seq = bilstm_tokenizer.texts_to_sequences(train_texts)
    val_seq = bilstm_tokenizer.texts_to_sequences(val_texts)
    test_seq = bilstm_tokenizer.texts_to_sequences(test_texts)

    # Pad sequences (post-padding ensures the LSTM reads real words first)
    train_padded = pad_sequences(
        train_seq, maxlen=max_len, padding='post', truncating='post')
    val_padded = pad_sequences(
        val_seq, maxlen=max_len, padding='post', truncating='post')
    test_padded = pad_sequences(
        test_seq, maxlen=max_len, padding='post', truncating='post')

    return bilstm_tokenizer, train_padded, val_padded, test_padded


# Subword Tokenization
def extract_distilbert_features(train_texts, val_texts, test_texts, max_len=50):  # noqa: E501
    """
    Uses HuggingFace's pretrained WordPiece tokenizer.
    Outputs input_ids and attention_masks required by Transformer architectures.

    Args:
        train_texts (list or pd.Series): Raw text strings for training.
        val_texts (list or pd.Series): Raw text strings for validation.
        test_texts (list or pd.Series): Raw text strings for testing.
        max_len (int): The exact length every sequence will be padded/truncated to.

    Returns:
        bert_tokenizer (DistilBertTokenizer): The pretrained HuggingFace tokenizer.
        train_encodings (dict): Contains 'input_ids' and 'attention_mask' TensorFlow tensors for training.
        val_encodings (dict): Contains 'input_ids' and 'attention_mask' TensorFlow tensors for validation.
        test_encodings (dict): Contains 'input_ids' and 'attention_mask' TensorFlow tensors for testing.
    """  # noqa: E501

    # Load the pretrained subword tokenizer matching the model
    bert_tokenizer = DistilBertTokenizer.from_pretrained(
        'distilbert-base-uncased')

    def tokenize_for_bert(texts):
        # Convert pandas series to list if necessary
        text_list = texts.tolist() if hasattr(texts, 'tolist') else texts

        # Tokenize returning TensorFlow tensors ('tf')
        return bert_tokenizer(
            text_list,
            padding='max_length',
            truncation=True,
            max_length=max_len,
            return_tensors='tf'
        )

    train_encodings = tokenize_for_bert(train_texts)
    val_encodings = tokenize_for_bert(val_texts)
    test_encodings = tokenize_for_bert(test_texts)

    return bert_tokenizer, train_encodings, val_encodings, test_encodings
