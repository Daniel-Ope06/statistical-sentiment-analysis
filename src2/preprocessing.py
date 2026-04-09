import re
import emoji
from emot.emo_unicode import EMOTICONS_EMO  # type: ignore


def preprocess_sentiment(text):
    """
    Preprocess text for neural sentiment models
    """
    # Safety check to prevent crashes on empty/NaN rows in the dataset
    if not isinstance(text, str):
        return ""

    # Change backticks to standard apostrophes
    text = text.replace("`", "'")

    # Convert standalone '@' to 'at' (e.g., "meet me @ 8")
    text = re.sub(r'(?:\s|^)@(?:\s|$)', ' at ', text)

    # Replace mentions and URLs with structural tokens
    text = re.sub(r"http\S+|www\S+|https\S+", ' urltoken ', text)
    text = re.sub(r'\@\w+', ' usertoken ', text)

    # Translate emoticons (<3 -> love)
    # Replaces spaces in the description with underscores
    for emot_char, emot_desc in EMOTICONS_EMO.items():
        if emot_char in text:
            tokenized_desc = "_".join(emot_desc.replace(",", "").split())
            text = text.replace(emot_char, f" {tokenized_desc} ")

    # Translate emojis (🔥 -> fire)
    text = emoji.demojize(text, delimiters=(" ", " "))

    # Tag hashtags (prefix with 'hashtag')
    text = re.sub(r'#(\w+)', r' hashtag \1 ', text)

    # Tag ALL CAPS words (prefix with 'allcaps')
    text = re.sub(r'\b[A-Z]{2,}\b', lambda m: f" allcaps {m.group(0).lower()} ", text)  # noqa: E501

    # Lowercase all text
    text = text.lower()

    # Preserve emotional punctuation and censored profanity
    text = re.sub(r'\*{2,}', ' censoredswear ', text)
    text = re.sub(r'!+', ' exclamationmark ', text)
    text = re.sub(r'\?+', ' questionmark ', text)

    # Strip remaining punctuation
    # EXCEPT apostrophes (contractions) & underscores (emojis/emoticons)
    text = re.sub(r'[^\w\s\'_]', ' ', text)

    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text
