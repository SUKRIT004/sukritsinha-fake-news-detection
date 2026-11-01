"""Text cleaning utilities for the Fake News Detection project."""

import re
import html


def clean_text(text: str) -> str:
    """Normalize HTML entities, URLs, whitespace and non-letter characters."""
    if text is None:
        return ""

    text = html.unescape(str(text))
    text = text.lower()
    text = re.sub(r"<[^>]+>", " ", text)              # remove HTML tags
    text = re.sub(r"https?://\S+|www\.\S+", " ", text) # remove URLs
    text = re.sub(r"\S+@\S+", " ", text)             # remove email addresses
    text = re.sub(r"[^a-z\s]", " ", text)             # retain English letters
    text = re.sub(r"\s+", " ", text).strip()
    return text
