# -*- coding: utf-8 -*-
import unicodedata
from .patterns import (
    RE_NONBREAKING_SPACE,
    RE_LINEBREAK,
    RE_HYPHENATED_WORD,
    QUOTE_TRANSLATION_TABLE
)
from textblob import TextBlob
from typing import Text

def normalize_case(text: Text) -> Text:
    return text.lower()

def normalize_spelling(text: Text) -> Text:
    return str(TextBlob(text).correct())


def normalize_hyphenated_words(text: Text) -> Text:
    return RE_HYPHENATED_WORD.sub(r"\1\2", text)


def normalize_quotation_marks(text: Text) -> Text:
    return text.translate(QUOTE_TRANSLATION_TABLE)


def normalize_unicode(text: Text, form="NFC") -> Text:
    return unicodedata.normalize(form, text)


def normalize_whitespace(text: Text) -> Text:
    return RE_NONBREAKING_SPACE.sub(" ", RE_LINEBREAK.sub(r"\n", text)).strip()

