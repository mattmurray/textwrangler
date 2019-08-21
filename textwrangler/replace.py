
from typing import Text
import contractions
import inflect
from .patterns import (
    RE_CURRENCY_SYMBOL,
    RE_EMAIL,
    RE_NUMBER,
    RE_HASHTAG,
    RE_PHONE_NUMBER,
    RE_URL,
    RE_SHORT_URL,
    RE_USER_HANDLE
)
import nltk


def replace_contractions(text: Text) -> Text:
    return contractions.fix(text)

def replace_currency_symbols(text: Text, replace_with="_CUR_") -> Text:
    return RE_CURRENCY_SYMBOL.sub(replace_with, text)


def replace_emails(text: Text, replace_with="_EMAIL_") -> Text:
    return RE_EMAIL.sub(replace_with, text)


def replace_numbers(text: Text, replace_with="_NUMBER_") -> Text:
    return RE_NUMBER.sub(replace_with, text)


def replace_hashtags(text: Text, replace_with="_TAG_") -> Text:
    return RE_HASHTAG.sub(replace_with, text)


def replace_phone_numbers(text: Text, replace_with="_PHONE_") -> Text:
    return RE_PHONE_NUMBER.sub(replace_with, text)


def replace_urls(text: Text, replace_with="_URL_") -> Text:
    return RE_URL.sub(replace_with, RE_SHORT_URL.sub(replace_with, text))


def replace_user_handles(text: Text, replace_with="_USER_") -> Text:
    return RE_USER_HANDLE.sub(replace_with, text)

def replace_numbers_with_text_representations(text: Text) -> Text:
    p = inflect.engine()
    return ' '.join([(p.number_to_words(token) if token.isdigit() else token) for token in nltk.word_tokenize(text)])




# stem words
# lemmatize words
