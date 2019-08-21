
from typing import Text

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

def replace_currency_symbols(text: Text, replace_with="_CUR_") -> Text:
    """
    Replace all currency symbols in ``text`` with ``replace_with``.
    Args:
        text (str)
        replace_with (str)
    Returns:
        str
    """
    return RE_CURRENCY_SYMBOL.sub(replace_with, text)


def replace_emails(text: Text, replace_with="_EMAIL_") -> Text:
    """
    Replace all email addresses in ``text`` with ``replace_with``.
    Args:
        text (str)
        replace_with (str)
    Returns:
        str
    """
    return RE_EMAIL.sub(replace_with, text)


def replace_numbers(text: Text, replace_with="_NUMBER_") -> Text:
    """
    Replace all numbers in ``text`` with ``replace_with``.
    Args:
        text (str)
        replace_with (str)
    Returns:
        str
    """
    return RE_NUMBER.sub(replace_with, text)


def replace_hashtags(text: Text, replace_with="_TAG_") -> Text:
    """
    Replace all hashtags in ``text`` with ``replace_with``.
    Args:
        text (str)
        replace_with (str)
    Returns:
        str
    """
    return RE_HASHTAG.sub(replace_with, text)


def replace_phone_numbers(text: Text, replace_with="_PHONE_") -> Text:
    """
    Replace all phone numbers in ``text`` with ``replace_with``.
    Args:
        text (str)
        replace_with (str)
    Returns:
        str
    """
    return RE_PHONE_NUMBER.sub(replace_with, text)


def replace_urls(text: Text, replace_with="_URL_") -> Text:
    """
    Replace all URLs in ``text`` with ``replace_with``.
    Args:
        text (str)
        replace_with (str)
    Returns:
        str
    """
    return RE_URL.sub(replace_with, RE_SHORT_URL.sub(replace_with, text))


def replace_user_handles(text: Text, replace_with="_USER_") -> Text:
    """
    Replace all user handles in ``text`` with ``replace_with``.
    Args:
        text (str)
        replace_with (str)
    Returns:
        str
    """
    return RE_USER_HANDLE.sub(replace_with, text)


# stem words
# lemmatize words
# replace contractions
