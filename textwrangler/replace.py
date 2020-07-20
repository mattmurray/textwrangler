# -*- coding: utf-8 -*-
from typing import Text
import contractions
import inflect
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
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


class TextReplacer(BaseEstimator, TransformerMixin):
    '''
    Parameters
    ----------

    contractions : default: False
        If True, contractions of tokens are resolved using the contractions library. For example:

        "He's" -> "He is".

        See https://github.com/kootenpv/contractions.

    currency_symbols : default: False
        If True, currency symbols are replaced with " _CUR_ ".

    emails : default: False
        If True, email addresses are replaced with " _EMAIL_ ".

    hashtags : default: False
        If True, Twitter hashtags are replaced with " _TAG_ ".

    numbers : default: False
        If True, numerical tokens are replaced with " _NUMBER_ ".

    numbers_with_text_repr : default: False
        If True, numerical representations are replaced with text representations with the inflect library. For example:

        "12" -> "Twelve".

        See https://github.com/jazzband/inflect.

    phone_numbers : default: False
        If True, phone numbers are replaced with " _PHONE_ ".

    urls : default: False
        If True, URLs are replaced with " _URL_ ".

    user_handles : default: False
        If True, Twitter user handles are replaced with " _USER_ ".
    '''

    def __init__(self, contractions=False, currency_symbols=False, emails=False, hashtags=False, numbers=False,
                 numbers_with_text_repr=False, phone_numbers=False, urls=False, user_handles=False):

        self.contractions = contractions
        self.currency_symbols = currency_symbols
        self.emails = emails
        self.numbers = numbers
        self.hashtags = hashtags
        self.phone_numbers = phone_numbers
        self.urls = urls
        self.user_handles = user_handles
        self.numbers_with_text_repr = numbers_with_text_repr

    def _contractions(self, text: Text) -> Text:
        return contractions.fix(text)

    def _currency_symbols(self, text: Text, replace_with="_CUR_") -> Text:
        return RE_CURRENCY_SYMBOL.sub(replace_with, text)

    def _emails(self, text: Text, replace_with="_EMAIL_") -> Text:
        return RE_EMAIL.sub(replace_with, text)

    def _numbers(self, text: Text, replace_with="_NUMBER_") -> Text:
        return RE_NUMBER.sub(replace_with, text)

    def _hashtags(self, text: Text, replace_with="_TAG_") -> Text:
        return RE_HASHTAG.sub(replace_with, text)

    def _phone_numbers(self, text: Text, replace_with="_PHONE_") -> Text:
        return RE_PHONE_NUMBER.sub(replace_with, text)

    def _urls(self, text: Text, replace_with="_URL_") -> Text:
        return RE_URL.sub(replace_with, RE_SHORT_URL.sub(replace_with, text))

    def _user_handles(self, text: Text, replace_with="_USER_") -> Text:
        return RE_USER_HANDLE.sub(replace_with, text)

    def _numbers_with_text_repr(self, text: Text) -> Text:
        p = inflect.engine()
        return ' '.join([(p.number_to_words(token) if token.isdigit() else token) for token in nltk.word_tokenize(text)])

    def fit(self, X, y=None):
        return self

    def transform(self, text, y=None):

        output = []
        if type(text) == str:
            text = [text]

        for item in text:

            if self.contractions == True:
                item = self._contractions(item)

            if self.currency_symbols == True:
                item = self._currency_symbols(item)

            if self.emails == True:
                item = self._emails(item)

            if self.numbers == True:
                item = self._numbers(item)

            if self.hashtags == True:
                item = self._hashtags(item)

            if self.phone_numbers == True:
                item = self._phone_numbers(item)

            if self.urls == True:
                item = self._urls(item)

            if self.user_handles == True:
                item = self._user_handles(item)

            if self.numbers_with_text_repr == True:
                item = self._numbers_with_text_repr(item)

            output.append(item)

        return output