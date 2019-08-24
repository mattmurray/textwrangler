from typing import Text
import contractions
import inflect
import nltk
from sklearn.base import TransformerMixin
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

nltk.download('punkt')

class TextReplacer(TransformerMixin):

    def __init__(self, contractions=False, currency_symbols=True, emails=False, numbers=False, hashtags=False,
                 phone_numbers=False, urls=False, user_handles=False, numbers_with_text_repr=False):

        self.output = []
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

        if type(text) == str:
            self.text = [text]
        else:
            self.text = text

        for item in self.text:
            output_text = item

            if self.contractions == True:
                output_text = self._contractions(output_text)

            if self.currency_symbols == True:
                output_text = self._currency_symbols(output_text)

            if self.emails == True:
                output_text = self._emails(output_text)

            if self.numbers == True:
                output_text = self._numbers(output_text)

            if self.hashtags == True:
                output_text = self._hashtags(output_text)

            if self.phone_numbers == True:
                output_text = self._phone_numbers(output_text)

            if self.urls == True:
                output_text = self._urls(output_text)

            if self.user_handles == True:
                output_text = self._user_handles(output_text)

            if self.numbers_with_text_repr == True:
                output_text = self._numbers_with_text_repr(output_text)

            self.output.append(output_text)

        return self.output