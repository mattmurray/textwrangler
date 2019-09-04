# -*- coding: utf-8 -*-
import string
from typing import Text
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.base import BaseEstimator, TransformerMixin
import unicodedata
from textwrangler import TextNormalizer

class TextRemover(TextNormalizer, BaseEstimator, TransformerMixin):
    '''
    Parameters
    ----------

    :param accents : default: True
    If True, removes all accents from characters. For example, 'Café' -> 'Cafe'.

    :param html : default: False
    If True, strips HTML tags from the text using BeautifulSoup.

    :param numbers : default: False
    If True, removes all numerical characters from the string.

    :param punctuation : default: True
    If True, removes all punctuation characters from the string.

    :param stop_words : default: False
    If True, removes all stop words from the string.
    '''

    def __init__(self, accents=True, html=False, numbers=False, punctuation=True, stop_words=False):
        self.punctuation = punctuation
        self.accents = accents
        self.numbers = numbers
        self.html = html
        self.stop_words = stop_words

    def _punctuation(self, text: Text) -> Text:
        return text.translate(str.maketrans({a: ' ' for a in string.punctuation}))

    def _accents(self, text: Text) -> Text:
        return unicodedata.normalize('NFD', text).encode('ascii', 'ignore').decode("utf-8")

    def _numbers(self, text: Text) -> Text:
        return text.translate({ord(k): None for k in string.digits})

    def _html(self, text: Text) -> Text:
        return BeautifulSoup(text, "html.parser").get_text()

    def _stop_words(self, text: Text) -> Text:
        return ' '.join(token for token in text.split() if token not in stopwords.words('english'))

    def fit(self, X, y=None):
        return self

    def transform(self, text, y=None):

        output = []
        if type(text) == str:
            text = [text]

        for item in text:

            if self.punctuation == True:
                item = self._punctuation(item)

            if self.accents == True:
                item = self._accents(item)

            if self.numbers == True:
                item = self._numbers(item)

            if self.html == True:
                item = self._html(item)

            if self.stop_words == True:
                item = self._stop_words(item)

            item = self._normalize_whitespace(item)
            output.append(item)

        return output