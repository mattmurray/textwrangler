# -*- coding: utf-8 -*-
import unicodedata
import re
from typing import Text
from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin
from .patterns import (
    RE_NONBREAKING_SPACE,
    RE_LINEBREAK,
    QUOTE_TRANSLATION_TABLE
)

class TextNormalizer(BaseEstimator, TransformerMixin):

    def __init__(self, case=True, hyphenated_words=True, quotation_marks=True, spelling=False, unicode_characters=True,
                 whitespace=True):

        '''

        Parameters
        ----------

        :param case : default: True
        If True, all characters are converted to lowercase.

        :param hyphenated_words : default: True
        If True, hyphens in hypenated words are converted to spaces. For example,

        "High-tech" -> "High tech"
        "Data-scientist" -> "Data scientist"

        :param quotation_marks : default: True
        If True, all quotation marks are converted to standard ASCII equivalents.
        Copied from Textacy's preprocessing functionality (but without the SpaCy dependency).

        :param spelling : default: False
        If True, the correction of spelling mistakes is attempted with TextBlob's correct method.
        See https://textblob.readthedocs.io/en/dev/api_reference.html#textblob.blob.TextBlob.correct.

        :param unicode_characters : default: True
        If True, unicode characters are normalized.
        Copied from Textacy's preprocessing functionality (but without the SpaCy dependency).

        :param whitespace : default: True
        If True, cleans leading/trailing whitespace and large whitespace gaps with single spaces.
        Copied from Textacy's preprocessing functionality (but without the SpaCy dependency).
        '''

        self.case = case
        self.hyphenated_words = hyphenated_words
        self.quotation_marks = quotation_marks
        self.spelling = spelling
        self.unicode_characters = unicode_characters
        self.whitespace = whitespace

    def _normalize_case(self, text: Text) -> Text:
        return text.lower()

    def _normalize_spelling(self, text: Text) -> Text:
        return str(TextBlob(text).correct())

    def _normalize_hyphenated_words(self, text: Text) -> Text:
        return re.sub(r'([a-zA-Z])-([a-zA-Z])', r'\1 \2', text)

    def _normalize_quotation_marks(self, text: Text) -> Text:
        return text.translate(QUOTE_TRANSLATION_TABLE)

    def _normalize_unicode(self, text: Text, form="NFC") -> Text:
        return unicodedata.normalize(form, text)

    def _normalize_whitespace(self, text: Text) -> Text:
        return RE_NONBREAKING_SPACE.sub(" ", RE_LINEBREAK.sub(r"\n", text)).strip()

    def fit(self, X, y=None):
        return self

    def transform(self, text, y=None):

        output = []
        if type(text) == str:
            text = [text]

        for item in text:

            if self.spelling == True:
                item = self._normalize_spelling(item)

            if self.case == True:
                item = self._normalize_case(item)

            if self.hyphenated_words == True:
                item = self._normalize_hyphenated_words(item)

            if self.quotation_marks == True:
                item = self._normalize_quotation_marks(item)

            if self.unicode_characters == True:
                item = self._normalize_unicode(item)

            if self.whitespace == True:
                item = self._normalize_whitespace(item)

            output.append(item)

        return output
