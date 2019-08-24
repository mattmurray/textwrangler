from textblob import TextBlob
from typing import Text
import unicodedata
from .patterns import (
    RE_NONBREAKING_SPACE,
    RE_LINEBREAK,
    RE_HYPHENATED_WORD,
    QUOTE_TRANSLATION_TABLE
)
from sklearn.base import TransformerMixin

class TextNormalizer(TransformerMixin):

    def __init__(self, case=True, spelling=False, hyphenated_words=True, quotation_marks=True, unicode=True,
                 whitespace=True):

        self.output = []
        self.text = None
        self.case = case
        self.spelling = spelling
        self.hyphenated_words = hyphenated_words
        self.quotation_marks = quotation_marks
        self.unicode = unicode
        self.whitespace = whitespace

    def _normalize_case(self, text: Text) -> Text:
        return text.lower()

    def _normalize_spelling(self, text: Text) -> Text:
        return str(TextBlob(text).correct())

    def _normalize_hyphenated_words(self, text: Text) -> Text:
        return RE_HYPHENATED_WORD.sub(r"\1\2", text)

    def _normalize_quotation_marks(self, text: Text) -> Text:
        return text.translate(QUOTE_TRANSLATION_TABLE)

    def _normalize_unicode(self, text: Text, form="NFC") -> Text:
        return unicodedata.normalize(form, text)

    def _normalize_whitespace(self, text: Text) -> Text:
        return RE_NONBREAKING_SPACE.sub(" ", RE_LINEBREAK.sub(r"\n", text)).strip()

    def fit(self, X, y=None):
        return self

    def transform(self, text, y=None):

        if type(text) == str:
            self.text = [text]
        else:
            self.text = text

        for item in self.text:
            output_text = item

            if self.spelling == True:
                output_text = self._normalize_spelling(output_text)

            if self.case == True:
                output_text = self._normalize_case(output_text)

            if self.hyphenated_words == True:
                output_text = self._normalize_hyphenated_words(output_text)

            if self.quotation_marks == True:
                output_text = self._normalize_quotation_marks(output_text)

            if self.unicode == True:
                output_text = self._normalize_unicode(output_text)

            if self.whitespace == True:
                output_text = self._normalize_whitespace(output_text)

            self.output.append(output_text)

        return self.output
