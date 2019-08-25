import string
import unidecode
from typing import Text
from bs4 import BeautifulSoup
from nltk.corpus import stopwords
from sklearn.base import TransformerMixin
import unicodedata
from textwrangler import TextNormalizer

class TextRemover(TextNormalizer, TransformerMixin):

    def __init__(self, punctuation=True, accents=True, numbers=False, html=False, stop_words=False):

        self.punctuation = punctuation
        self.accents = accents
        self.numbers = numbers
        self.html = html
        self.stop_words = stop_words

    def _punctuation(self, text: Text) -> Text:
        return text.translate(str.maketrans({a: ' ' for a in string.punctuation}))

    def _accents(self, text: Text) -> Text:
        # return unidecode.unidecode(text)
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
            output_text = item

            if self.punctuation == True:
                output_text = self._punctuation(output_text)

            if self.accents == True:
                output_text = self._accents(output_text)

            if self.numbers == True:
                output_text = self._numbers(output_text)

            if self.html == True:
                output_text = self._html(output_text)

            if self.stop_words == True:
                output_text = self._stop_words(output_text)

            output_text = self._normalize_whitespace(output_text)

            output.append(output_text)

        return output