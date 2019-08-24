import string
import unidecode
from typing import Text
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

class TextRemove:

    def __init__(self, punctuation=True, accents=True, numbers=False, html=False, stop_words=False):

        self.output = []
        self.punctuation = punctuation
        self.accents = accents
        self.numbers = numbers
        self.html = html
        self.stop_words = stop_words

    def _punctuation(self, text: Text) -> Text:
        return text.translate(str.maketrans({a: None for a in string.punctuation}))

    def _accents(self, text: Text) -> Text:
        return unidecode.unidecode(text)

    def _numbers(self, text: Text) -> Text:
        return text.translate({ord(k): None for k in string.digits})

    def _html(self, text: Text) -> Text:
        return BeautifulSoup(text, "html.parser").get_text()

    def _stop_words(self, text: Text) -> Text:
        return ' '.join(token for token in text.split() if token not in stopwords.words('english'))

    def transform(self, text):

        if type(text) == str:
            self.text = [text]
        else:
            self.text = text

        for item in self.text:
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

            self.output.append(output_text)

        return self.output