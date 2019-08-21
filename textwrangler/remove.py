import string
import unidecode
from typing import Text
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

def remove_punctuation(text: Text) -> Text:
    return text.translate(str.maketrans({a: None for a in string.punctuation}))

def remove_accents(text: Text) -> Text:
    return unidecode.unidecode(text)

def remove_numbers(text: Text) -> Text:
    return text.translate({ord(k): None for k in string.digits})

def remove_html(text: Text) -> Text:
    return BeautifulSoup(text, "html.parser").get_text()

def remove_stop_words(text: Text) -> Text:
    return ' '.join(token for token in text.split() if token not in stopwords.words('english'))

