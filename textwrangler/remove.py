

import string
import unidecode
from typing import Text
from bs4 import BeautifulSoup

def remove_punctuation(text: Text) -> Text:
    return text.translate(str.maketrans({a: None for a in string.punctuation}))

def remove_accents(text: Text) -> Text:
    return unidecode.unidecode(text)

def remove_numbers(text: Text) -> Text:
    return text.translate({ord(k): None for k in string.digits})

def remove_html(text: Text) -> Text:
    return BeautifulSoup(text, "html.parser").get_text()



# remove stop words