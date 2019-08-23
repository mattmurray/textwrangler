from typing import Text, Dict, List
import nltk
from nltk.corpus import stopwords
import numpy as np
import os
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors
import textstat
from langdetect import detect, detect_langs
from textblob import TextBlob

class FeatureExtraction:

    def __init__(self, token_count=True, string_length=True, average_token_size=True,
                 stop_word_count=True, numerical_token_count=True, upper_token_count=True,
                 readability_scores=True, language=True, polarity=True, subjectivity=True, title_token_count=True):

        self.output = []
        self.text = None
        self.token_count = token_count
        self.string_length = string_length
        self.average_token_size = average_token_size
        self.stop_word_count = stop_word_count
        self.numerical_token_count = numerical_token_count
        self.upper_token_count = upper_token_count
        self.readability_scores = readability_scores
        self.language = language
        self.polarity = polarity
        self.subjectivity = subjectivity
        self.title_token_count = title_token_count

    def _extract_token_count(self, text: Text) -> Dict:
        return {'token_count': len(text.split())}

    def _extract_string_length(self, text: Text, exclude_whitespace=True) -> Dict:
        if exclude_whitespace == True:
            return {'string_length': len(''.join(text.split()))}
        else:
            return {'string_length': len(text)}

    def _extract_average_token_size(self, text: Text) -> Dict:
      tokens = text.split()
      return {'average_token_size': sum(len(token) for token in tokens)/len(tokens)}

    def _extract_stop_word_count(self, text: Text) -> Dict:
        # nltk.download('stopwords')
        return {'stop_word_count': len([token for token in text.split() if token in stopwords.words('english')])}

    def _extract_numerical_token_count(self, text: Text) -> Dict:
        return {'numerical_token_count': len([token for token in text.split() if token.isdigit()])}

    def _extract_upper_token_count(self, text: Text) -> Dict:
        return {'upper_token_count': len([token for token in text.split() if token.isupper()])}

    def _extract_title_token_count(self, text: Text) -> Dict:
        return {'title_token_count': len([token for token in text.split() if token.istitle()])}

    def _extract_polarity(self, text: Text) -> Dict:
        return {'polarity': TextBlob(text).sentiment.polarity}

    def _extract_subjectivity(self, text: Text) -> Dict:
        return {'subjectivity': TextBlob(text).sentiment.subjectivity}

    def _extract_readability_scores(self, text: Text, scores=None) -> Dict:

        output = {}
        if scores == None or 'flesch_reading_ease' in scores:
            output['flesch_reading_ease'] = textstat.flesch_reading_ease(text)

        if scores == None or 'smog_index' in scores:
            output['smog_index'] = textstat.smog_index(text)

        if scores == None or 'flesch_kincaid_grade' in scores:
            output['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)

        if scores == None or 'coleman_liau_index' in scores:
            output['coleman_liau_index'] = textstat.coleman_liau_index(text)

        if scores == None or 'automated_readability_index' in scores:
            output['automated_readability_index'] = textstat.automated_readability_index(text)

        if scores == None or 'dale_chall_readability_score' in scores:
            output['dale_chall_readability_score'] = textstat.dale_chall_readability_score(text)

        if scores == None or 'difficult_words' in scores:
            output['difficult_words'] = textstat.difficult_words(text)

        if scores == None or 'linsear_write_formula' in scores:
            output['linsear_write_formula'] = textstat.linsear_write_formula(text)

        if scores == None or 'gunning_fog' in scores:
            output['gunning_fog'] = textstat.gunning_fog(text)

        if scores == None or 'text_standard' in scores:
            output['text_standard'] = textstat.text_standard(text, float_output=True)

        return output

    def _extract_language(self, text: Text, output_probabilities=True) -> Dict:
        if output_probabilities == False:
            return {'lang': detect(text)}
        else:
            return {f'lang_{item.lang}': item.prob for item in detect_langs(text)}

    def transform(self, text):

        self.output = []

        if type(text) == str:
            self.text = [text]
        else:
            self.text = text

        for item in self.text:
            output = {}

            if self.token_count == True:
                output = {**output, **self._extract_token_count(item)}

            if self.string_length == True:
                output = {**output, **self._extract_string_length(item)}

            if self.average_token_size == True:
                output = {**output, **self._extract_average_token_size(item)}

            if self.stop_word_count == True:
                output = {**output, **self._extract_stop_word_count(item)}

            if self.numerical_token_count == True:
                output = {**output, **self._extract_numerical_token_count(item)}

            if self.upper_token_count == True:
                output = {**output, **self._extract_upper_token_count(item)}

            if self.title_token_count == True:
                output = {**output, **self._extract_title_token_count(item)}

            if self.readability_scores == True:
                output = {**output, **self._extract_readability_scores(item)}

            if self.language == True:
                output = {**output, **self._extract_language(item)}

            if self.polarity == True:
                output = {**output, **self._extract_polarity(item)}

            if self.subjectivity == True:
                output = {**output, **self._extract_subjectivity(item)}

            self.output.append(output)

        return self.output


# extract people / count names
# emoji count
# easy data augmentation / generate augmentations


    # def _download_embeddings(embeddings_path="../embeddings"):
    #     if len(os.listdir(embeddings_path)) == 0:
    #         with urlopen('http://nlp.stanford.edu/data/glove.6B.zip') as url:
    #             with ZipFile(BytesIO(url.read())) as zfile:
    #                 zfile.extractall(embeddings_path)
    #
    # def _create_word2vec_files(embeddings_path="../embeddings"):
    #     embeddings_files = os.listdir(embeddings_path)
    #     w2v_files = os.listdir('../resources')
    #     if len(w2v_files) == 0 and len(embeddings_files) > 0:
    #         for emb_file in embeddings_files:
    #             glove2word2vec(f'{embeddings_path}/{emb_file}', f'./resources/{emb_file}.resources')
    #
    # def extract_word_vectors(text: Text, vector_dim=100) -> np.ndarray:
    #     w2v_file = f'resources/glove.6B.{vector_dim}d.txt.resources'
    #     if w2v_file not in os.listdir('../resources'):
    #         _download_embeddings()
    #         _create_word2vec_files()
    #
    #     model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    #     vectors = []
    #     for token in text.split():
    #         vectors.append(model[token])
    #     return sum(vectors)/len(vectors)