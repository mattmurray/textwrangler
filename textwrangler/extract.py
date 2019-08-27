# -*- coding: utf-8 -*-
from typing import Text, Dict, List
from tqdm import tqdm
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
import string
from langdetect import detect, detect_langs
from textblob import TextBlob
from sklearn.base import TransformerMixin
from better_profanity import profanity
import multiprocessing as mp
import traceback
from gensim.models import Word2Vec

class TextFeatureExtractor(TransformerMixin):

    def __init__(self, n_jobs=1, token_count=True, string_length=True, average_token_size=True,
                 stop_word_count=True, numerical_token_count=True, upper_token_count=True,
                 readability_scores=True, language=False, polarity=True, subjectivity=True, title_token_count=True,
                 unique_token_proportion=True, number_of_unique_tokens=True, question_mark_count=True,
                 exclamation_mark_count=True, title_token_proportion=True, upper_token_proportion=True,
                 numerical_token_proportion=True, stop_word_proportion=True, punctuation_character_count=True,
                 punctuation_proportion=True, contains_profanity=True):

        self.n_jobs = n_jobs
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
        self.exclamation_mark_count = exclamation_mark_count
        self.question_mark_count = question_mark_count
        self.number_of_unique_tokens = number_of_unique_tokens
        self.unique_token_proportion = unique_token_proportion
        self.stop_word_proportion = stop_word_proportion
        self.numerical_token_proportion = numerical_token_proportion
        self.title_token_proportion = title_token_proportion
        self.upper_token_proportion = upper_token_proportion
        self.punctuation_character_count = punctuation_character_count
        self.punctuation_proportion = punctuation_proportion
        self.contains_profanity = contains_profanity

    def _extract_profanity_check(self, text: Text) -> Dict:
        return {'contains_profanity': int(profanity.contains_profanity(text))}

    def _extract_token_count(self, text: Text) -> Dict:
        return {'token_count': len(text.split())}

    def _extract_punctuation_character_count(self, text: Text) -> Dict:
        return {'punctuation_character_count': len([c for c in text if c in string.punctuation])}

    def _extract_string_length(self, text: Text, exclude_whitespace=True) -> Dict:
        if exclude_whitespace == True:
            return {'string_length': len(''.join(text.split()))}
        else:
            return {'string_length': len(text)}

    def _extract_average_token_size(self, text: Text) -> Dict:
      tokens = text.split()
      if len(tokens) == 0:
          return {'average_token_size': 0}
      else:
          return {'average_token_size': sum(len(token) for token in tokens)/len(tokens)}

    def _extract_stop_word_count(self, text: Text) -> Dict:
        # nltk.download('stopwords')
        return {'stop_word_count': len([token for token in text.split() if token in stopwords.words('english')])}

    def _extract_numerical_token_count(self, text: Text) -> Dict:
        return {'numerical_token_count': len([token for token in text.split() if token.isdigit()])}

    def _extract_upper_token_count(self, text: Text) -> Dict:
        return {'upper_token_count': len([token for token in text.split() if token.isupper()])}

    def _extract_exclamation_mark_count(self, text: Text) -> Dict:
        return {'exclamation_mark_count': text.count('!')}

    def _extract_question_mark_count(self, text: Text) -> Dict:
        return {'question_mark_count': text.count('?')}

    def _extract_number_of_unique_tokens(self, text: Text) -> Dict:
        return {'number_of_unique_tokens': len(set(w for w in text.split()))}

    def _extract_unique_token_proportion(self, text: Text) -> Dict:
        number_of_unique_tokens = self._extract_number_of_unique_tokens(text)['number_of_unique_tokens']
        token_count = self._extract_token_count(text)['token_count']
        if number_of_unique_tokens == 0 or token_count == 0:
            return {'unique_token_proportion': 0}
        else:
            return {'unique_token_proportion': float(number_of_unique_tokens / token_count)}

    def _extract_title_token_count(self, text: Text) -> Dict:
        return {'title_token_count': len([token for token in text.split() if token.istitle()])}

    def _extract_polarity(self, text: Text) -> Dict:
        return {'polarity': TextBlob(text).sentiment.polarity}

    def _extract_subjectivity(self, text: Text) -> Dict:
        return {'subjectivity': TextBlob(text).sentiment.subjectivity}

    def _extract_stop_word_proportion(self, text: Text) -> Dict:
        token_count = self._extract_token_count(text)['token_count']
        stop_word_count = self._extract_stop_word_count(text)['stop_word_count']
        if stop_word_count == 0 or token_count == 0:
            return {'stop_word_proportion': 0}
        else:
            return {'stop_word_proportion': float(stop_word_count / token_count)}

    def _extract_numerical_token_proportion(self, text: Text) -> Dict:
        token_count = self._extract_token_count(text)['token_count']
        numerical_token_count = self._extract_numerical_token_count(text)['numerical_token_count']
        if numerical_token_count == 0 or token_count == 0:
            return {'numerical_token_proportion': 0}
        else:
            return {'numerical_token_proportion': float(numerical_token_count / token_count)}

    def _extract_upper_token_proportion(self, text: Text) -> Dict:
        token_count = self._extract_token_count(text)['token_count']
        upper_token_count = self._extract_upper_token_count(text)['upper_token_count']
        if upper_token_count == 0 or token_count == 0:
            return {'upper_token_proportion': 0}
        else:
            return {'upper_token_proportion': float(upper_token_count / token_count)}

    def _extract_title_token_proportion(self, text: Text) -> Dict:
        token_count = self._extract_token_count(text)['token_count']
        title_token_count = self._extract_title_token_count(text)['title_token_count']
        if title_token_count == 0 or token_count == 0:
            return {'title_token_proportion': 0}
        else:
            return {'title_token_proportion': float(title_token_count/token_count)}

    def _extract_punctuation_proportion(self, text: Text) -> Dict:
        string_length = self._extract_string_length(text, exclude_whitespace=True)['string_length']
        punctuation_character_count = self._extract_punctuation_character_count(text)['punctuation_character_count']
        if punctuation_character_count == 0 or string_length == 0:
            return {'punctuation_character_proportion': 0}
        else:
            return {'punctuation_character_proportion': float(punctuation_character_count/string_length)}

    # def _extract_sentence_count(self, text: Text) -> Dict:
    #     pass

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

    def fit(self, X, y=None):
        return self


    def _process_item(self, item):

        try:
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

            if self.title_token_count == True:
                output = {**output, **self._extract_title_token_count(item)}

            if self.exclamation_mark_count == True:
                output = {**output, **self._extract_exclamation_mark_count(item)}

            if self.question_mark_count == True:
                output = {**output, **self._extract_question_mark_count(item)}

            if self.number_of_unique_tokens == True:
                output = {**output, **self._extract_number_of_unique_tokens(item)}

            if self.unique_token_proportion == True:
                output = {**output, **self._extract_unique_token_proportion(item)}

            if self.title_token_proportion == True:
                output = {**output, **self._extract_title_token_proportion(item)}

            if self.upper_token_proportion == True:
                output = {**output, **self._extract_upper_token_proportion(item)}

            if self.numerical_token_proportion == True:
                output = {**output, **self._extract_numerical_token_proportion(item)}

            if self.stop_word_proportion == True:
                output = {**output, **self._extract_stop_word_proportion(item)}

            if self.punctuation_character_count == True:
                output = {**output, **self._extract_punctuation_character_count(item)}

            if self.punctuation_proportion == True:
                output = {**output, **self._extract_punctuation_proportion(item)}

            if self.contains_profanity == True:
                output = {**output, **self._extract_profanity_check(item)}

            return output

        except Exception as e:
            print('Caught exception in worker thread (item: \n{}):'.format(item))

            # This prints the type, value, and stack trace of the
            # current exception being handled.
            traceback.print_exc()
            print()
            raise e

    def transform(self, text, y=None):

        if type(text) == str:
            text = [text]

        pool = mp.Pool(processes=self.n_jobs)
        results = [pool.map(self._process_item, text)]
        return results[0]

class TrainWordToVec:

    def __init__(self, size=150, window=10, min_count=2, workers=10, iter=10, skip_gram=0):

        '''

        :param size:
        The size of the dense vector to represent each token or word (i.e. the context or neighboring words).
        If you have limited data, then size should be a much smaller value since you would only have so many
        unique neighbors for a given word. If you have lots of data, it’s good to experiment with various sizes.
        A value of 100–150 has worked well for me for similarity lookups.

        :param window:
        The maximum distance between the target word and its neighboring word. If your neighbor’s position is
        greater than the maximum window width to the left or the right, then, some neighbors would not be considered
        as being related to the target word. In theory, a smaller window should give you terms that are more related.
        Again, if your data is not sparse, then the window size should not matter too much, as long as it’s not overly
        narrow or overly broad. If you are not too sure about this, just use the default value.

        :param min_count:
        Minimium frequency count of words. The model would ignore words that do not satisfy the min_count. Extremely
        infrequent words are usually unimportant, so its best to get rid of those. Unless your dataset is really tiny,
        this does not really affect the model in terms of your final results. The settings here probably has more of
        an effect on memory usage and storage requirements of the model files.

        :param workers:
        How many threads to use.

        :param iter:
        Number of iterations (epochs) over the corpus. 5 is a good starting point. I always use a minimum of 10
        iterations.

        '''
        self.size = size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.iter = iter
        self.skip_gram = skip_gram
        self.model = None

    def fit(self, documents):
        self.model = Word2Vec(size=self.size, window=self.window, min_count=self.min_count,
                         workers=self.workers, iter=self.iter, sg=self.skip_gram)

        self.model.build_vocab(documents)
        self.model.train(documents, total_examples=self.model.corpus_count, epochs=self.model.iter)


    def save_model(self, path):
        self.model.save(path)

    def save_keyed_vectors(self, path):
        self.model.wv.save(path)

    def get_most_similar(self, word, top_n=5, positive=True):
        if self.model is not None:
            if positive == True:
                return self.model.wv.most_similar(positive=[word], topn=top_n)
            else:
                return self.model.wv.most_similar(negative=[word], topn=top_n)
        else:
            return "No trained model."


# extract people / count names
# emoji count
# easy data augmentation / generate augmentations