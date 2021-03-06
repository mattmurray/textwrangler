# -*- coding: utf-8 -*-
from typing import Text, Dict
from nltk.corpus import stopwords
import textstat
import string
from langdetect import detect, detect_langs
from textblob import TextBlob
from sklearn.base import BaseEstimator, TransformerMixin
from better_profanity import profanity
import multiprocessing as mp
import traceback

class TextFeatureExtractor(BaseEstimator, TransformerMixin):
    '''
    Parameters
    ----------

    average_token_size : default: False
        If True, returns the mean character length of the tokens in the string.

    contains_profanity : default: False
        If True, checks for profanity in the string using the better-profanity library.

        See https://github.com/snguyenthanh/better_profanity.

    exclamation_mark_count : default: False
        If True, counts the number of exclamation marks in the string.

    language : default: False
        If True, detects the language of the string using the langdetect library, a port of the Google
        language-detection library. Note that by default it is set to False because it is slow to compute.

        See https://github.com/Mimino666/langdetect.

    number_of_unique_tokens : default: False
        If True, counts the number of unique tokens in the string.

    numerical_token_count : default: False
        If True, counts the number of tokens in the string that are numbers.

    numerical_token_proportion : default: False
        If True, calculates the proportion of all tokens in the string that are numbers.

    n_jobs : default: 1
        The number of cores to use when processing.

    polarity : default: False
        If True, returns the polarity score calculated with TextBlob sentiment analysis. The polarity score is a
        float ranging from -1.0 to +1.0.

        See https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis.

    punctuation_character_count : default: False
        If True, returns the number of punctuation characters in the string.

    punctuation_proportion : default: False
        If True, returns the amount of punctuation in the string as a proportion of the total character count.

    question_mark_count : default: False
        If True, counts the number of question marks in the string.

    readability_scores : default: False
        If True, returns a set of readability scores as calculated by the textstat library.

        See https://github.com/shivam5992/textstat.

    stop_word_count : default: False
        If True, returns the number of tokens that are stop words.

    stop_word_proportion : default: False
        If True, returns the proportion of all the tokens in the string that are stop words.

    string_length : default: False
        If True, returns the total number of characters in the string, excluding white space.

    subjectivity : default: False
        If True, returns the subjectiity score score calculated with TextBlob sentiment analysis. The subjectivity score
        is a float ranging from 0.0 (very objective/factual) to 1.0 (very subjective/opinionated).

        See https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis.

    title_token_count : default: False
        If True, counts the number of tokens in the string with a capitalised first character. For example:

        * 'London'
        * 'Government'
        * 'Mr'

    title_token_proportion : default: False
        If True, returns the proportion of all the tokens in the string that have a capitalised first character.

    token_count : default: False
        If True, returns a count of the number of tokens in the string.

    upper_token_count : default: False
        If True, returns the count of tokens in the string where all characters are upper cased. For example:

        * 'GREAT'
        * 'NO'

    upper_token_proportion : default: False
        If True, returns the proportion of all the tokens in the string that are upper cased.

    unique_token_proportion : default: False
        If True, returns the proportion of all the tokens in the string that are unique.

    '''

    def __init__(self, average_token_size=False, contains_profanity=False, exclamation_mark_count=False, language=False,
                 number_of_unique_tokens=False, numerical_token_count=False, numerical_token_proportion=False, n_jobs=1,
                 polarity=False, punctuation_character_count=False, punctuation_proportion=False, question_mark_count=False,
                 readability_scores=False, stop_word_count=False, stop_word_proportion=False, string_length=False,
                 subjectivity=False, title_token_count=False, title_token_proportion=False, token_count=False,
                 upper_token_count=False, upper_token_proportion=False, unique_token_proportion=False):

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
            traceback.print_exc()
            print()
            raise e

    def transform(self, text, y=None):
        if type(text) == str:
            text = [text]

        pool = mp.Pool(processes=self.n_jobs)
        results = [pool.map(self._process_item, text)]
        return results[0]

