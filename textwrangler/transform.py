# -*- coding: utf-8 -*-
from textwrangler.normalize import TextNormalizer
from textwrangler.remove import TextRemover
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from gensim.models import KeyedVectors
import nltk
import numpy as np

class FingerPrintTransformer(TextRemover, TextNormalizer, BaseEstimator, TransformerMixin):

    def __init__(self, n_gram=None, return_fingerprints=False):

        '''
        Parameters
        ----------

        :param n_gram : default: None
        If None, the standard fingerprint clustering method is applied.
        If an int is supplied, n-gram fingerprint clustering is applied.

        :param return_fingerprints : default: False
        If False, cleaned strings are returned based on the most common string in each fingerprint cluster.
        If True, the actual fingerprints are returned.

        Adapted from https://gist.github.com/cjdd3b/0386f139bb953f046c6e.
        '''

        self.n_gram = n_gram
        self.return_fingerprints = return_fingerprints

    def __unique_preserving_order(self, seq):
        return list(dict.fromkeys(seq))

    def __get_fingerprint(self, text):
        return self._accents(' '.join(self.__unique_preserving_order(sorted(text.split()))))

    def __get_ngram_fingerprint(self, text, n):
        return self._accents(''.join(self.__unique_preserving_order(sorted([text[i:i + n] for i in range(len(text) - n + 1)]))).strip())

    def __get_fingerprints(self, text):
        if type(text) == str:
            output_text = [text]
        else:
            output_text = text

        output = []
        for item in output_text:
            item = item.strip()  # remove trailing whitespace
            item = self._normalize_case(item)  # lowercase string
            item = self._normalize_unicode(item)
            item = self._normalize_quotation_marks(item)
            item = self._punctuation(item)  # remove punctuation
            if self.n_gram == None:
                item = self.__get_fingerprint(item)
            else:
                item = self.__get_ngram_fingerprint(item, self.n_gram)
            output.append(item)

        return output

    def fit(self, text, y=None):
        return self

    def transform(self, text, y=None):
        if self.return_fingerprints == True:
            return self.__get_fingerprints(text)
        else:
            fingerprint_tuples = list(zip(text, self.__get_fingerprints(text)))

            # group original text into fingerprints
            fingerprint_groups = {}
            for tup in fingerprint_tuples:
                if tup[1] in fingerprint_groups.keys():
                    fingerprint_groups[tup[1]].append(tup[0])
                else:
                    fingerprint_groups[tup[1]] = [tup[0]]

            # get the most common original string for each fingerprint
            fingerprint_most_common = {}
            for key in fingerprint_groups.keys():
                fingerprint_most_common[key] = Counter(fingerprint_groups[key]).most_common(1)[0][0]

            # transform the original strings into the most common string for each fingerprint group
            return [fingerprint_most_common[tup[1]] for tup in fingerprint_tuples]


class VectorTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, w2v_path, weighting='tfidf'):
        self.w2v_path = w2v_path
        self.vocab = None
        self.oov = None
        self.vectors = None
        self.bow_vectorizer = None
        self.weighting = weighting

    def __load_w2v(self, w2v_filepath, binary=False):
        return KeyedVectors.load_word2vec_format(w2v_filepath, binary=binary)

    def __create_vocab(self, text_list):
        word_counter = Counter()
        for s in text_list:
            word_counter.update(nltk.word_tokenize(s))
        self.vocab = word_counter

    def __check_coverage(self, vocab, model):
        known_words = {}
        unknown_words = {}
        known_word_count = 0
        unknown_word_count = 0

        for word in vocab.keys():
            try:
                known_words[word] = model[word]
                known_word_count += vocab[word]
            except:
                unknown_words[word] = vocab[word]
                unknown_word_count += vocab[word]
                pass

        print('Found embeddings for {:.2%} of vocab'.format(len(known_words) / len(vocab)))
        print('Found embeddings for {:.2%} of all text'.format(
            known_word_count / (known_word_count + unknown_word_count)))

        return known_words, unknown_words

    def __get_embedding(self, feature, model):
        try:
            return model[feature]
        except:
            return np.zeros(300, )

    def __get_document_vectors(self, text, vectors):
        bow_vecs = self.bow_vectorizer.transform(text)
        bow_vectors = bow_vecs.todense()
        vocab_embeddings = np.vstack(
            ([self.__get_embedding(feature, vectors) for feature in self.bow_vectorizer.get_feature_names()]))
        document_vectors = np.dot(bow_vectors, vocab_embeddings)
        return document_vectors

    def fit(self, text):
        if self.weighting == 'tfidf':
            self.bow_vectorizer = TfidfVectorizer(lowercase=False)
        else:
            self.bow_vectorizer = CountVectorizer(lowercase=False, binary=True)

        self.bow_vectorizer.fit(text)
        self.__create_vocab(text)
        known_words, unknown_words = self.__check_coverage(self.vocab, self.__load_w2v(self.w2v_path))
        self.oov = unknown_words
        self.vectors = known_words
        return self

    def transform(self, text):
        return self.__get_document_vectors(text, self.vectors)
