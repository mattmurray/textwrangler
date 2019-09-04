# -*- coding: utf-8 -*-
from textwrangler.normalize import TextNormalizer
from textwrangler.remove import TextRemover
from collections import Counter
from sklearn.base import BaseEstimator, TransformerMixin

# Adapted from https://gist.github.com/cjdd3b/0386f139bb953f046c6e.
class FingerPrintTransformer(TextRemover, TextNormalizer, BaseEstimator, TransformerMixin):
    '''
    Parameters
    ----------

    n_gram : default: None
        If None, the standard fingerprint clustering method is applied.
        If an int is supplied, n-gram fingerprint clustering is applied.

    return_fingerprints : default: False
        If False, cleaned strings are returned based on the most common string in each fingerprint cluster.
        If True, the actual fingerprints are returned.
    '''

    def __init__(self, n_gram=None, return_fingerprints=False):

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


