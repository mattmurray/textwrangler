# -*- coding: utf-8 -*-
from textwrangler.normalize import TextNormalize
from textwrangler.remove import TextRemove
from collections import Counter

class TextTransform(TextRemove, TextNormalize):

    def _unique_preserving_order(self, seq):
        '''
        Returns unique tokens in a list, preserving order. Fastest version found in this
        exercise: http://www.peterbe.com/plog/uniqifiers-benchmark
        '''
        seen = set()
        seen_add = seen.add
        return [x for x in seq if not (x in seen or seen_add(x))]

    def _get_fingerprint(self, text):
        '''
        Gets conventional fingerpint.
        '''
        return self._accents(' '.join(self._unique_preserving_order(sorted(text.split()))))

    def get_fingerprints(self, text):
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
            item = self._get_fingerprint(item)
            output.append(item)

        return output

    def transform(self, text, return_fingerprints=True):

        if return_fingerprints == True:
            return self.get_fingerprints(text)
        else:
            fingerprint_tuples = list(zip(text, self.get_fingerprints(text)))

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



    def get_ngram_fingerprint(self, text, n=1):
        '''
        Gets ngram fingerpint based on n-length shingles of the string.
        Default is 1.
        '''
        return self._accents(''.join(self._unique_preserving_order(sorted([self.text[i:i + n] for i in range(len(text) - n + 1)]))))

