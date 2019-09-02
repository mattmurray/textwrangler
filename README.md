# textwrangler

A small library making use of some widely used NLP tools for simple text cleaning tasks. 

### Dependencies

- beautifulsoup4
- better_profanity
- contractions
- langdetect
- textblob
- textstat
- textsearch
- inflect
- nltk
- numpy
- scikit-learn
- unidecode


### Installation

```
pip install textwrangler
```

### Usage

There are currently five classes for wrangling text:

```python
from textwrangler import TextFeatureExtractor, TextNormalizer, TextRemover, TextReplacer, FingerPrintTransformer
```

### Example

A simple example with a small list of strings:

```python
text = ["Five weeks ago I spoke to you from these steps and said that this government wasn't going to hang around and that we would not wait until Brexit day - October 31 - to deliver on the priorities of the British people.",
"And so I am proud to say that on Wednesday Chancellor Sajid Javid is going to set out the most ambitious spending round for more than a decade.",
"I said I wanted to make your streets safer – and that is why we are recruiting another 20,000 police officers."]
```

#### Extracting features

```python
feature_extractor = TextFeatureExtractor()
text_features = feature_extractor.transform(text)

print(text_features[:2])
>> [{'token_count': 42,
>>   'string_length': 174,
>>   'average_token_size': 4.142857142857143,
>>   'stop_word_count': 19,
>>   'numerical_token_count': 1,
>>   'upper_token_count': 1,
>>   'title_token_count': 5,
>>   'flesch_reading_ease': 47.8,
>>   'smog_index': 0.0,
>>   'flesch_kincaid_grade': 16.5,
>>   'coleman_liau_index': 7.96,
>>   'automated_readability_index': 19.1,
>>   'dale_chall_readability_score': 6.8,
>>   'difficult_words': 3,
>>   'linsear_write_formula': 25.0,
>>   'gunning_fog': 18.0,
>>   'text_standard': 17.0,
>>   'polarity': 0.0,
>>   'subjectivity': 0.0,
>>   'exclamation_mark_count': 0,
>>   'question_mark_count': 0,
>>   'number_of_unique_tokens': 36,
>>   'unique_token_proportion': 0.8571428571428571,
>>   'title_token_proportion': 0.11904761904761904,
>>   'upper_token_proportion': 0.023809523809523808,
>>   'numerical_token_proportion': 0.023809523809523808,
>>   'stop_word_proportion': 0.4523809523809524,
>>   'punctuation_character_count': 4,
>>   'punctuation_character_proportion': 0.022988505747126436,
>>   'contains_profanity': 0},
>>  {'token_count': 28,
>>   'string_length': 116,
>>   'average_token_size': 4.142857142857143,
>>   'stop_word_count': 14,
>>   'numerical_token_count': 0,
>>   'upper_token_count': 1,
>>   'title_token_count': 6,
>>   'flesch_reading_ease': 68.44,
>>   'smog_index': 0.0,
>>   'flesch_kincaid_grade': 10.7,
>>   'coleman_liau_index': 6.85,
>>   'automated_readability_index': 12.1,
>>   'dale_chall_readability_score': 6.72,
>>   'difficult_words': 3,
>>   'linsear_write_formula': 16.0,
>>   'gunning_fog': 14.06,
>>   'text_standard': 7.0,
>>   'polarity': 0.37,
>>   'subjectivity': 0.63,
>>   'exclamation_mark_count': 0,
>>   'question_mark_count': 0,
>>   'number_of_unique_tokens': 27,
>>   'unique_token_proportion': 0.9642857142857143,
>>   'title_token_proportion': 0.21428571428571427,
>>   'upper_token_proportion': 0.03571428571428571,
>>   'numerical_token_proportion': 0,
>>   'stop_word_proportion': 0.5,
>>   'punctuation_character_count': 1,
>>   'punctuation_character_proportion': 0.008620689655172414,
>>   'contains_profanity': 0}]


```

#### Normalizing strings

```python
normalizer = TextNormalizer(case=True, hyphenated_words=True, quotation_marks=True, 
                    spelling=False, unicode_characters=True, whitespace=True)

normalized_text = normalizer.transform(text)

print(normalized_text)
>> ["five weeks ago i spoke to you from these steps and said that this government wasn't going to hang around and that we would not wait until brexit day - october 31 - to deliver on the priorities of the british people.", 
>> 'and so i am proud to say that on wednesday chancellor sajid javid is going to set out the most ambitious spending round for more than a decade.', 
>> 'i said i wanted to make your streets safer – and that is why we are recruiting another 20,000 police officers.']


```

#### Replacing text

```python
replacer = TextReplacer(contractions=True, numbers=False)
text_with_replacements = replacer.transform(normalized_text)

print(text_with_replacements)
>> ['five weeks ago i spoke to you from these steps and said that this government was not going to hang around and that we would not wait until brexit day - october 31 - to deliver on the priorities of the british people.', 
>> 'and so i am proud to say that on wednesday chancellor sajid javid is going to set out the most ambitious spending round for more than a decade.', 
>> 'i said i wanted to make your streets safer – and that is why we are recruiting another 20,000 police officers.']

```

#### Removing text

```python
remover = TextRemover(punctuation=True, stop_words=True, numbers=True)
text_with_removals = remover.transform(text_with_replacements)

print(text_with_removals)
>> ['five weeks ago spoke steps said government going hang around would wait brexit day october deliver priorities british people',
>>  'proud say wednesday chancellor sajid javid going set ambitious spending round decade',
>>  'said wanted make streets safer recruiting another police officers']

```


#### Fingerprint clustering

Useful for tidying up inconsistent labelling:

```python
messy_names = ["Firstname Lastname", 
               "Firstname LASTNAME", 
               "LASTNAME, FIRSTNAME", 
               "Lastname, First-name", 
               "LastName, FirstName", 
               "firstname lastname",
               "FIRSTNAME LASTNAME:",
               "Lastname. Firstname",
               "Lastname; Firstname",
               "firstname.lastname"]
               

fingerprints = FingerPrintTransformer(n_gram=None, return_fingerprints=False)
n_gram_fingerprints1 = FingerPrintTransformer(n_gram=1, return_fingerprints=False)
n_gram_fingerprints2 = FingerPrintTransformer(n_gram=2, return_fingerprints=False)

print(fingerprints.transform(messy_names))
>> ['Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Lastname, First-name',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname']


print(n_gram_fingerprints1.transform(messy_names))
>> ['Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'Firstname Lastname']

print(n_gram_fingerprints2.transform(messy_names))
>> ['Firstname Lastname',
>>  'Firstname Lastname',
>>  'LASTNAME, FIRSTNAME',
>>  'Lastname, First-name',
>>  'LASTNAME, FIRSTNAME',
>>  'Firstname Lastname',
>>  'Firstname Lastname',
>>  'LASTNAME, FIRSTNAME',
>>  'LASTNAME, FIRSTNAME',
>>  'Firstname Lastname']

```


