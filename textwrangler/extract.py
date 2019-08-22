from typing import Text, Dict, List
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

def extract_token_count(text: Text) -> int:
    return len(text.split())

def extract_string_length(text: Text, exclude_whitespace=True) -> int:
    if exclude_whitespace == True:
        len(''.join(text.split()))
    else:
        return len(text)

def extract_average_token_size(text: Text) -> float:
  tokens = text.split()
  return sum(len(token) for token in tokens)/len(tokens)

def extract_stop_word_count(text: Text) -> int:
    return len([token for token in text.split() if token in stopwords.words('english')])

def extract_numerical_character_count(text: Text) -> int:
    return len([token for token in text.split() if token.isdigit()])

def extract_upper_token_count(text: Text) -> int:
    return len([token for token in text.split() if token.isupper()])

def _download_embeddings(embeddings_path="../embeddings"):
    if len(os.listdir(embeddings_path)) == 0:
        with urlopen('http://nlp.stanford.edu/data/glove.6B.zip') as url:
            with ZipFile(BytesIO(url.read())) as zfile:
                zfile.extractall(embeddings_path)

def _create_word2vec_files(embeddings_path="../embeddings"):
    embeddings_files = os.listdir(embeddings_path)
    w2v_files = os.listdir('../word2vec')
    if len(w2v_files) == 0 and len(embeddings_files) > 0:
        for emb_file in embeddings_files:
            glove2word2vec(f'{embeddings_path}/{emb_file}', f'./word2vec/{emb_file}.word2vec')

def extract_word_vectors(text: Text, vector_dim=100) -> np.ndarray:
    w2v_file = f'word2vec/glove.6B.{vector_dim}d.txt.word2vec'
    if w2v_file not in os.listdir('../word2vec'):
        _download_embeddings()
        _create_word2vec_files()

    model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    vectors = []
    for token in text.split():
        vectors.append(model[token])
    return sum(vectors)/len(vectors)

def extract_readability_scores(text: Text, scores=None) -> Dict:

    output = {}
    if 'flesch_reading_ease' in scores or scores == None:
        output['flesch_reading_ease'] = textstat.flesch_reading_ease(text)
    if 'smog_index' in scores or scores == None:
        output['smog_index'] = textstat.smog_index(text)
    if 'flesch_kincaid_grade' in scores or scores == None:
        output['flesch_kincaid_grade'] = textstat.flesch_kincaid_grade(text)
    if 'coleman_liau_index' in scores or scores == None:
        output['coleman_liau_index'] = textstat.coleman_liau_index(text)
    if 'automated_readability_index' in scores or scores == None:
        output['automated_readability_index'] = textstat.automated_readability_index(text)
    if 'dale_chall_readability_score' in scores or scores == None:
        output['dale_chall_readability_score'] = textstat.dale_chall_readability_score(text)
    if 'difficult_words' in scores or scores == None:
        output['difficult_words'] = textstat.difficult_words(text)
    if 'linsear_write_formula' in scores or scores == None:
        output['linsear_write_formula'] = textstat.linsear_write_formula(text)
    if 'gunning_fog' in scores or scores == None:
        output['gunning_fog'] = textstat.gunning_fog(text)
    if 'text_standard' in scores or scores == None:
        output['text_standard'] = textstat.text_standard(text, float_output=True)

    return output

def extract_language(text: Text, output_probabilities=False):
    if output_probabilities == False:
        return detect(text)
    else:
        return {f'lang_{item.lang}': item.prob for item in detect_langs(text)}




# extract people / count names
# emoji count
# easy data augmentation / generate augmentations


