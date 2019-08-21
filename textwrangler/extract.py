from typing import Text
from nltk.corpus import stopwords
import numpy as np
import os
from urllib.request import urlopen
from zipfile import ZipFile
from io import BytesIO
from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors  # load the Stanford GloVe model

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

def _download_embeddings(embeddings_path="./embeddings"):
    if len(os.listdir(embeddings_path)) == 0:
        with urlopen('http://nlp.stanford.edu/data/glove.6B.zip') as url:
            with ZipFile(BytesIO(url.read())) as zfile:
                zfile.extractall(embeddings_path)

def _create_word2vec_files(embeddings_path="./embeddings"):
    embeddings_files = os.listdir(embeddings_path)
    w2v_files = os.listdir('./word2vec')
    if len(w2v_files) == 0 and len(embeddings_files) > 0:
        for emb_file in embeddings_files:
            glove2word2vec(f'{embeddings_path}/{emb_file}', f'./word2vec/{emb_file}.word2vec')

def extract_word_vectors(text: Text, vector_dim=100) -> np.ndarray:
    w2v_file = f'./word2vec/glove.6B.{vector_dim}d.txt.word2vec'
    if w2v_file not in os.listdir('./word2vec'):
        _download_embeddings()
        _create_word2vec_files()

    model = KeyedVectors.load_word2vec_format(w2v_file, binary=False)
    vectors = []
    for token in text.split():
        vectors.append(model[token])
    return sum(vectors)/len(vectors)



# extract language
# readability tests/scores
# extract people / count names
# emoji count
# easy data augmentation / generate augmentations


