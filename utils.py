import nltk
import pickle
import re
import numpy as np
import pandas as pd

nltk.download('stopwords')
from nltk.corpus import stopwords

# Paths for all resources for the bot.
RESOURCE_PATH = {
    'INTENT_RECOGNIZER': 'intent_recognizer.pkl',
    'TAG_CLASSIFIER': 'tag_classifier.pkl',
    'TFIDF_VECTORIZER': 'tfidf_vectorizer.pkl',
    'THREAD_EMBEDDINGS_FOLDER': 'thread_embeddings_by_tags',
    'WORD_EMBEDDINGS': 'word_embeddings.tsv',
    'STARSPACE_EMBEDDINGS': 'starspace_embeddings.tsv', 
    'EMBEDDINGS_DIM': 'embeddings_dim.tsv'
}


def text_prepare(text):
    """Performs tokenization and simple preprocessing."""
    
    replace_by_space_re = re.compile('[/(){}\[\]\|@,;]')
    bad_symbols_re = re.compile('[^0-9a-z #+_]')
    stopwords_set = set(stopwords.words('english'))

    text = text.lower()
    text = replace_by_space_re.sub(' ', text)
    text = bad_symbols_re.sub('', text)
    text = ' '.join([x for x in text.split() if x and x not in stopwords_set])

    return text.strip()


    
def question_to_vec(question, embeddings, dim):
    """Transforms a string to an embedding by averaging word embeddings."""
    
    if len(question.split()) == 0:
        array_len = 1
    else:
        array_len = len(question.split())
        
    emb_array = np.zeros((array_len, dim))
    emb_na_cnt = 0
    
    for i, word in enumerate(question.split()):
        if word in embeddings:
            emb_array[i,:] = embeddings[word]
        else:
            emb_array[i,:] = np.nan
            emb_na_cnt += 1
            
    if emb_na_cnt == array_len:
        result = np.zeros((1, dim))
    else:
        result = np.nanmean(emb_array, axis = 0)
    
    return result.squeeze()
 
    
    
def load_embeddings(embeddings_path):
    """Loads pre-trained word embeddings from tsv file.

    Args:
      embeddings_path - path to the embeddings file.

    Returns:
      embeddings - dict mapping words to vectors;
      embeddings_dim - dimension of the vectors.
    """
    
    words = []
    embeds = []

    for line in open(embeddings_path):
        w, *em = line.strip().split('\t')
        words.append(w)
        embeds.append(em)

    embeddings_dim = len(embeds[0])
    embeddings = zip(words, embeds)
    embeddings = dict(embeddings)

    return (embeddings, embeddings_dim)
    


def unpickle_file(filename):
    """Returns the result of unpickling the file content."""
    with open(filename, 'rb') as f:
        return pickle.load(f)
