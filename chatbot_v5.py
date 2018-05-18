#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 16 18:46:50 2018

@author: vlad
"""

from keras.models import Model
from keras.layers.recurrent import LSTM
from keras.layers import Dense, Input, Embedding
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint
from collections import Counter
import nltk
import numpy as np
from sklearn.model_selection import train_test_split
import urllib.request
import os
import sys
import zipfile
from keras.preprocessing.text import Tokenizer
from random import shuffle


BATCH_SIZE = 64
NUM_EPOCHS = 100
GLOVE_EMBEDDING_SIZE = 100
HIDDEN_UNITS = 256
MAX_INPUT_SEQ_LENGTH = 40
MAX_TARGET_SEQ_LENGTH = 40
MAX_VOCAB_SIZE = 400000
WEIGHT_FILE_PATH = '/home/vlad/Documents/Chatbot/word-glove-weights.h5'


df = readOpensubsData('/home/vlad/Documents/Chatbot/data/', max_len=20, fast_preprocessing=True)

def load_glove():
    embeddings_index = {}
    GLOVE_DIR = '/home/vlad/Documents/Toxic/glove'
    f = open(os.path.join(GLOVE_DIR, 'glove.6B.100d.txt'))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    f.close()
    return embeddings_index

word2em = load_glove()


#shuffle data pairs and tokenize texts
max_words = 30000
tokenizer = Tokenizer(num_words=max_words, filters='"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                                   lower=True)
shuffle(df)
input_texts, output_texts = zip(*df)
tokenizer.fit_on_texts(list(input_texts) + list(output_texts))

#append 'start' and 'end' tokens to the start and end of sentences
def append_start_end(input_texts):
    output_texts = []
    for line in output_texts:
        line = 'start ' + line + ' end'
        output_texts.append(line)
    return output_texts

output_texts = append_start_end(output_texts)

#create indexes of word to index and reverse
word2idx = tokenizer.word_index
idx2word = dict([(idx, word) for word, idx in word2idx.items()])

num_decoder_tokens = len(idx2word)+1


input_texts_word2em = []

encoder_max_seq_length = 0
decoder_max_seq_length = 0

for input_words, target_words in zip(input_texts, output_texts):
    encoder_input_wids = []
    for w in input_words.split():
        emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
        if w in word2em:
            emb = word2em[w]
        encoder_input_wids.append(emb)

    input_texts_word2em.append(encoder_input_wids)
    encoder_max_seq_length = max(len(encoder_input_wids), encoder_max_seq_length)
    decoder_max_seq_length = max(len(target_words), decoder_max_seq_length)

context = dict()
context['num_decoder_tokens'] = num_decoder_tokens
context['encoder_max_seq_length'] = encoder_max_seq_length
context['decoder_max_seq_length'] = decoder_max_seq_length

print(context)
np.save('models/' + DATA_SET_NAME + '/word-glove-context.npy', context)


#build batch generator
def generate_batch(input_word2em_data, output_text_data):
    
    num_batches = len(input_word2em_data) // BATCH_SIZE
    
    while True:
        
        for i in range(0, num_batches):
            
            start = i * BATCH_SIZE
            end = (i + 1) * BATCH_SIZE
            encoder_input_data_batch = pad_sequences(input_word2em_data[start:end],
                            encoder_max_seq_length)
            decoder_output_data_batch = np.zeros(shape=(BATCH_SIZE, 
                            decoder_max_seq_length, num_decoder_tokens))
            decoder_input_data_batch = np.zeros(shape=(BATCH_SIZE, 
                            decoder_max_seq_length, GLOVE_EMBEDDING_SIZE))
            
            for lineIdx, target_words in enumerate(output_text_data[start:end]):
            
                for idx, w in enumerate(target_words.split()):
                
                    w2idx = word2idx['unknown']  # default unknown
                    
                    if w in word2idx:
                        w2idx = word2idx[w]
                    
                    if w in word2em:
                        decoder_input_data_batch[lineIdx, idx, :] = word2em[w]
                    
                    if idx > 0:
                        decoder_output_data_batch[lineIdx, idx - 1, w2idx] = 1

            yield [encoder_input_data_batch, decoder_input_data_batch], decoder_output_data_batch



#define model layers
encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name='encoder_lstm')
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, return_sequences=True, name='decoder_lstm')
decoder_outputs, decoder_state_h, decoder_state_c = decoder_lstm(decoder_inputs,
                                                initial_state=encoder_states)
decoder_dense = Dense(units=num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#split into train and validation sets
Xtrain, Xtest, Ytrain, Ytest = train_test_split(input_texts_word2em, output_texts, test_size=0.1)

train_gen = generate_batch(Xtrain, Ytrain)
test_gen = generate_batch(Xtest, Ytest)

train_num_batches = len(Xtrain) // BATCH_SIZE
test_num_batches = len(Xtest) // BATCH_SIZE

checkpoint = ModelCheckpoint(filepath=WEIGHT_FILE_PATH, save_best_only=True)
model.fit_generator(generator=train_gen, steps_per_epoch=train_num_batches,
                    epochs=NUM_EPOCHS, verbose=1, validation_data=test_gen, 
                    validation_steps=test_num_batches, callbacks=[checkpoint])

model.save_weights(WEIGHT_FILE_PATH)