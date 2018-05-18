#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 17 09:41:27 2018

@author: vlad
"""

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import nltk
import os
import sys
import zipfile
import urllib.request

HIDDEN_UNITS = 256
WHITELIST = 'abcdefghijklmnopqrstuvwxyz1234567890?.,'
GLOVE_EMBEDDING_SIZE = 100


def in_white_list(_word):
    for char in _word:
        if char in WHITELIST:
            return True

    return False


def load_glove():
    download_glove()
    word2em = {}
    file = open(GLOVE_MODEL, mode='rt', encoding='utf8')
    for line in file:
        words = line.strip().split()
        word = words[0]
        embeds = np.array(words[1:], dtype=np.float32)
        word2em[word] = embeds
    file.close()
    return word2em


       
max_encoder_seq_length = context['encoder_max_seq_length']
max_decoder_seq_length = context['decoder_max_seq_length']
num_decoder_tokens = context['num_decoder_tokens']

#the below redefines the same layers as in the training model, loads trained
#weights, and creates two additional models specilizing in encoding and decoding
#job functions using weights from the trained model
encoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='encoder_inputs')
encoder_lstm = LSTM(units=HIDDEN_UNITS, return_state=True, name="encoder_lstm")
encoder_outputs, encoder_state_h, encoder_state_c = encoder_lstm(encoder_inputs)
encoder_states = [encoder_state_h, encoder_state_c]

decoder_inputs = Input(shape=(None, GLOVE_EMBEDDING_SIZE), name='decoder_inputs')
decoder_lstm = LSTM(units=HIDDEN_UNITS, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.load_weights(WEIGHT_FILE_PATH)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')

encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_inputs = [Input(shape=(HIDDEN_UNITS,)), Input(shape=(HIDDEN_UNITS,))]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)

def respond(input_text):
    input_seq = []
    input_emb = []
    
    for word in nltk.word_tokenize(input_text.lower()):
        if not in_white_list(word):
            continue
        emb = np.zeros(shape=GLOVE_EMBEDDING_SIZE)
        if word in word2em:
            emb = word2em[word]
        input_emb.append(emb)
    
    input_seq.append(input_emb)
    input_seq = pad_sequences(input_seq, max_encoder_seq_length)
    states_value = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
    target_seq[0, 0, :] = word2em['start']
    output_text = ''
    output_text_len = 0
    terminated = False
    
    while not terminated:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sample_token_idx = np.argmax(output_tokens[0, -1, :])
        sample_word = idx2word[sample_token_idx]
        output_text_len += 1

        if sample_word != 'start' and sample_word != 'end':
            output_text += ' ' + sample_word

        if sample_word == 'end' or output_text_len >= max_decoder_seq_length:
            terminated = True

        target_seq = np.zeros((1, 1, GLOVE_EMBEDDING_SIZE))
        if sample_word in word2em:
            target_seq[0, 0, :] = word2em[sample_word]

        states_value = [h, c]
    
    return output_text.strip()

