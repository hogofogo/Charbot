# Chatbot

## Overview

This is work in progress: 
- I have a working chatbot; I might add improvements
- I need to build web service.

This started as an extension of a NLP course project I did. The project itself focused on building a stock overflow chatbot to provide answers to free form questions, essentially building a semantic vector of the question, comparing against similar vectors for previously answered questions, picking the closest and serving an appropriate response. However the stack overflow bot has been trained on stack overflow data, and can not handle a normal chitchat. 

I have decided to build a chatbot using a seq2seq model, that would be able to generate human-like responses and train it on a sizable database of movie dialog subtitlles, which contains ~160K pairs. This is large enough to require a GPU processing, although of course very small compared to industrial-scale products. Nevertheless, once trained, the machine managed to carry on a reasonably coherent conversation - see example below.


## Archtecture

For the time being, I built a simple encoder-decoder LSTM model, two flavors: one for training and one for inference. It has generally followed the outlines explained in this blog:
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

The model is word-based. I have used GloVe embeddings to encode tokens. Depending on performance, the model can be easily enhanced by adding additional layers. I have not yet tried adding regularization but may do it depending on the model performance.


## Data cleaning

For the time being, it's a pretty straightforward process processing sentences getting rid of bad symbols, converting to lowercase, removing extra spaces, etc. I may try experimenting with a subset of data filtering questions and answers only; for the time being it's everything.

## Training

I have trained the model for ~6 epochs and stopped once validation loss stopped improving; took ~ 1 hour.



## Results

These is the exchange I had with the machine:
[respond('hi')
Out[60]: 'hi'

respond('how are you?')
Out[61]: 'i m fine'

respond('what s your name?')
Out[62]: 'nothing'

respond('you mean your name is nothing?')
Out[63]: 'i m not a problem'

respond('where are you from?')
Out[64]: 'i m not a problem'

respond('i am beginning to think you actually are')
Out[65]: 'you re right'

respond('are you being sarcastic?')
Out[66]: 'yeah'

respond('where were you born?')
Out[67]: 'you re a good'

respond('i know i am good')
Out[68]: 'i know'

respond('can you please use longer sentence?')
Out[69]: 'what'

respond('use more words please')
Out[70]: 'i m not leaving'

respond('you are dumb, don t you think?')
Out[71]: 'yeah'

respond('you are an asshole')
Out[72]: 'you are']


Further experiments may include:
- deeper model;
- more focused dataset (e.g. select questions only)
- I need to build a web service engine

