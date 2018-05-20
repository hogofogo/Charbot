# Chatbot

## Overview

There are three parts to the project:
1. Build a chatbot engine for serving best response from stack overflow to technical questions
2. Build a chatbot engine (this is the seq2seq model) for serving free-flow response to non-technical questions
3. Integrate q&a engines into an off-the-shelf bot (Telegram)

Username at Telegram is @hogofogo_bot. 

The first part of the project itself focused on building a stack overflow chatbot to provide answers to free form questions, essentially building a semantic vector of the question, comparing against similar vectors for previously answered questions, picking the closest and serving an appropriate response. The responses are pre-built; the implementation is in jupyter notebook.

The second part of the project is a chitchat boat that is able to process human input and respond, and is the focus of the below. 

I have decided to build a chatbot using a seq2seq model, that would be able to generate human-like responses and train it on a sizable database of movie dialog subtitlles, which contains ~160K pairs. This is large enough to require a GPU processing. Once trained, the machine managed to carry on a reasonably coherent conversation - see flow example below. Some of the answers even come across as witty.

The repo also contains the scripts necessary to build data and spin up the bot. 

## Archtecture

For the time being, I built a simple encoder-decoder LSTM model, two flavors: one for training and one for inference. It has generally follows the outlines explained in this keras blog:
https://blog.keras.io/a-ten-minute-introduction-to-sequence-to-sequence-learning-in-keras.html

The model is word-based. I have used GloVe embeddings to encode tokens. Depending on performance, the model can be easily enhanced by adding additional layers. 


## Data cleaning

It's been a pretty straightforward process processing sentences getting rid of bad symbols, converting to lowercase, removing extra spaces, etc. I may try experimenting with a subset of data filtering questions and answers only; for the time being it includes everything I got.

## Training

I have trained the model for ~6 epochs and stopped once validation loss stopped improving; this took ~ 1 hour.



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

respond('to be or not to be')
Out[73]: 'you re a little'

respond('a little what')
Out[74]: 'a little of a girl'

respond('that s not very nice')
Out[75]: 'i m sorry'


Further experiments may include:
- deeper model;
- more focused dataset (e.g. select questions only)
- I need to build a web service engine - [priority for now]

