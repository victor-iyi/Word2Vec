"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 28 October, 2017 @ 9:55 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""

import sys

import datetime as dt
import numpy as np

from nltk import word_tokenize, sent_tokenize



class Word2Vec:
    def __init__(self, filename, window=2, max_word=None, logging=True):
        self.window = window
        self.logging = logging
        # Read corpus
        corpus_text = open(filename, mode='r', encoding='utf-8').read()
        if max_word:
            corpus_text = corpus_text[:max_word]
        corpus_text = corpus_text.lower()
        # word2id & id2word
        unique_words = set(word_tokenize(corpus_text))
        self._vocab_size = len(unique_words)
        self._word2id = {w: i for i, w in enumerate(unique_words)}
        self._id2word = {i: w for i, w in enumerate(unique_words)}
        # Sentences
        raw_sentences = sent_tokenize(corpus_text)
        self._sentences = [word_tokenize(sent) for sent in raw_sentences]
        # Free some memory
        del corpus_text
        del unique_words
        del raw_sentences
        # Building data
        data = []
        for s, sent in enumerate(self._sentences):
            for i, word in enumerate(sent):
                start = max(i - self.window, 0)
                end = min(self.window+i, len(sent)) + 1
                word_window = sent[start:end]
                for context in word_window:
                    if context is not word:
                        data.append([word, context])
            if self.logging:
                sys.stdout.write('\r{:,} of {:,} sentences.'.format(s+1, len(self._sentences)))
        # Xs and ys
        _X = []
        _y = []
        start_time = dt.datetime.now()
        for i, word_data in enumerate(data):
            _X.append(self.one_hot(self._word2id[ word_data[0] ]))
            _y.append(self.one_hot(self._word2id[ word_data[1] ]))
            sys.stdout.write('\rProcessing {:,} of {:,}\tSo far = {}'.format(i+1, 
                                                                             len(data), 
                                                                             dt.datetime.now() - start_time))
        # Convert Xs and ys to numpy array
        self._X = np.asarray(_X)
        self._y = np.asarray(_y)
        self._num_examples = self._X.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        
        # Free memory
        del start_time
        del _X
        del _y

    def one_hot(self, idx):
        temp = np.zeros(shape=[self._vocab_size])
        temp[idx] = 1.
        return temp
    
    @property
    def features(self):
        return self._X
    
    @property
    def labels(self):
        return self._y
    
    @property
    def num_examples(self):
        return self._num_examples
    
    @property
    def vocab_size(self):
        return self._vocab_size
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def next_batch(self, batch_size, shuffle=True):
        start = self._index_in_epoch
        # Shuffle for first epoch
        if self._epochs_completed == 0 and start == 0 and shuffle:
            permute = np.arange(self._num_examples)
            np.random.shuffle(permute)
            self._X = self._X[permute]
            self._y = self._y[permute]
        # Go to next batch
        if start + batch_size > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Get the rest examples in this epoch
            rest_examples = self._num_examples - start
            rest_features = self._X[start:self._num_examples]
            rest_labels = self._y[start:self._num_examples]
            # Shuffle the data
            if shuffle:
                permute = np.arange(self._num_examples)
                np.random.shuffle(permute)
                self._X = self._X[permute]
                self._y = self._y[permute]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size - rest_examples
            end = self._index_in_epoch
            features = np.concatenate((rest_features, self._X[start:end]), axis=0)
            labels = np.concatenate((rest_labels, self._y[start:end]), axis=0)
            return features, labels
        else:
            self._index_in_epoch += batch_size
            end = self._index_in_epoch
            return self._X[start:end], self._y[start:end]
            