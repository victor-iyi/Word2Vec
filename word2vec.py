"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 28 October, 2017 @ 9:55 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""

import datetime as dt
import multiprocessing

import numpy as np
import gensim.models.word2vec as w2v

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
        # Creatnig features & labels
        self._X = np.zeros(shape=[len(self._sentences), self._vocab_size])
        self._y = np.zeros(shape=[len(self._sentences), self._vocab_size])
        
        start_time = dt.datetime.now()
        for s, sent in enumerate(self._sentences):
            for i, word in enumerate(sent):
                start = max(i - self.window, 0)
                end = min(self.window+i, len(sent)) + 1
                word_window = sent[start:end]
                for context in word_window:
                    if context is not word:
                        # data.append([word, context])
                        self._X[s] = self.one_hot(self._word2id[word])
                        self._y[s] = self.one_hot(self._word2id[context])
            if self.logging:
                print(('\rProcessing {:,} of {:,} sentences. '
                      'Time taken: {}').format(s+1, len(self._sentences),
                                               dt.datetime.now() - start_time),
                      end='')
        self._num_examples = self._X.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
        # Free memory
        del start_time

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
    def word2id(self):
        return self._word2id
    
    @property
    def id2word(self):
        return self._id2word
    
    @property
    def sentences(self):
        return self._sentences
    
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


# !---------------------------------------------- genisim.models.word2vec ----------------------------------------------! #


class GensimWord2Vec:
    """    
     |  __init__(self, filename, sentences=None, size=100, alpha=0.025, window=5, min_count=5, 
                 max_vocab_size=None, sample=0.001, seed=1, workers=3, min_alpha=0.0001, 
                 sg=0, hs=0, negative=5, cbow_mean=1, hashfxn=<built-in function hash>, 
                 iter=5, null_word=0, trim_rule=None, sorted_vocab=1, batch_words=10000, 
                 compute_loss=False)
     |      Initialize the model from an iterable of `sentences`. Each sentence is a
     |      list of words (unicode strings) that will be used for training.
     |
     |      `filename` path to the file to be loaded
     |
     |      The `sentences` iterable can be simply a list, but for larger corpora,
     |      consider an iterable that streams the sentences directly from disk/network.
     |      See :class:`BrownCorpus`, :class:`Text8Corpus` or :class:`LineSentence` in
     |      this module for such examples.
     |      
     |      If you don't supply `sentences`, the model is left uninitialized -- use if
     |      you plan to initialize it in some other way.
     |      
     |      `sg` defines the training algorithm. By default (`sg=0`), CBOW is used.
     |      Otherwise (`sg=1`), skip-gram is employed.
     |      
     |      `size` is the dimensionality of the feature vectors.
     |      
     |      `window` is the maximum distance between the current and predicted word within a sentence.
     |      
     |      `alpha` is the initial learning rate (will linearly drop to `min_alpha` as training progresses).
     |      
     |      `seed` = for the random number generator. Initial vectors for each
     |      word are seeded with a hash of the concatenation of word + str(seed).
     |      Note that for a fully deterministically-reproducible run, you must also limit the model to
     |      a single worker thread, to eliminate ordering jitter from OS thread scheduling. (In Python
     |      3, reproducibility between interpreter launches also requires use of the PYTHONHASHSEED
     |      environment variable to control hash randomization.)
     |      
     |      `min_count` = ignore all words with total frequency lower than this.
     |      
     |      `max_vocab_size` = limit RAM during vocabulary building; if there are more unique
     |      words than this, then prune the infrequent ones. Every 10 million word types
     |      need about 1GB of RAM. Set to `None` for no limit (default).
     |      
     |      `sample` = threshold for configuring which higher-frequency words are randomly downsampled;
     |          default is 1e-3, useful range is (0, 1e-5).
     |      
     |      `workers` = use this many worker threads to train the model (=faster training with multicore machines).
     |      
     |      `hs` = if 1, hierarchical softmax will be used for model training.
     |      If set to 0 (default), and `negative` is non-zero, negative sampling will be used.
     |      
     |      `negative` = if > 0, negative sampling will be used, the int for negative
     |      specifies how many "noise words" should be drawn (usually between 5-20).
     |      Default is 5. If set to 0, no negative samping is used.
     |      
     |      `cbow_mean` = if 0, use the sum of the context word vectors. If 1 (default), use the mean.
     |      Only applies when cbow is used.
     |      
     |      `hashfxn` = hash function to use to randomly initialize weights, for increased
     |      training reproducibility. Default is Python's rudimentary built in hash function.
     |      
     |      `iter` = number of iterations (epochs) over the corpus. Default is 5.
     |      
     |      `trim_rule` = vocabulary trimming rule, specifies whether certain words should remain
     |      in the vocabulary, be trimmed away, or handled using the default (discard if word count < min_count).
     |      Can be None (min_count will be used), or a callable that accepts parameters (word, count, min_count) and
     |      returns either `utils.RULE_DISCARD`, `utils.RULE_KEEP` or `utils.RULE_DEFAULT`.
     |      Note: The rule, if given, is only used to prune vocabulary during build_vocab() and is not stored as part
     |      of the model.
     |      
     |      `sorted_vocab` = if 1 (default), sort the vocabulary by descending frequency before
     |      assigning word indexes.
     |      
     |      `batch_words` = target size (in words) for batches of examples passed to worker threads (and
     |      thus cython routines). Default is 10000. (Larger batches will be passed if individual
     |      texts are longer than 10000 words, but the standard cython code truncates to that maximum.)
    """
    def __init__(self, filename, **kwargs):
        corpus = open(filename, mode='r', encoding='utf-8').read()
        raw_sentences = sent_tokenize(corpus)
        self._sentences = [word_tokenize(sent) for sent in raw_sentences]
        workers = multiprocessing.cpu_count()
        sg = 1 # 0 - CBOW while 1 - skip gram
        self._model = w2v.Word2Vec(sentences=self._sentences,
                                   workers=workers, **kwargs)
        # Free memory
        del corpus, raw_sentences

    # !- Properties
    @property
    def model(self):
        return self._model
    
    @property
    def sentences(self):
        return self._sentences

    @property
    def word_count(self):
        return self._model.corpus_count

