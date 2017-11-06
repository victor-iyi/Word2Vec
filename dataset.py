"""
  @author Victor I. Afolabi
  A.I. Engineer & Software developer
  javafolabi@gmail.com
  
  Created on 02 November, 2017 @ 11:24 PM.
  
  Copyright © 2017. Victor. All rights reserved.
"""
import os
import sys
import pickle
import datetime as dt

import numpy as np


# Base `Dataset` class
class Dataset():
    def __init__(self, data_dir, **kwargs):
        self._data_dir = data_dir
        # Keyword arguments
        self._logging = kwargs['logging'] if 'logging' in kwargs else True
    
    def create(self):
        """Create datasets"""
        self._process()
        self._num_examples = self._X.shape[0]
        self._epochs_completed = 0
        self._index_in_epoch = 0
    
    def save(self, save_file, force=False):
        """Saves the dataset object."""
        if os.path.isfile(save_file) and not force:
            raise FileExistsError('{} already exist. Set `force=True` to override.'.format(save_file))
        dirs = save_file.split('/')
        if len(dirs) > 1 and not os.path.isdir('/'.join(dirs[:-1])):
            os.makedirs('/'.join(dirs[:-1]))
        f = open(save_file, mode='wb')
        pickle.dump(self, f)
        f.close()
        # with open(save_file, 'wb') as f:
        #     pickle.dump(obj=self, file=f)

    def load(self, save_file):
        """Load a saved Dataset object"""
        if not os.path.isfile(save_file):
            raise FileNotFoundError('{} was not found.'.format(save_file))
        with open(save_file, 'rb') as f:
            self = pickle.load(file=f)
        return self
    
    def next_batch(self, batch_size, shuffle=True):
        """Get the next batch in the dataset"""
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
    
    def train_test_split(self, test_size=0.1, **kwargs):
        """
        Splits dataset into training and testing set.

        :param dataset: Dataset to be split.
        :param test_size: float, default 0.1
                    Size of the testing data in %.
                    Default is 0.1 or 10% of the dataset.
        :keyword valid_portion: float
                    Size of validation set in %.
                    This will be taking from training set
                    after splitting into training and testing set.
        :return: np.array of train_X, train_y, test_X, test_y
        """
        test_size = int(len(self._X) * test_size)

        train_X = self._X[:-test_size]
        train_y = self._y[:-test_size]
        test_X = self._X[-test_size:]
        test_y = self._y[-test_size:]

        if 'valid_portion' in kwargs:
            valid_portion = kwargs['valid_portion']
            valid_portion = int(len(train_X) * valid_portion)

            train_X = train_X[:-valid_portion]
            train_y = train_y[:-valid_portion]
            val_X = train_X[-valid_portion:]
            val_y = train_y[-valid_portion:]
            return np.array([train_X, train_y, test_X, test_y, val_X, val_y])

        return np.array([train_X, train_y, test_X, test_y])
    
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
    def index_in_epoch(self):
        return self._index_in_epoch
    
    @property
    def num_classes(self):
        return len(self._labels)
    
    @property
    def epochs_completed(self):
        return self._epochs_completed
    
    def _process(self):
        pass
    
    def _create_label(self, label):
        hot = np.zeros(shape=[len(self._labels)], dtype=int)
        hot[self._labels.index(label)] = 1
        return hot
    
    def _one_hot(self, arr):
        arr, uniques = list(arr), list(set(arr))
        encoding = np.zeros(shape=[len(arr), len(uniques)], dtype=np.int32)
        for i, a in enumerate(arr):
            encoding[i, uniques.index(a)] = 1.
        return encoding


# !-------------------------------------- Image Dataset --------------------------------------! #
# `ImageDataset` class for image datasets
class ImageDataset(Dataset):
    
    def __init__(self, data_dir, grayscale=False, flatten=False, size=50, **kwargs):
        super().__init__(data_dir, **kwargs)
        self.grayscale = grayscale
        self.flatten = flatten
        self.size = size
        self._labels = [l for l in os.listdir(self._data_dir) if l[0] is not '.']

    @property
    def images(self):
        return self._X
    
    @property
    def channels(self):
        return 1 if len(self._X[0].shape) <= 2 else self._X[0].shape[-1]
    
    def _process(self):
        try:
            from PIL import Image
        except Exception as e:
            raise ModuleNotFoundError('{}'.format(e))
        datasets = []
        for i, label in enumerate(self._labels):
            image_dir = os.path.join(self._data_dir, label)
            image_list = [d for d in os.listdir(image_dir) if d[0] is not '.']
            for j, file in enumerate(image_list):
                try:
                    path = os.path.join(image_dir, file)
                    img = Image.open(path)
                    img = img.resize((self.size, self.size))
                    if self.grayscale:
                        img = img.convert('L')
                    img = np.array(img, dtype=np.float32)
                    if self.flatten:
                        img = img.flatten()
                    datasets.append([img, self._create_label(label)])
                except Exception as e:
                    sys.stderr.write('{}'.format(e))
                    sys.stderr.flush()
                if self._logging:
                    sys.stdout.write('\rProcessing {} of {} class labels & {} of {} images'.format(
                    i+1, len(self._labels), j+1, len(image_list)))
        # dataset into features & labels
        datasets = np.asarray(datasets)
        np.random.shuffle(datasets)
        self._X = np.array([img for img in datasets[:,0]])
        self._y = np.array([label for label in datasets[:,1]])
        del datasets  # free memory

# !-------------------------------------- Text Dataset --------------------------------------! #
# `TextDataset` for textual dataset
class TextDataset(Dataset):
    def __init__(self, data_dir, window=2, max_word=None, **kwargs):
        super().__init__(data_dir, **kwargs)
        self._window = window
        self._max_word = max_word

        # TODO: Look into `data_dir`. You may wanna get all files in there and read as a BIG corpus
        corpus_text = open(self._data_dir, mode='r', encoding='utf-8').read()
        if self._max_word:
            corpus_text = corpus_text[:self._max_word]
        corpus_text = corpus_text.lower()
        # import nltk.word_tokenize, nltk.sent_tokenize
        try:
            from nltk import word_tokenize, sent_tokenize
        except Exception as e:
            raise ModuleNotFoundError('{}'.format(e))
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

    def _process(self):
        # Creatnig features & labels
        self._X = np.zeros(shape=[len(self._sentences), self._vocab_size])
        self._y = np.zeros(shape=[len(self._sentences), self._vocab_size])
        
        start_time = dt.datetime.now()
        for s, sent in enumerate(self._sentences):
            for i, word in enumerate(sent):
                start = max(i - self._window, 0)
                end = min(self._window+i, len(sent)) + 1
                word_window = sent[start:end]
                for context in word_window:
                    if context is not word:
                        # data.append([word, context])
                        self._X[s] = self._one_hot(self._word2id[word])
                        self._y[s] = self._one_hot(self._word2id[context])
            if self._logging:
                sys.stdout.write('\rProcessing {:,} of {:,} sentences. Time taken: {}'.format(s+1, len(self._sentences),
                                                                                              dt.datetime.now() - start_time))
        # Free memory
        del start_time

    def _one_hot(self, idx):
        temp = np.zeros(shape=[self._vocab_size])
        temp[idx] = 1.
        return temp


if __name__ == '__main__':
    data_dir = 'datasets/flowers'
    save_file = 'datasets/saved/features-{0}x{0}.pkl'

    data = ImageDataset(data_dir=data_dir)
    data.create()  # creates features & label
    data.save(save_file.format(data.size))  # saves this object
    # data = data.load(save_file.format(data.size))  # loads saved object
    
    # Split into training, testing & validation set.
    X_train, y_train, X_test, y_test, X_val, y_val = data.train_test_split(test_size=0.2, valid_portion=0.1)
    # X_train, y_train, X_test, y_test = data.train_test_split(test_size=0.2)

    print('\nTrain: X{}\tTest: y{}\tValid: X{}'.format(X_train.shape, y_test.shape, X_val.shape))




