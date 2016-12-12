import string
import os
from urllib import urlretrieve
import numpy as np
import random
import tensorflow as tf
import zipfile

CHARACTER_SIZE = (len(string.ascii_lowercase) + 1)  # [a-z] + ' '
FIRST_LETTER = ord(string.ascii_lowercase[0])
VOCABULARY_SIZE = CHARACTER_SIZE ** 2  # [a-z] + ' ' (bigram)

PROJECT_ROOT = '/Users/shleifer/lstmux'
DATASET_FILE = os.path.join(PROJECT_ROOT, 'text8.pkl')
URL = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        filename, _ = urlretrieve(URL + filename, filename)

    stat_info = os.stat(filename)

    if stat_info.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(stat_info.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')

    return filename


def read_data(filename):
    with zipfile.ZipFile(filename) as zip_file:
        for name in zip_file.namelist():
            return tf.compat.as_str(zip_file.read(name))


def char2id(char):
    if char in string.ascii_lowercase:
        return ord(char) - FIRST_LETTER + 1
    elif char == ' ':
        return 0
    else:
        print('Unexpected character: %s' % char)
        return 0


def id2char(char_id):
    if char_id > 0:
        return chr(char_id + FIRST_LETTER - 1)
    else:
        return ' '


def ngram2id(bigram):
    """easily extensible to ngram2id actually"""
    char_id = 0
    for digit, char in enumerate(bigram):
        char_id += char2id(char) * (CHARACTER_SIZE ** digit)
    return char_id


def id2_ngram(char_id, n_chars=3):
    seqs = []
    for char in reversed(range(n_chars)):
        part = char_id // (CHARACTER_SIZE ** char)
        char_id -= part * (CHARACTER_SIZE ** char)
        seqs.append(id2char(part))
    return ''.join(reversed(seqs))


def random_distribution(size):
    """Generate a random column of probabilities."""
    b = np.random.uniform(0.0, 1.0, size=[1, size])
    return b / np.sum(b, 1)[:, None]


def log_prob(predictions, labels):
    """Log-probability of the true labels in a predicted batch."""
    predictions[predictions < 1e-10] = 1e-10
    return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample_distribution(distribution):
    """Sample one element from a distribution assumed to be an array of normalized probabilities."""
    r = random.uniform(0, 1)
    s = 0

    for i in range(len(distribution)):
        s += distribution[i]

        if s >= r:
            return i

    return len(distribution) - 1

class BatchGenerator(object):
    def __init__(self, text, batch_size, num_unrollings, token_size=2, vocab_size=VOCABULARY_SIZE):
        self._text = text
        self.vocab_size = vocab_size
        self._text_size = len(text)
        self._batch_size = batch_size
        self._num_unrollings = num_unrollings
        segment = self._text_size // batch_size
        self._cursor = [offset * segment for offset in range(batch_size)]
        self.token_size = token_size
        self._last_batch = self._next_batch()

    def _next_batch(self):
        """Generate a single batch from the current cursor position in the data."""
        batch = np.zeros(shape=(self._batch_size, self.vocab_size), dtype=np.float)

        for batch_id in range(self._batch_size):
            cur_pos = self._cursor[batch_id]
            text_slice = self._text[cur_pos: cur_pos + self.token_size]
            gram_id = ngram2id(text_slice)
            batch[batch_id, gram_id] = 1.0
            self._cursor[batch_id] = (cur_pos + self.token_size) % (self._text_size - 1)

        return batch

    def next(self):
        """Generate the next array of batches from the data. The array consists of
        the last batch of the previous array, followed by num_unrollings new ones.
        """
        batches = [self._last_batch]

        for step in range(self._num_unrollings):
            batches.append(self._next_batch())

        self._last_batch = batches[-1]

        return batches
