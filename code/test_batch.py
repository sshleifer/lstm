import unittest
from code.helpers import BatchGenerator, read_data, CHARACTER_SIZE

from code.helpers import ngram2id
from code.tri_char_rnn import main, id2_ngram


class GenTests(unittest.TestCase):

    text = read_data('text8.zip')
    train_text = text[:1000]

    def test_batch_generator_one(self):
        bg = BatchGenerator(self.train_text, 64, 10, token_size=1)
        bg.next()

    def test_two(self):
        bg = BatchGenerator(self.train_text, 64, 10, token_size=2)
        bg.next()

    def test_three(self):
        bg = BatchGenerator(self.train_text, 64, 10, token_size=3,
                            vocab_size=CHARACTER_SIZE**3)
        bg.next()

    def test_encode(self):
        seqs = [' ab', 'def', 'fed', 'ab', 'a', ' ']
        for seq in seqs:
            char_id = ngram2id(seq)
            self.assertEqual(id2_ngram(char_id, len(seq)), seq)
