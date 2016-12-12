import unittest
from code.helpers import BatchGenerator, read_data, CHARACTER_SIZE

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
