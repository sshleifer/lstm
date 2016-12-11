from code.ptb_word_lm import generate_text

TRAIN_PATH = 'data/'
#MODEL_PATH = 'log_test/'

MODEL_PATH = '/Users/shleifer/lstmux/fun.meta'

if __name__ == '__main__':
    text = generate_text(TRAIN_PATH, MODEL_PATH)
    print text
