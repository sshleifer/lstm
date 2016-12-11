from code.ptb_word_lm import generate_text

TRAIN_PATH = 'data/'
MODEL_PATH = '/Users/shleifer/lstmux/new_log/'

if __name__ == '__main__':
    text = generate_text(TRAIN_PATH, MODEL_PATH)
    print text
