#!/bin/bash

python ptb_word_lm.py --data_path=/Users/shleifer/lstm-char-cnn/data/ptb/ --model small > log_small.txt
python ptb_word_lm.py --data_path=/Users/shleifer/lstm-char-cnn/data/ptb/ --model medium > log_medium.txt
python ptb_word_lm.py --data_path=/Users/shleifer/lstm-char-cnn/data/ptb/ --model large > log_large.txt
