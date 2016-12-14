#!/bin/bash


python code/tri_char_rnn.py --token_size=3 --num_unrollings=10 --num_steps=10000 > f_3_10.txt
python code/tri_char_rnn.py --token_size=3 --num_unrollings=5 --num_steps=10000 > f_3_5.txt 
python code/tri_char_rnn.py --token_size=3 --num_unrollings=15 --num_steps=10000 > f_3_15.txt

python code/tri_char_rnn.py --token_size=2 --num_unrollings=10 --num_steps=10000 > f_2_10.txt
python code/tri_char_rnn.py --token_size=2 --num_unrollings=5 --num_steps=10000 > f_2_5.txt 
python code/tri_char_rnn.py --token_size=2 --num_unrollings=15 --num_steps=10000 > f_2_15.txt

