#!/bin/bash
run () {
    python code/tri_char_rnn.py --token-size=$1 --num-unrollings=$2 --num-steps=$3 > $4
}
#run 3 5 1001 f_3_5.txt
#run 3 15 1001 f_3_15.txt
#run 3 10 1001 f_3_10.txt

run 2 5 10001 f_2_5v2.txt
run 2 15 1001 f_2_15v2.txt
run 2 10 10001 f_2_10v2.txt
