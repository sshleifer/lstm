alias gt="python code/gen_test.py"
alias tr="python code/ptb_word_lm.py --data_path=data --model test --save_path=new_log --logdir=new_log"

trun (){
    python code/ptb_word_lm.py --data_path=data --model $1 --save_path=$2 --logdir=new_log
}

export PYTHONPATH='.'
