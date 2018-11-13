#!/usr/bin/env bash

for num in 31 33 34 35 36 37 40 41 43 44 45 46; do
    log_dir="log/event/$num"
    save_dir="log/model/$num"
    answer_dir="log/answer/$num"
    python config.py --mode train --log_dir $log_dir --save_dir $save_dir --answer_dir $answer_dir
done

for num in 31 33 34 35 36 37 40 41 43 44 45 46; do
    log_dir="log/event/$num"
    save_dir="log/model/$num"
    answer_dir="log/answer/$num"
    python config.py --mode test --log_dir $log_dir --save_dir $save_dir --answer_dir $answer_dir
done
