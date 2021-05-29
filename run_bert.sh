#!/bin/bash

# laptop
# original
python run.py --model_name ${1} --train_data_name laptop --mode train --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name laptop --test_data_name rest --mode test --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name laptop --test_data_name arts_laptop --mode test --learning_rate 1e-5 --weight_decay 0.0
# weight
python run.py --model_name ${1} --train_data_name laptop --mode train --weight --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name laptop --test_data_name rest --mode test --weight --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name laptop --test_data_name arts_laptop --mode test --weight --learning_rate 1e-5 --weight_decay 0.0
# dropout
python run.py --model_name ${1} --train_data_name laptop --mode train --dropout --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name laptop --test_data_name rest --mode test --dropout --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name laptop --test_data_name arts_laptop --mode test --dropout --learning_rate 1e-5 --weight_decay 0.0

# rest
# original
python run.py --model_name ${1} --train_data_name rest --mode train --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name rest --test_data_name laptop --mode test --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name rest --test_data_name arts_rest --mode test --learning_rate 1e-5 --weight_decay 0.0
# weight
python run.py --model_name ${1} --train_data_name rest --mode train --weight --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name rest --test_data_name laptop --mode test --weight --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name rest --test_data_name arts_laptop --mode test --weight --learning_rate 1e-5 --weight_decay 0.0
# dropout
python run.py --model_name ${1} --train_data_name rest --mode train --dropout --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name rest --test_data_name laptop --mode test --dropout --learning_rate 1e-5 --weight_decay 0.0
python run.py --model_name ${1} --train_data_name rest --test_data_name arts_rest --mode test --dropout --learning_rate 1e-5 --weight_decay 0.0