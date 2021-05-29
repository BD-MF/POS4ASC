#!/bin/bash

# laptop
# original
python run.py --model_name ${1} --train_data_name laptop --mode train
python run.py --model_name ${1} --train_data_name laptop --test_data_name rest --mode test
python run.py --model_name ${1} --train_data_name laptop --test_data_name arts_laptop --mode test
# weight
python run.py --model_name ${1} --train_data_name laptop --mode train --weight
python run.py --model_name ${1} --train_data_name laptop --test_data_name rest --mode test --weight
python run.py --model_name ${1} --train_data_name laptop --test_data_name arts_laptop --mode test --weight
# dropout
python run.py --model_name ${1} --train_data_name laptop --mode train --dropout
python run.py --model_name ${1} --train_data_name laptop --test_data_name rest --mode test --dropout
python run.py --model_name ${1} --train_data_name laptop --test_data_name arts_laptop --mode test --dropout

# rest
# original
python run.py --model_name ${1} --train_data_name rest --mode train
python run.py --model_name ${1} --train_data_name rest --test_data_name laptop --mode test
python run.py --model_name ${1} --train_data_name rest --test_data_name arts_rest --mode test
# weight
python run.py --model_name ${1} --train_data_name rest --mode train --weight
python run.py --model_name ${1} --train_data_name rest --test_data_name laptop --mode test --weight
python run.py --model_name ${1} --train_data_name rest --test_data_name arts_laptop --mode test --weight
# dropout
python run.py --model_name ${1} --train_data_name rest --mode train --dropout
python run.py --model_name ${1} --train_data_name rest --test_data_name laptop --mode test --dropout
python run.py --model_name ${1} --train_data_name rest --test_data_name arts_rest --mode test --dropout