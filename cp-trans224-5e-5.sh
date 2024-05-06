#!/bin/bash

python -m cellpose --file_list file_list.npy --dir absolute_data_path --verbose --train --min_train_masks 0 --all_channels --pretrained_model None --nimg_per_epoch 800 --SGD 0 --learning_rate 0.00005 --n_epochs 2000 --model_name_out cp-trans224-5en5 --use_gpu --train_size --no_norm --transformer


