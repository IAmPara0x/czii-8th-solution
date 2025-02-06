#!/usr/bin/bash


# Tiny unet pretrain

python ./02-pretrain-unet.py model.channels="[32, 64, 128, 128]" model.strides="[2, 2, 1]" model.num_res_units=1 model.dropout=0.3 train.num_epochs=128 validation.batch_size=24 task.num_samples=24 optimizer.patience=30
