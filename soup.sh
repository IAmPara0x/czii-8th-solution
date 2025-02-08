#!/usr/bin/bash

python ./04-create-soup.py model.channels="[32, 64, 128, 128]" model.strides="[2, 2, 1]" model.num_res_units=1 model.dropout=0.3 train.num_epochs=200 validation.batch_size=24 task.num_samples=24 optimizer.patience=32 version="tiny_unet_soup" checkpoints.dir="./checkpoints/tiny-unet"  train.use_ema=null soup.select_strategy="dice_score"
