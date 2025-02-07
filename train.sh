#!/usr/bin/bash

set -xe

ALL_FOLDS=('TS_5_4' 'TS_6_4' 'TS_69_2' 'TS_6_6' 'TS_73_6' 'TS_86_3' 'TS_99_9')


# Tiny unet pretrain
python ./02-pretrain-unet.py model.channels="[32, 64, 128, 128]" model.strides="[2, 2, 1]" model.num_res_units=1 model.dropout=0.3 train.num_epochs=128 validation.batch_size=24 task.num_samples=24 optimizer.patience=30 version="pretrain" checkpoints.dir="./checkpoints/tiny-unet" logs.dir="./logs/tiny-unet" train.use_ema=null

# Tiny unet train all folds
for FOLD in "${ALL_FOLDS[@]}"; do
    echo "Training FOLD: $FOLD"
    python ./03-train-unet.py model.channels="[32, 64, 128, 128]" model.strides="[2, 2, 1]" model.num_res_units=1 model.dropout=0.3 train.num_epochs=200 validation.batch_size=24 task.num_samples=24 optimizer.patience=32 version="fold_${FOLD}" checkpoints.dir="./checkpoints/tiny-unet" logs.dir="./logs/tiny-unet" train.use_ema=null train.val_ids="[${FOLD}]" train.use_other_tomos=null train.pretrain_ckpt="./checkpoints/tiny-unet/pretrain-epoch=112-val_dice_mean=0.2793-val_loss=0.8007.ckpt"
done

# python ./02-pretrain-unet.py model.channels="[32, 64, 128, 256]" model.strides="[2, 2, 1]" model.num_res_units=2 model.dropout=0.3 train.num_epochs=128 validation.batch_size=24 task.num_samples=24 optimizer.patience=30 version="medium_unet_pretrain"

# python ./02-pretrain-unet.py model.channels="[32, 96, 256, 384]" model.strides="[2, 2, 1]" model.num_res_units=2 model.dropout=0.3 train.num_epochs=128 validation.batch_size=16 task.num_samples=16 optimizer.patience=30 version="large_unet_pretrain"
