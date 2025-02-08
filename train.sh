#!/usr/bin/bash

set -xe

ALL_FOLDS=('TS_5_4' 'TS_6_4' 'TS_69_2' 'TS_6_6' 'TS_73_6' 'TS_86_3' 'TS_99_9')

# Tiny unet pretrain
python ./02-pretrain-unet.py model.channels="[32, 64, 128, 128]" model.strides="[2, 2, 1]" model.num_res_units=1 model.dropout=0.3 train.num_epochs=128 validation.batch_size=24 task.num_samples=24 optimizer.patience=30 version="pretrain" checkpoints.dir="./checkpoints/tiny-unet" logs.dir="./logs/tiny-unet" train.use_ema=null

# Tiny unet train all folds
for FOLD in "${ALL_FOLDS[@]}"; do
    echo "Training FOLD: $FOLD"
    python ./03-train-unet.py model.channels="[32, 64, 128, 128]" model.strides="[2, 2, 1]" model.num_res_units=1 model.dropout=0.3 train.num_epochs=200 validation.batch_size=24 task.num_samples=24 optimizer.patience=32 version="fold_${FOLD}" checkpoints.dir="./checkpoints/tiny-unet" logs.dir="./logs/tiny-unet" train.use_ema=null train.val_ids="[${FOLD}]" train.use_other_tomos=null train.use_pretrain=true
done

# Medium unet pretrain
python ./02-pretrain-unet.py model.channels="[32, 64, 128, 256]" model.strides="[2, 2, 1]" model.num_res_units=2 model.dropout=0.3 train.num_epochs=128 validation.batch_size=24 task.num_samples=16 optimizer.patience=30 version="pretrain" checkpoints.dir="./checkpoints/medium-unet" logs.dir="./logs/medium-unet" train.use_ema=null

# Medium unet train all folds
for FOLD in "${ALL_FOLDS[@]}"; do
    echo "Training FOLD: $FOLD"
    python ./03-train-unet.py model.channels="[32, 64, 128, 256]" model.strides="[2, 2, 1]" model.num_res_units=2 model.dropout=0.3 train.num_epochs=200 validation.batch_size=24 task.num_samples=16 optimizer.patience=16 version="fold_${FOLD}" checkpoints.dir="./checkpoints/medium-unet" logs.dir="./logs/medium-unet" train.use_ema=null train.val_ids="[${FOLD}]" train.use_other_tomos=true train.use_pretrain=true
done

BIG_MODEL_FOLDS=('TS_69_2' 'TS_86_3' 'TS_99_9')
# 384 unets pretrain
python ./02-pretrain-unet.py \
  model.channels="[32, 96, 256, 384]" \
  model.strides="[2, 2, 1]" \
  model.num_res_units=2 \
  model.dropout=0.2 \
  train.num_epochs=2000 \
  validation.batch_size=24 \
  task.num_samples=16 \
  optimizer.patience=30 \
  version="pretrain" \
  pretrain.synthetic_ids="['TS_0', 'TS_1', 'TS_2', 'TS_3', 'TS_4', 'TS_5', 'TS_6', 'TS_7', 'TS_8', 'TS_9', 'TS_10', 'TS_11', 'TS_12', 'TS_13', 'TS_14', 'TS_15', 'TS_16', 'TS_17', 'TS_18', 'TS_19', 'TS_20', 'TS_21', 'TS_22', 'TS_23', 'TS_24', 'TS_25', 'TS_26']" \
  pretrain.val_ids="['TS_0']" \
  pretrain.val_data_dir="./data/synthetic-data" \
  checkpoints.dir="./checkpoints/big-unet" \
  logs.dir="./logs/big-unet" \
  train.use_ema=true \
  train.ema.momentum=0.01 \
  train.ema.warmup=100

# 384 unets train on 3 folds, denoised
for FOLD in "${BIG_MODEL_FOLDS[@]}"; do
    echo "Training big denoised unet FOLD: $FOLD"
    python ./03-train-unet.py model.channels="[32, 96, 256, 384]" model.strides="[2, 2, 1]" model.num_res_units=2 model.dropout=0.2 train.num_epochs=2000 validation.batch_size=24 task.num_samples=16 optimizer.patience=100 version="fold_${FOLD}" checkpoints.dir="./checkpoints/big-denoised-unet" logs.dir="./logs/big-denoised-unet" train.use_ema=true train.val_ids="[${FOLD}]" train.use_other_tomos=null train.use_pretrain=true
done
# 384 unets train on 3 folds, all tomos
for FOLD in "${BIG_MODEL_FOLDS[@]}"; do
    echo "Training big all-tomo unet FOLD: $FOLD"
    python ./03-train-unet.py model.channels="[32, 96, 256, 384]" model.strides="[2, 2, 1]" model.num_res_units=2 model.dropout=0.2 train.num_epochs=2000 validation.batch_size=24 task.num_samples=16 optimizer.patience=100 version="fold_${FOLD}" checkpoints.dir="./checkpoints/big-alltomo-unet" logs.dir="./logs/big-alltomo-unet" train.use_ema=true train.val_ids="[${FOLD}]" train.use_other_tomos=true train.use_pretrain=true
done
