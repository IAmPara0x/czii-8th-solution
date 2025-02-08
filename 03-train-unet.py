#!/usr/bin/env python

import numpy as np
import torch
import os

from typing import List
from monai.data.meta_tensor import MetaTensor

from src.utils import *
from src.augmentations import *

from monai.data.dataloader import DataLoader
from monai.data.dataset import Dataset, CacheDataset
from src.callbacks import get_callbacks
from src.model import UNetModel

import lightning.pytorch as pl

from omegaconf import OmegaConf
import glob


torch.serialization.add_safe_globals([MetaTensor])
torch.set_float32_matmul_precision('medium')


def load_data(cfg) -> List[dict]:
    data_list = []
    
    for i in (cfg.train.val_ids + cfg.train.train_ids):
        tomo_types = ["denoised", "ctfdeconvolved", "isonetcorrected"]
        for tomo_type in tomo_types:
            # Load the original image and label
            image_path = os.path.join(cfg.train.data_dir, f"train_image_{i}_{tomo_type}.npy")
            label_path = os.path.join(cfg.train.data_dir, f"train_label_{i}_{tomo_type}.npy")
            
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")
            
            image = np.load(image_path)
            label = np.load(label_path).astype(np.int64)
            
            print(f"Original Label {i}: dtype={label.dtype}, min={label.min()}, max={label.max()}, shape={label.shape}")
            assert label.dtype in [np.int64, np.int32], f"Label {i} is not integer type."
            assert label.min() >= 0 and label.max() < cfg.task.num_classes, f"Label {i} has values out of range."
            
            # Append with 'id'
            data_list.append({"id": i,"tomo_type": tomo_type, "image": image, "label": label})
            
                
    return data_list




if __name__ == "__main__":

    experiments = ["TS_6_4", "TS_6_6", "TS_69_2", "TS_73_6", "TS_86_3", "TS_99_9", "TS_5_4"]
    
    cfg = get_cfg("config.yml")

    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([f"train.train_ids={list(set(experiments) - set(cfg.train.val_ids))}"]))
    print(OmegaConf.to_yaml(cfg))
    
    seed_everything(cfg.task.seed)

    data_list = load_data(cfg)

    print(f"Total samples loaded: {len(data_list)}")

    # Initialize empty lists for train and validation
    train_files = []
    val_files = []

    # Iterate over data_list and distribute items accordingly
    for d in data_list:
        if d['id'] not in cfg.train.val_ids and (cfg.train.use_other_tomos or d["tomo_type"] == "denoised"):
            train_files.append(d)
        elif d['id'] in cfg.train.val_ids and d["tomo_type"] == "denoised":
            val_files.append(d)

    # Clear data_list to free memory
    del data_list

    print(f"Number of training samples: {len(train_files)}")
    print(f"Number of validation samples: {len(val_files)}")
    print("Training IDs:", [d['id'] for d in train_files])
    print("Validation IDs:", [d['id'] for d in val_files])
    
   
    # Create training dataset with non-random transforms cached
    raw_train_ds = CacheDataset(data=train_files, transform=non_random_transforms, cache_rate=0)
  
    # Apply random transforms for data augmentation
    random_transforms = get_random_transforms(cfg)
    train_ds = Dataset(data=raw_train_ds, transform=random_transforms)
    del raw_train_ds

    # Create DataLoader for training
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        persistent_workers=False,  # Changed to True for performance if needed
        drop_last=True

    )

    # Prepare validation patches
    val_images, val_labels = [dcts['image'] for dcts in val_files], [dcts['label'] for dcts in val_files]
    val_image_patches, _ = extract_3d_patches_minimal_overlap(val_images, [cfg.task.patch_size, cfg.task.patch_size, cfg.task.patch_size])
    val_label_patches, _ = extract_3d_patches_minimal_overlap(val_labels, [cfg.task.patch_size, cfg.task.patch_size, cfg.task.patch_size])
    val_patched_data = [{"image": img, "label": lbl} for img, lbl in zip(val_image_patches, val_label_patches)]

    # Create validation dataset with non-random transforms cached
    valid_ds = CacheDataset(data=val_patched_data, transform=non_random_transforms, cache_rate=0)

    # Create DataLoader for validation
    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.validation.batch_size,
        shuffle=False,
        num_workers=cfg.validation.num_workers,
        pin_memory=True,
        persistent_workers=False,  # Changed to True for performance if needed
        drop_last=True
    )

    logger, callbacks = get_callbacks(cfg)

    if cfg.train.use_pretrain:
        pretrain_models_ckpts = glob.glob(os.path.join(cfg.checkpoints.dir, f"pretrain*"))
        model = UNetModel.load_from_checkpoint(select_model(pretrain_models_ckpts, target='val_loss'),cfg=cfg)
        print(select_model(pretrain_models_ckpts, target='val_loss'))
    else:
        model = UNetModel(cfg)

    trainer = pl.Trainer(
        max_epochs=cfg.train.num_epochs,
        accelerator="gpu",
        devices=1,  # Use only one GPU
        precision='16-mixed',  # Updated precision setting
        log_every_n_steps=2,
        enable_progress_bar=True,
        logger=logger,
        callbacks=callbacks
    )

    trainer.fit(model, train_loader, valid_loader)
