import dataclasses

import click
import numpy as np
import torch
import lightning.pytorch as pl
import os

import yaml
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint
from pathlib import Path
import torch.nn as nn
from copy import deepcopy
from monai.data import DataLoader, Dataset, CacheDataset, decollate_batch
from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    Orientationd,
    AsDiscrete,
    RandFlipd,
    NormalizeIntensityd,
    RandCropByLabelClassesd,
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss, FocalLoss
from monai.metrics import DiceMetric
from monai.data import MetaTensor
# Define MetaTensor safety for serialization
torch.serialization.add_safe_globals([MetaTensor])
import random


seed = 42
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True

USE_SYNTHETIC = False

# Mapping of class IDs to names
ID_TO_NAME = {
    1: "apo-ferritin",
    2: "beta-amylase",
    3: "beta-galactosidase",
    4: "ribosome",
    5: "thyroglobulin",
    6: "virus-like-particle"
}


def load_data(
        data_dir: str,
        ids: list[str],
        tomo_types: list[str],
        num_classes: int = 7,
) -> list[dict]:
    """
    Load training data from numpy files.

    Parameters:
    - data_dir (str): Directory where the data files are stored.
    - is_augmented (bool): Whether to load augmented data.
    - tomo_types (list[str]): types of tomograms to load, e.g. ["denoised", "ctfdeconvolved", "isonetcorrected"]

    Returns:
    - List[dict]: List of dictionaries with "id", "image", and "label" keys.
    """
    data_list = []

    for i in ids:
        print(i)
        # tomo_types = ["denoised"]
        for tomo_type in tomo_types:
            # Load the original image and label
            image_path = os.path.join(data_dir, f"train_image_{i}_{tomo_type}.npy")
            label_path = os.path.join(data_dir, f"train_label_{i}_{tomo_type}.npy")

            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")
            if not os.path.exists(label_path):
                raise FileNotFoundError(f"Label file not found: {label_path}")

            image = np.load(image_path)
            label = np.load(label_path).astype(np.int64)

            print(f"Original Label {i}: dtype={label.dtype}, min={label.min()}, max={label.max()}, shape={label.shape}")
            assert label.dtype in [np.int64, np.int32], f"Label {i} is not integer type."
            assert label.min() >= 0 and label.max() < num_classes, f"Label {i} has values out of range."

            # Append with "id"
            data_list.append({"id": i, "tomo_type": tomo_type, "image": image, "label": label})

    return data_list


def calculate_patch_starts(dimension_size: int, patch_size: int) -> list[int]:
    """
    Calculate the starting positions of patches along a single dimension
    with minimal overlap to cover the entire dimension.
    """
    if dimension_size <= patch_size:
        return [0]

    # Calculate number of patches needed
    n_patches = np.ceil(dimension_size / patch_size)

    if n_patches == 1:
        return [0]

    # Calculate overlap
    total_overlap = (n_patches * patch_size - dimension_size) / (n_patches - 1)

    # Generate starting positions
    positions = []
    for i in range(int(n_patches)):
        pos = int(i * (patch_size - total_overlap))
        if pos + patch_size > dimension_size:
            pos = dimension_size - patch_size
        if pos not in positions:  # Avoid duplicates
            positions.append(pos)

    return positions


def extract_3d_patches_minimal_overlap(
        arrays: list[np.ndarray],
        patch_size: tuple[int, int, int]
) -> tuple[list[np.ndarray], list[tuple[int, int, int]]]:
    """
    Extract 3D patches from multiple arrays with minimal overlap to cover the entire array.
    Now supports different patch sizes per dimension (e.g., (184, 96, 96)).
    """
    if not arrays or not isinstance(arrays, list):
        raise ValueError("Input must be a non-empty list of arrays")

    # Verify all arrays have the same shape
    shape = arrays[0].shape
    print(shape)
    if not all(arr.shape == shape for arr in arrays):
        raise ValueError("All input arrays must have the same shape")

    # patch_size is now a tuple
    px, py, pz = patch_size
    if px > shape[0] or py > shape[1] or pz > shape[2]:
        raise ValueError(
            f"Patch size {patch_size} cannot exceed volume shape {shape}"
        )

    m, n, l = shape
    patches = []
    coordinates = []

    # Calculate starting positions for each dimension
    x_starts = calculate_patch_starts(m, px)
    y_starts = calculate_patch_starts(n, py)
    z_starts = calculate_patch_starts(l, pz)

    # Extract patches from each array
    for arr in arrays:
        for x in x_starts:
            for y in y_starts:
                for z in z_starts:
                    patch = arr[x:x + px, y:y + py, z:z + pz]
                    patches.append(patch)
                    coordinates.append((x, y, z))

    return patches, coordinates


class EMA(nn.Module):
    def __init__(self, model: nn.Module, momentum=0.00001, warmup: int = None):
        # https://www.kaggle.com/competitions/hubmap-hacking-the-human-vasculature/discussion/429060
        # https://github.com/Lightning-AI/pytorch-lightning/issues/10914
        super(EMA, self).__init__()
        self.module = deepcopy(model)
        self.module.eval()
        self.momentum = momentum
        self.decay = 1 - self.momentum
        self.warmup = 0 if warmup is None else warmup
        self.i_updates = 0

    def _update(self, model, update_fn):
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        if self.i_updates < self.warmup:
            # print(f"setting ema <- model: {self.i_updates}/{self.warmup}")
            self.set(model)
        else:
            self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)
        self.i_updates += 1

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class UNetModel(pl.LightningModule):
    def __init__(
            self,
            patience: int = 100,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 7,
            channels: tuple[int, ...] | list[int] = (32, 64, 128, 128),
            strides: tuple[int, ...] | list[int] = (2, 2, 1),
            num_res_units: int = 1,  # Increased residual units
            lr: float = 1e-3,
            kernel_size: tuple[int, ...] | list[int] | int = 3,
            dropout=0.3,
    ):
        super().__init__()
        self.save_hyperparameters(
            ignore=["patience"]
        )
        self.patience = patience
        print("===========================")
        print(f"spatial_dims = {self.hparams.spatial_dims}")
        print(f"in_channels = {self.hparams.in_channels}")
        print(f"out_channels = {self.hparams.out_channels}")
        print(f"channels = {self.hparams.channels}")
        print(f"kernel_size = {self.hparams.kernel_size}")
        print(f"strides = {self.hparams.strides}")
        print(f"num_res_units = {self.hparams.num_res_units}")
        print(f"dropout = {self.hparams.dropout}")
        print("===========================")
        self.model = UNet(
            spatial_dims=self.hparams.spatial_dims,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            channels=self.hparams.channels,
            kernel_size=self.hparams.kernel_size,
            strides=self.hparams.strides,
            num_res_units=self.hparams.num_res_units,
            dropout=self.hparams.dropout
        )
        self.ema_model = EMA(self.model, 0.01, 100)
        self.class_weights = None
        self.ce_loss = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True, weight=self.class_weights)
        self.focal_loss = FocalLoss(include_background=False, to_onehot_y=True, use_softmax=True, weight=self.class_weights)
        self.metric_fn = DiceMetric(include_background=False, reduction="none", ignore_empty=True)

    def backward(self, loss: torch.Tensor, *args: any, **kwargs: any) -> None:
        if self.ema_model is not None:
            self.ema_model.update(self.model)
        return pl.LightningModule.backward(self, loss, *args, **kwargs)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images = batch["image"]
        labels = batch["label"]
        x, y = images, labels
        y_hat = self(x)
        loss = self.ce_loss(y_hat, y)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self.ema_model.module(x)
        # y_hat = self.model(x)
        val_loss = self.ce_loss(y_hat, y)
        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        y_hat_list = decollate_batch(y_hat)
        y_list = decollate_batch(y)

        metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=self.hparams.out_channels)(i) for i in y_hat_list]
        metric_val_labels = [AsDiscrete(to_onehot=self.hparams.out_channels)(i) for i in y_list]

        self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)

    def on_validation_epoch_end(self):
        metric = self.metric_fn.aggregate()
        metric_mean = torch.nanmean(metric, dim=0)  # shape: [num_classes]
        print("metric_mean", metric_mean)
        val_metric = 0.0
        for class_idx in range(metric_mean.shape[0]):
            value = metric_mean[class_idx]
            if self.class_weights is not None:
                weight = self.class_weights[class_idx]
            else:
                weight = 1.0 / 7.0
            val_metric += weight * value
        val_metric = metric_mean.mean()  # mean across classes

        self.log("val_dice_mean", val_metric, on_epoch=True, sync_dist=True)
        print(metric_mean)
        for class_idx in range(metric_mean.shape[0]):
            class_id = class_idx + 1
            class_name = ID_TO_NAME.get(class_id, f"class_{class_id}")
            value = metric_mean[class_idx]
            if torch.isnan(value):
                value = torch.nan_to_num(value, nan=0.0)

            self.log(f"val_dice_{class_name}", value, on_epoch=True, sync_dist=True)
        self.metric_fn.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=self.patience // 2, factor=0.5, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1
            }
        }


@dataclasses.dataclass
class Config:
    train_data_dir: str  # directory containing the training data
    tomo_types: list[str]  # tomogram types to work on

    patch_size: int  # patch size for the training process
    num_classes: int  # number of classes for the model

    learning_rate: float  # learning rate
    num_epochs: int  # max number of training epochs
    batch_size_train: int  # batch size for training
    batch_size_valid: int  # batch size for validation
    num_workers: int  # number of dataloader workers
    patience: int  # patience for early stopping
    load_from_checkpoint: str | None  # if given, training will resume from this checkpoint

    ids: list[str]  # ids for all tomograms (train + validation)
    checkpoint_dir_name: str


def load_config(config_name: str) -> Config:
    """
    loads the configs in question, e.g. config_name=finetune_denoised would load finetune_denoised.yaml
    """
    config_path = Path(__file__).parent / f"{config_name}.yaml"
    with open(config_path, "r") as f:
        config_yaml = yaml.load(f, yaml.FullLoader)
    config = Config(
        train_data_dir=config_yaml["train_data_dir"],
        tomo_types=config_yaml["tomo_types"],
        patch_size=config_yaml["patch_size"],
        num_classes=config_yaml["num_classes"],
        learning_rate=config_yaml["learning_rate"],
        num_epochs=config_yaml["num_epochs"],
        batch_size_train=config_yaml["batch_size_train"],
        batch_size_valid=config_yaml["batch_size_valid"],
        num_workers=config_yaml["num_workers"],
        patience=config_yaml["patience"],
        ids=config_yaml["ids"],
        load_from_checkpoint=config_yaml["load_from_checkpoint"],
        checkpoint_dir_name=config_yaml["checkpoint_dir_name"],
    )
    return config


@click.command
@click.option("--config-name", default="finetune_denoised", required=True)
@click.option("--val", default="TS_5_4", required=True)
def main(config_name: str, val: str):
    print(f"running config: {config_name}, val on {val}")
    config = load_config(config_name)
    print(f"{config=}")

    # Load training data
    data_list = load_data(config.train_data_dir, config.ids, config.tomo_types)

    print(f"Total samples loaded: {len(data_list)}")

    # Initialize empty lists for train and validation
    train_files = []
    val_files = []

    # Iterate over data_list and distribute items accordingly
    for d in data_list:
        if d["id"] == val:
            val_files.append(d)
        else:
            train_files.append(d)

    # Clear data_list to free memory
    del data_list

    print(f"Number of training samples: {len(train_files)}")
    print(f"Number of validation samples: {len(val_files)}")
    print("Training IDs:", [d["id"] for d in train_files])
    print("Validation IDs:", [d["id"] for d in val_files])

    # Non-random transforms to be cached
    non_random_transforms = Compose([
        EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
        NormalizeIntensityd(keys="image"),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
    ])

    # Random transforms to be applied during training
    random_transforms = Compose([
        RandCropByLabelClassesd(
            keys=["image", "label"],
            label_key="label",
            spatial_size=[config.patch_size, config.patch_size, config.patch_size],
            num_classes=config.num_classes,
            num_samples=16,
        ),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
        RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
    ])

    # Create training dataset with non-random transforms cached
    raw_train_ds = CacheDataset(data=train_files, transform=non_random_transforms, cache_rate=1)

    # Apply random transforms for data augmentation
    train_ds = Dataset(data=raw_train_ds, transform=random_transforms)
    del raw_train_ds
    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size_train,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=False,
        drop_last=True,
    )

    # Prepare validation patches
    val_images, val_labels = [dcts["image"] for dcts in val_files], [dcts["label"] for dcts in val_files]
    val_image_patches, _ = extract_3d_patches_minimal_overlap(val_images, (config.patch_size, config.patch_size, config.patch_size))
    val_label_patches, _ = extract_3d_patches_minimal_overlap(val_labels, (config.patch_size, config.patch_size, config.patch_size))
    val_patched_data = [{"image": img, "label": lbl} for img, lbl in zip(val_image_patches, val_label_patches)]

    # Create validation dataset with non-random transforms cached
    valid_ds = CacheDataset(data=val_patched_data, transform=non_random_transforms, cache_rate=0)

    # Create DataLoader for validation
    valid_loader = DataLoader(
        valid_ds,
        batch_size=config.batch_size_valid,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=False,  # Changed to True for performance if needed
        drop_last=True

    )

    # Initialize ModelCheckpoint to monitor "val_dice_mean" and save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Metric to monitor
        dirpath=config.checkpoint_dir_name,  # Directory to save checkpoints
        filename=val + "-best-checkpoint-{epoch:02d}-{val_dice_mean:.4f}-{val_loss:.4f}",  # Checkpoint filename format
        save_top_k=1,  # Save only the best model
        mode="min",  # "min" because lower "val_loss" is better
        verbose=True,  # Verbosity
    )

    logger = WandbLogger("sergio_unets_" + val)

    # Initialize callbacks
    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=config.patience,
        mode="min",
        verbose=True,
        check_finite=True
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    if config.load_from_checkpoint is not None:
        print(f"{config.load_from_checkpoint=}, resuming")
        model: UNetModel = UNetModel.load_from_checkpoint(
            config.load_from_checkpoint,
            strict=False,
        )
        model.model = deepcopy(model.ema_model.module)  # so that the model's weight is initialised based off of the pretrained model's weight
    else:
        print(f"{config.load_from_checkpoint=}, training from scratch")
        model = UNetModel(
            # this is the model_config_five in the submission
            spatial_dims=3,
            in_channels=1,
            out_channels=config.num_classes,
            kernel_size=3,
            channels=(32, 96, 256, 384),
            strides=(2, 2, 1),
            num_res_units=2,
            lr=config.learning_rate,
            dropout=0.2,
        )

    trainer = pl.Trainer(
        max_epochs=config.num_epochs,
        accelerator="gpu",
        devices=1,  # Use only one GPU
        precision="16-mixed",  # Updated precision setting
        log_every_n_steps=2,
        enable_progress_bar=True,
        logger=logger,
        callbacks=[early_stopping, lr_monitor, checkpoint_callback],  # Added EarlyStopping and LR Monitor callbacks
    )
    trainer.fit(model, train_loader, valid_loader)


if __name__ == "__main__":
    main()
