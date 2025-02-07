import lightning.pytorch as pl

from monai.data.utils import decollate_batch
from monai.transforms.post.array import AsDiscrete

from monai.networks.nets.unet import UNet
from monai.losses.dice import  DiceCELoss
from monai.metrics.meandice import DiceMetric

import torch
import torch.nn as nn

from copy import deepcopy

import gc

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
            self.set(model)
        else:
            self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)
        self.i_updates += 1

    def set(self, model):
        self._update(model, update_fn=lambda e, m: m)


class UNetModel(pl.LightningModule):
    def __init__( self, cfg):
        super().__init__()
       
        self.cfg = cfg
        self.model = UNet(norm="INSTANCE", **cfg.model)
        self.out_channels = cfg.model.out_channels
      
        
        self.ce_loss = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True,)        
     
        self.metric_fn = DiceMetric(include_background=False, reduction="none", ignore_empty=True)

        if cfg.train.use_ema:
            self.ema_model = EMA(self.model, cfg.train.ema.momentum, cfg.train.ema.warmup)
        else:
            self.ema_model = None


    def forward(self, x):
        return self.model(x)

    def backward(self, loss: torch.Tensor, *args: any, **kwargs: any) -> None:
        if self.ema_model is not None:
            self.ema_model.update(self.model)
        return pl.LightningModule.backward(self, loss, *args, **kwargs)


    def training_step(self, batch, _):
        images = batch['image']    
        labels = batch['label']    
        x, y = images, labels
        y_hat = self(x)
        loss = self.ce_loss(y_hat, y) 
        self.log('train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, _):
        x, y = batch['image'], batch['label']
        if self.ema_model is None:
            y_hat = self(x)
        else:
            y_hat = self.ema_model.module(x)
        val_loss = self.ce_loss(y_hat, y)
       
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)

        y_hat_list = decollate_batch(y_hat)
        y_list = decollate_batch(y)

        metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=self.out_channels)(i) for i in y_hat_list]
        metric_val_labels = [AsDiscrete(to_onehot=self.out_channels)(i) for i in y_list]

        self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
        gc.collect()
        torch.cuda.empty_cache()

        
    def on_validation_epoch_end(self):
        metric = self.metric_fn.aggregate()
        metric_mean = torch.nanmean(metric, dim=0)     # shape: [num_classes]
        val_metric = metric_mean.mean()       # mean across classes

        
        self.log('val_dice_mean', val_metric, on_epoch=True, sync_dist=True)
        print(metric_mean)
        for class_idx in range(metric_mean.shape[0]):
            class_id = class_idx + 1
            class_name = self.cfg.task.id_to_name.get(class_id, f"class_{class_id}")
            value = metric_mean[class_idx]
            if torch.isnan(value):
                value = torch.nan_to_num(value, nan=0.0)
            
            self.log(f'val_dice_{class_name}', value, on_epoch=True, sync_dist=True)
        self.metric_fn.reset()
        gc.collect()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.optimizer.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=self.cfg.optimizer.patience//2, factor=0.5, verbose=True
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

