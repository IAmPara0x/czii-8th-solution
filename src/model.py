import lightning.pytorch as pl
import torch

from monai.data.utils import decollate_batch
from monai.transforms.post.array import AsDiscrete

from monai.networks.nets.unet import UNet
from monai.losses.dice import  DiceCELoss
from monai.metrics.meandice import DiceMetric

import gc



class UNetModel(pl.LightningModule):
    def __init__( self, cfg):
        super().__init__()
       
        self.cfg = cfg
        self.model = UNet(norm="INSTANCE", **cfg.model)
        self.out_channels = cfg.model.out_channels
      
        
        self.ce_loss = DiceCELoss(include_background=True, to_onehot_y=True, softmax=True,)        
     
        self.metric_fn = DiceMetric(include_background=False, reduction="none", ignore_empty=True)

    def forward(self, x):
        return self.model(x)

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
        y_hat = self(x)
        val_loss = self.ce_loss(y_hat, y) 
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        
        y_hat_list = decollate_batch(y_hat)
        y_list = decollate_batch(y)
        
        metric_val_outputs = [AsDiscrete(argmax=True, to_onehot=self.out_channels)(i) for i in y_hat_list]
        metric_val_labels = [AsDiscrete(to_onehot=self.out_channels)(i) for i in y_list]
        
        self.metric_fn(y_pred=metric_val_outputs, y=metric_val_labels)
        
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

