from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping
from lightning.pytorch.callbacks import LearningRateMonitor, EarlyStopping, ModelCheckpoint


def get_callbacks(cfg):


    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',    # Metric to monitor
        dirpath=cfg.checkpoints.dir,      # Directory to save checkpoints
        filename=f'{cfg.version}'+'-{epoch:02d}-{val_dice_mean:.4f}-{val_loss:.4f}',  # Checkpoint filename format
        save_top_k=2,               # Save only the best model
        mode='min',                 # 'max' because higher 'val_dice_mean' is better
        verbose=True                # Verbosity
    )


    logger =  TensorBoardLogger(
      save_dir=cfg.logging.dir,
      name=f'{cfg.version}',
      version=0
    )

    # Initialize callbacks
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=cfg.optimizer.patience,
        mode='min',
        verbose=True,
        check_finite=True
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    
    return logger, [early_stopping, lr_monitor, checkpoint_callback]
