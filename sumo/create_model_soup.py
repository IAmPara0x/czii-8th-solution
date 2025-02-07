import torch
from tqdm import tqdm
from monai.networks.nets import UNet
from dataclasses import dataclass
from pytorch_lightning import LightningModule
from typing import *
import torch.nn as nn
from copy import deepcopy
from pathlib import Path
import yaml


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


class Model(LightningModule):
    def __init__(
            self,
            spatial_dims: int = 3,
            in_channels: int = 1,
            out_channels: int = 7,
            channels: Union[Tuple[int, ...], List[int]] = (32, 64, 128, 128),
            strides: Union[Tuple[int, ...], List[int]] = (2, 2, 1),
            num_res_units: int = 1,
            use_ema: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.model = UNet(
            spatial_dims=self.hparams.spatial_dims,
            in_channels=self.hparams.in_channels,
            out_channels=self.hparams.out_channels,
            channels=self.hparams.channels,
            strides=self.hparams.strides,
            num_res_units=self.hparams.num_res_units,
        )
        # self.class_weights = torch.tensor([1, 1, 0.1, 2, 1, 8, 1], dtype=torch.float32)
        self.use_ema = use_ema
        self.ema_model = EMA(self.model, 0.001, 1000)

    def configure_ema_model(self):
        # sets either self.model or self.ema_model as the one doing forward, delete the irrelevant one to save gpu space
        if self.use_ema:
            print(f"model does uses ema, swapping ema_model into model")
            self.model = deepcopy(self.ema_model.module)
            del self.ema_model
        else:
            print(f"model does NOT uses ema, discarding ema_model")
            del self.ema_model

    def forward(self, x):
        return self.model(x)


@dataclass
class ModelSpec:
    model_path: str
    model_config: dict
    use_ema: bool


def create_model_soup(
        model_dir: str,
        out_path: str,
):
    """
    read all model weights inside model_dir, create model soup, write the final weights out to out_path
    """
    model_config = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 7,
        "channels": (32, 96, 256, 384),
        "strides": (2, 2, 1),
        "num_res_units": 2,
    }

    model_dir = Path(model_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    model_paths = model_dir.glob("TS_*ckpt")
    model_specs = []
    for p in model_paths:
        model_specs.append(ModelSpec(
            model_path=p,
            model_config=model_config,
            use_ema=True,
        ))
    print(f"souping {len(model_specs)} models")
    for m in model_specs:
        print(m)

    weights = 1.0 / len(model_specs)
    state_dict_soup = {}

    for i, model_spec in tqdm(enumerate(model_specs), total=len(model_specs)):
        model = Model(**model_spec.model_config, use_ema=model_spec.use_ema)
        checkpoint = torch.load(model_spec.model_path)
        model.load_state_dict(checkpoint["state_dict"], strict=False)  # strict=False for backward compat with those without ema_model

        for p in model.state_dict():
            if i == 0:
                state_dict_soup[p] = model.state_dict()[p] * weights
            else:
                state_dict_soup[p] += model.state_dict()[p] * weights

    model = Model(**model_specs[0].model_config, use_ema=model_specs[0].use_ema)
    model.load_state_dict(state_dict_soup)
    torch.save(model.state_dict(), out_path)
    print(f"model soup written to: {out_path}")
    print(f"================")


def main():
    this_file_dir = Path(__file__).parent
    config_paths = [
        this_file_dir / "finetune_denoised.yaml",
        this_file_dir / "finetune_all.yaml",
    ]
    for p in config_paths:
        with open(p, "r") as f:
            config_yaml = yaml.load(f, yaml.FullLoader)
        checkpoint_dir_name = config_yaml["checkpoint_dir_name"]
        checkpoint_dir = this_file_dir / checkpoint_dir_name
        model_out_path = checkpoint_dir / f"{checkpoint_dir.name}_soup.pth"
        create_model_soup(checkpoint_dir, model_out_path)


if __name__ == "__main__":
    main()
