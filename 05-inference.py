#!/usr/bin/env python

from dataclasses import dataclass
import torch
from lightning.pytorch import LightningModule

from monai.networks.nets.unet import UNet
from copy import deepcopy

import cupy as cp
from cucim.skimage.feature import peak_local_max
import skimage.measure as measure
from cucim.core.operations.morphology import distance_transform_edt
from skimage.segmentation import watershed

from tqdm import tqdm
from monai.inferers.inferer import SlidingWindowInferer
import copick
import time

from src.utils import *
from src.model import EMA
from src.augmentations import inference_transforms

import torch.amp as amp

class Model(LightningModule):
    def __init__(
            self,
            spatial_dims= 3,
            in_channels= 1,
            out_channels= 7,
            channels= (32, 64, 128, 128),
            strides= (2, 2, 1),
            num_res_units= 1,
            use_ema= False,
    ):
        super().__init__()
     
        self.model = UNet(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=out_channels,
            channels=channels,
            strides=strides,
            num_res_units=num_res_units,
        )
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
    """
    informations required to load the model
    """
    model_path: str  # checkpoint path
    model_config: dict  # kwargs for the model initialisations
    use_ema: bool  # whether to load the model's weight or use the model's ema weight

def load_models(model_specs, device_id: int):
    """
    Load any number of models according to their specifications:
    """
    models = []
    for model_spec in model_specs:
        # Create your model with the config
        model = Model(**model_spec.model_config, use_ema=model_spec.use_ema)

        # Load state dict
        checkpoint = torch.load(model_spec.model_path)
        if "state_dict" in checkpoint:
            print("found state_dict key, loading state_dict")
            model.load_state_dict(checkpoint["state_dict"], strict=False)  # strict=False for backward compat with those without ema_model
        else:
            print("can't find state_dict key, loading checkpoint directly")
            model.load_state_dict(checkpoint, strict=False)
        model.configure_ema_model()

        model.to(f"cuda:{device_id}")
        model.eval()

        models.append(model)
    return models

@torch.no_grad()
def tta_infer(inputs, model, inferer):
    count = 0
    predmask = inferer(inputs.unsqueeze(0), model)
    count += 1
    predmask += torch.flip(inferer(torch.flip(inputs, dims=[3]).unsqueeze(0), model), dims=[4])  # Flip prediction back
    count += 1
    predmask += torch.flip(inferer(torch.flip(inputs, dims=[2]).unsqueeze(0), model), dims=[3])  # Flip prediction back
    count += 1
    predmask += inferer(inputs.transpose(2, 3).unsqueeze(0), model).transpose(3, 4)
    count += 1
    return predmask / count


def infer_tomograph_locations_watershed_ensemble(models, tomo, exp):
    inferer_kwargs = [
        dict(
            roi_size=(160, 384, 384),
            sw_batch_size=1,
            overlap=0.25,
            mode="gaussian",
            padding_mode="reflect",
        ),
    ]
    tomo = inference_transforms({"image": tomo})["image"].to(f"cuda")

    # for each inferer config, run the model with tta and add logits to the ensembled score
    predmask_accum = None
    count = 0
    with amp.autocast(f"cuda"):
        for inferer_kwarg in inferer_kwargs:
            for model in models:
                print(f"running {exp} with {inferer_kwarg}")
                inferer = SlidingWindowInferer(**inferer_kwarg)

                this_model_pred = tta_infer(tomo, model, inferer).squeeze()
                if predmask_accum is None:
                    # this is the first time we have a tensor, Initialize an accumulator tensor
                    predmask_accum = this_model_pred
                    count = 1
                else:
                    # we had something initialised already, add to the accumulation
                    predmask_accum += this_model_pred
                    count += 1
            torch.cuda.empty_cache()

        # compute the average logits
        predmask = predmask_accum / count
        predmask = predmask.softmax(0)

    locations = {}

    # post-process via watershed segmentation and skip beta-amylase
    predmask = cp.asarray(predmask)
    for idx, p in tqdm(enumerate(LABELS_7)):
        if p == "beta-amylase":
            continue
        pidx = idx + 1
        r = PARTICLE_RADIUS_7[p] / 10
        blob_threshold = blob_thresholds_7[p]
        certainty_threshold = CERTAINTY_THRESHOLDS_7[p]

        image = predmask[pidx].T > certainty_threshold

        # 2. Compute the distance transform
        distance = distance_transform_edt(image)
        coords = peak_local_max(distance, min_distance=int(r), labels=image)
        coords = cp.asnumpy(coords)
        mask = np.zeros(distance.shape, dtype=bool)
        mask[tuple(coords.T)] = True
        markers = measure.label(mask)

        distance = cp.asnumpy(distance)
        image = cp.asnumpy(image)
        labels = watershed(-distance, markers, mask=image)

        regions = measure.regionprops(labels)
        centroids = [region.centroid for region in regions if region.area >= blob_threshold]
        locations[p] = centroids

    df = dict_to_df(locations, exp)
    df["x"] *= 10.012444
    df["y"] *= 10.012444
    df["z"] *= 10.012444
    return df

def inference_on_runs( runs: list, device_number: int,):
    """
    load models, then for each given experiments (runs), run the model on those and save out one .csv per experiment
    """
    print(f"[{device_number}]: loading models")
    models = load_models(model_specs, device_number)
    print(f"[{device_number}]: loaded models")
    locs = []
    for _, run in enumerate(runs):
        start = time.time()

        tomo = run.get_voxel_spacing(10)
        tomo = tomo.get_tomogram(tomo_type).numpy()

        with torch.cuda.device(f"cuda:{device_number}"):
            with cp.cuda.Device(device_number):
                loc_df = infer_tomograph_locations_watershed_ensemble(models, tomo, run.name)
                locs.append(loc_df)
           

            torch.cuda.empty_cache()

        end = time.time()

        print(f"time taken: {end - start}")
    df = pd.concat(locs)
    df.to_csv("submission.csv")


if __name__ == "__main__":

    model_config_one = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 7,
        "channels": (32, 64, 128, 128),
        "strides": (2, 2, 1),
        "num_res_units": 1,
    }
    model_config_four = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 7,
        "channels": (32, 64, 128, 256),
        "strides": (2, 2, 1),
        "num_res_units": 2,
    }
    model_config_five = {
        "spatial_dims": 3,
        "in_channels": 1,
        "out_channels": 7,
        "channels": (32, 96, 256, 384),
        "strides": (2, 2, 1),
        "num_res_units": 2,
    }
    LABELS_7 = ["apo-ferritin", "beta-amylase", "beta-galactosidase", "ribosome", "thyroglobulin", "virus-like-particle"]
    PARTICLE_RADIUS_7 = {
        "apo-ferritin": 60,
        "beta-amylase": 65,
        "beta-galactosidase": 90,
        "ribosome": 150,
        "thyroglobulin": 130,
        "virus-like-particle": 135
    }
    blob_thresholds_7 = {
        "apo-ferritin": 80.09733552923254,
        "beta-amylase": 100,
        "beta-galactosidase": 368.0,
        "ribosome": 750.0,
        "thyroglobulin": 480.0,
        "virus-like-particle": 1150.3465099894624
    }
    CERTAINTY_THRESHOLDS_7 = {
        "apo-ferritin": 0.1,
        "beta-amylase": 0.1,
        "beta-galactosidase": 0.1,
        "ribosome": 0.1,
        "thyroglobulin": 0.1,
        "virus-like-particle": 0.1
    }

    cfg = get_cfg("./config.yml")
    seed_everything(cfg.task.seed)

    model_specs = [
        ModelSpec(cfg.inference.tiny_unet, model_config_one, False),
        ModelSpec(cfg.inference.medium_unet, model_config_four, False),
        ModelSpec(cfg.inference.large_unet, model_config_five, True),
        ModelSpec(cfg.inference.large_unet_other_tomo, model_config_five, True),
    ]

    load_models(model_specs, cfg.inference.device_id)
    # UNetModel.load

    root = copick.from_file("./test_copick.config")
    copick_user_name = "copickUtils"
    copick_segmentation_name = "paintedPicks"
    voxel_size = 10
    tomo_type = "denoised"

    all_runs = root.runs
    print(f"running models sequentially")
    inference_on_runs(all_runs, cfg.inference.device_id)
   
