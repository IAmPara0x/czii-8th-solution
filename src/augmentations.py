
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import EnsureChannelFirstd
from monai.transforms.intensity.dictionary import NormalizeIntensityd
from monai.transforms.spatial.dictionary import Orientationd, RandFlipd
from monai.transforms.croppad.dictionary import RandCropByLabelClassesd


# Non-random transforms to be cached
non_random_transforms = Compose([
    EnsureChannelFirstd(keys=["image", "label"], channel_dim="no_channel"),
    NormalizeIntensityd(keys="image"),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
])

def get_random_transforms(cfg):
    return Compose([
           RandCropByLabelClassesd(
                keys=["image", "label"],
                label_key="label",
                spatial_size= [cfg.pretrain.patch_size, cfg.pretrain.patch_size, cfg.pretrain.patch_size],
                num_classes=cfg.task.num_classes,
                num_samples=cfg.task.num_samples,
            ),

            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2), 
        ])

# Inference transforms
inference_transforms = Compose([
    EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
    NormalizeIntensityd(keys="image"),
    Orientationd(keys=["image"], axcodes="RAS")
])
