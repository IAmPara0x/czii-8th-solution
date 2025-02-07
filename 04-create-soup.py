#!/usr/bin/env python

from monai.data.meta_tensor import MetaTensor
from omegaconf import OmegaConf

from collections import OrderedDict
import torch

from src.utils import *
from src.augmentations import *

torch.serialization.add_safe_globals([MetaTensor])
torch.set_float32_matmul_precision('medium')

def model_soup(models, method="uniform"):
    """
    Perform model soup by averaging weights of multiple models.
    
    Args:
        models (list): List of PyTorch models (must have identical architectures).
        method (str): "uniform" (default) or "greedy" for weight averaging strategy.
        
    Returns:
        torch.nn.Module: A new model with averaged weights.
    """
    assert len(models) > 0, "Provide at least one model for souping."
    
    # Get state_dicts from all models
    state_dicts = [model.model.state_dict() for model in models]
    
    # Create an empty OrderedDict for the new model's weights
    avg_state_dict = OrderedDict()

    # Uniform Model Soup (Simple Weight Averaging)
    if method == "uniform":
        for key in state_dicts[0]:  # Iterate over parameter keys
            avg_state_dict[key] = torch.mean(torch.stack([sd[key] for sd in state_dicts]), dim=0)
    
    else:
        raise ValueError(f"Unknown method: {method}. Supported: ['uniform']")
    
    # Load the averaged weights into a new model
    
    models[0].model.load_state_dict(avg_state_dict)
    
    return models[0]


if __name__ == "__main__":

    cfg = get_cfg("config.yml")
    print(OmegaConf.to_yaml(cfg))

    seed_everything(cfg.task.seed)
