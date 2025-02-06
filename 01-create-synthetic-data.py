#!/usr/bin/env python

import copick
import fileinput
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import zarr

from glob import glob
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, median_filter

def ndjson_to_pick(run, particle, src_path, dest_path):
    pick = {}
    pick['pickable_object_name'] = particle
    pick['user_id'] = 'curation'
    pick['session_id'] = '0'
    pick['run_name'] = run
    pick['voxel_spacing'] = None
    pick['unit'] = 'angstrom'
    pick['points'] = []

    lines = fileinput.input(files=[src_path])
    for line in lines:
        nd_point = json.loads(line)
        point = {}
        point['location'] = {}
        point['location']['x'] = 10.012*nd_point['location']['x']
        point['location']['y'] = 10.012*nd_point['location']['y']
        point['location']['z'] = 10.012*nd_point['location']['z']
        point['transformation_'] = [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        point['instance_id'] = 0
        pick['points'].append(point)
        
    lines.close()
    
    with open(dest_path, 'w') as f:
        f.write(json.dumps(pick))



def denoise_tomogram(tomogram, method='gaussian', **kwargs):
    """
    Apply denoising to a tomogram.

    Parameters:
        tomogram (np.ndarray): The input tomogram to denoise.
        method (str): The denoising method ('gaussian' or 'median').
        kwargs: Parameters for the respective method.
    
    Returns:
        np.ndarray: The denoised tomogram.
    """
    if method == 'gaussian':
        return gaussian_filter(tomogram, sigma=kwargs.get('sigma', 1))
    elif method == 'median':
        return median_filter(tomogram, size=kwargs.get('size', 3))
    else:
        raise ValueError(f"Unsupported denoising method: {method}")


if __name__ == "__main__":

    pick_map = {
        'ferritin_complex': 'apo-ferritin',
        'beta_amylase': 'beta-amylase',
        'beta_galactosidase': 'beta-galactosidase',
        'cytosolic_ribosome': 'ribosome',
        'thyroglobulin': 'thyroglobulin',
        'pp7_vlp': 'virus-like-particle',
    }


    ndjson_files = glob('/kaggle/input/czii10441/10441/**/*.ndjson',
                        recursive=True)

    for file in ndjson_files:
        print(file)
        run = file.split('/')[5]
        print(run)
        if "_" not in run:
            continue
        particle = pick_map[file.split('/')[10].split('-')[0]]
        dest_dir = f'/kaggle/working/overlay/ExperimentRuns/{run}/Picks'
        dest_path = f'{dest_dir}/curation_0_{particle}.json'
        os.makedirs(dest_dir, exist_ok=True)
        ndjson_to_pick(run, particle, file, dest_path)


    ndjson_files = glob('/kaggle/input/czii10441/10441/**/*.ndjson',
                        recursive=True)


