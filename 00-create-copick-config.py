#!/usr/bin/env python



copick_config_path = "./data/copick.config"
output_overlay = "./data/overlay"


if __name__ == "__main__":


    config_blob = """
    {
        "name": "czii_cryoet_mlchallenge_2024",
        "description": "2024 CZII CryoET ML Challenge training data.",
        "version": "1.0.0",

        "pickable_objects": [
            {
                "name": "apo-ferritin",
                "is_particle": true,
                "pdb_id": "4V1W",
                "label": 1,
                "color": [  0, 117, 220, 128],
                "radius": 60,
                "map_threshold": 0.0418
            },
            {
                "name": "beta-amylase",
                "is_particle": true,
                "pdb_id": "1FA2",
                "label": 2,
                "color": [153,  63,   0, 128],
                "radius": 65,
                "map_threshold": 0.035
            },
            {
                "name": "beta-galactosidase",
                "is_particle": true,
                "pdb_id": "6X1Q",
                "label": 3,
                "color": [ 76,   0,  92, 128],
                "radius": 90,
                "map_threshold": 0.0578
            },
            {
                "name": "ribosome",
                "is_particle": true,
                "pdb_id": "6EK0",
                "label": 4,
                "color": [  0,  92,  49, 128],
                "radius": 150,
                "map_threshold": 0.0374
            },
            {
                "name": "thyroglobulin",
                "is_particle": true,
                "pdb_id": "6SCJ",
                "label": 5,
                "color": [ 43, 206,  72, 128],
                "radius": 130,
                "map_threshold": 0.0278
            },
            {
                "name": "virus-like-particle",
                "is_particle": true,
                "pdb_id": "6N4V",            
                "label": 6,
                "color": [255, 204, 153, 128],
                "radius": 135,
                "map_threshold": 0.201
            }
        ],

        "overlay_root": "./data/overlay",

        "overlay_fs_args": {
            "auto_mkdir": true
        }

    }
    """


    with open(copick_config_path, "w") as f:
        f.write(config_blob)
