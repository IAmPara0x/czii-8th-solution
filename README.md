# 8th place solution of kaggle czii competition


# Quickstart
### Install Dependencies
```bash
pip3 install -r requirements.txt
```

### Training All Models
```bash
./get-data.sh # to get the data to train the models
./train.sh # to train the models
./soup.sh # to create model soup
```
#### Output Artifacts
- tiny unet model soup = `checkpoints/tiny-unet/`
- medium unet model soup = `checkpoints/medium-unet/`
- big unet model soup on denoised tomograms = `sumo/checkpoint_finetune_denoised/checkpoint_finetune_denoised_soup.pth`
- big unet model soup on all tomograms= `sumo/checkpoint_finetune_all/checkpoint_finetune_all_soup.pth`

#### Running this script does the following
- downloads data to `data/`
  - `data/train` for real experiments data
  - `data/synthetic-data` for real synthetic data
- train all models
  - pretrain and finetune tiny unet models, to `checkpoints/tiny-unet`
  - pretrain and finetune medium unet models, to `checkpoints/medium-unet`
  - pretrain big unet models, to `sumo/checkpoint_pretraining`
  - finetune big unet models on denoised tomograms, to `sumo/checkpoint_finetune_denoised`
  - finetune big unet models on all tomograms, to `sumo/checkpoint_finetune_all`
  - create model soup for each model trained

#### Key Assumptions
- we assume that user has the kaggle api setup as we use them to get the data from kaggle
- the data directory `data/` is empty before `./get-data.sh` is run
- all checkpoints directories are empty before any trainings are run


### Inference

```bash
python ./05-inference.py # run the command for inference
```

- For inference all the tomograms should be placed under ./data/test/static/ folder
- The folder is not created by default and must be created by the user
- The result of inference is submission.csv

---

# Dependencies

## Hardware
| User                 | OS                          | Memory            | GPU                 |
|----------------------|-----------------------------|-------------------|---------------------|
| IAmParadox           | Ubuntu                      | 64Gb              | RTX 4090            |
| sirapoabchaikunsaeng | Ubuntu 22.04 + Ubuntu 24.04 | 64Gb + 500Gb Swap | RTX 3090 + RTX 4090 |
| sersasj              | Kaggle Kernel               | -                 | 2x T4               |
| itsuki9180           |  TBD                        |  TBD              |  TBD                |

## Software
all software dependencies are listed inside `requirements.txt`, we did not do any additional software installations other than cuda and gpu drivers

---

## Configuration Files
- `config.yaml` is the base config for tiny and medium unet models. As there's a lot of field in there, please refer to the comments inside the config file to see what each field does
- `sumo/pretraining.yaml` is the config for pretraining big-unet models
- `sumo/finetune_denoised.yaml` is the config for finetuning big-unet models on denoised tomograms
- `sumo/finetune_all.yaml` is the config for finetuning big-unet models on all tomograms
