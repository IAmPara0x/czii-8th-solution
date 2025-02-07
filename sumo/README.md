## Sumo's Models

This directory reproduces two of the models in the [final submission](https://www.kaggle.com/code/iamparadox/3x-unet-1xsliding-window?scriptVersionId=220923693)

### Pretraining
```bash
python3 sumo/train.py --config-name=pretraining --val=TS_0

python3 sumo/train.py --config-name=finetune_denoised --val=TS_69_2 && \
python3 sumo/train.py --config-name=finetune_denoised --val=TS_86_3 && \
python3 sumo/train.py --config-name=finetune_denoised --val=TS_99_9

python3 sumo/train.py --config-name=finetune_all --val=TS_69_2 && \
python3 sumo/train.py --config-name=finetune_all --val=TS_86_3 && \
python3 sumo/train.py --config-name=finetune_all --val=TS_99_9
```
