#!/usr/bin/bash

echo "downloading post processed experiment data"
kaggle kernels output fnands/create-numpy-dataset-exp-name -p ./data/train
echo "downloaded post processed experiment data"

# note: putting all data into one notebook seems to lead to kaggle not downloading all data and leaving some
#  data behind, so I split it into 3 instead to side step this issue
echo "downloading post processed synthetic data 1/3..."
kaggle kernels output sirapoabchaikunsaeng/simulated-data-and-labels-syntetic -p ./data/synthetic-data
echo "downloading post processed synthetic data 2/3..."
kaggle kernels output sirapoabchaikunsaeng/simulated-data-and-labels-syntetic-2-3 -p ./data/synthetic-data
echo "downloading post processed synthetic data 3/3..."
kaggle kernels output sirapoabchaikunsaeng/simulated-data-and-labels-syntetic-3-3 -p ./data/synthetic-data
echo "downloaded post processed synthetic data"

