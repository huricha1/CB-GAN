import argparse
import numpy as np
import time
import random
import os
import sklearn.model_selection as skl
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch
import math
import dp_optimizer
from torch.nn.utils import clip_grad_norm_
import analysis
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
parser = argparse.ArgumentParser()

# experimentName is the current file name without extension
experimentName = os.path.splitext(os.path.basename(__file__))[0]
experimentName = 'rdp'
parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('~/data/MIMIC/processed/out_binary.matrix'),
                    help="Dataset file")
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('uci/'),
                    help="Experiment path")
parser.add_argument("--modelPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/' + experimentName + '/model'),
                    help="Experiment path")
opt = parser.parse_args()
df = pd.read_csv('../datasets/data.csv').drop('Unnamed: 0', axis=1)
df['y'] = (df['y'] == 1).replace(True,1).replace(False,0)
df.to_csv('uci/df.csv',index=False)
# Train/test split
train_df, test_df = skl.train_test_split(df,
                                   test_size = 0.2,
                                   stratify = df['y'])
trainData = train_df.to_numpy()
testData = test_df.to_numpy()
trainData = trainData.astype(np.float32)
testData = testData.astype(np.float32)
# ave synthetic data
np.save(os.path.join(opt.expPATH, "dataTrain.npy"), trainData, allow_pickle=False)
np.save(os.path.join(opt.expPATH, "dataTest.npy"), testData, allow_pickle=False)