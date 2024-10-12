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
import statistics
from sklearn.preprocessing import StandardScaler, MinMaxScaler
parser = argparse.ArgumentParser()

# experimentName is the current file name without extension
experimentName = os.path.splitext(os.path.basename(__file__))[0]
experimentName = 'rdp'
parser.add_argument("--DATASETPATH", type=str,
                    default=os.path.expanduser('~/data/MIMIC/processed/out_binary.matrix'),
                    help="Dataset file")
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('cancer/'),
                    help="Experiment path")
parser.add_argument("--modelPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/' + experimentName + '/model'),
                    help="Experiment path")
opt = parser.parse_args()
DATASETDIR = os.path.expanduser('../datasets/')
df = pd.read_csv(os.path.join(DATASETDIR, 'kag_risk_factors_cervical_cancer.csv'))
df=df.drop(['STDs: Time since first diagnosis'], axis=1)
df=df.drop(['STDs: Time since last diagnosis'], axis=1)
pd.set_option('display.max_columns', None)
df['Number of sexual partners'].fillna(3,inplace=True)
df['First sexual intercourse'].fillna(17,inplace=True)
df['Num of pregnancies'].fillna(2,inplace=True)

df['Smokes'].fillna(0,inplace=True)
df['Smokes (years)'].fillna(df['Smokes (years)'].mean(),inplace=True)
df['Smokes (packs/year)'].fillna(df['Smokes (packs/year)'].mean(),inplace=True)


df['Hormonal Contraceptives'].fillna(1,inplace=True)
df['Hormonal Contraceptives (years)'].fillna(df['Hormonal Contraceptives (years)'].mean(),inplace=True)

df['IUD'].fillna(0,inplace=True)
df['IUD (years)'].fillna(df['IUD (years)'].mean(),inplace=True)

for i in range(11,34):
    df.iloc[:,i].fillna(0, inplace=True)


# Train/test split
train_df, test_df = skl.train_test_split(df,
                                   test_size = 0.2,
                                   stratify = df['Biopsy'])
train_df.to_csv(os.path.join(opt.expPATH, "dataTrain.csv"),index=False)
test_df.to_csv(os.path.join(opt.expPATH, "dataTest.csv"),index=False)