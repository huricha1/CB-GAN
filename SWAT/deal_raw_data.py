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
from torch.nn.utils import clip_grad_norm_
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
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('swat/'),
                    help="Experiment path")
parser.add_argument("--modelPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/' + experimentName + '/model'),
                    help="Experiment path")
opt = parser.parse_args()
DATASETDIR = os.path.expanduser('../datasets/')
df = pd.read_csv(os.path.join(DATASETDIR, 'SWaT_Dataset_Attack_v0.csv'))
df=df.drop([' Timestamp'], axis=1)
df=df.drop(['P202'], axis=1)
df=df.drop(['P301'], axis=1)
df=df.drop(['P401'], axis=1)
df=df.drop(['P404'], axis=1)
df=df.drop(['P502'], axis=1)
df=df.drop(['P601'], axis=1)
df=df.drop(['P603'], axis=1)
df1=pd.DataFrame(df[['FIT101','LIT101','AIT201','AIT202','AIT203','FIT201',
                            'DPIT301','FIT301','LIT301','AIT401','AIT402','FIT401','LIT401','AIT501','AIT502','AIT503','AIT504','FIT501','FIT502','FIT503'
                     ,'FIT504','PIT501','PIT502','PIT503','FIT601','n']])
df2=pd.DataFrame(df[['MV101','P101','P102','MV201','P201','P203','P204','P205','P206','MV301','MV302',
                           'MV303','MV304','P302','P402','P403','UV401','P501','P602']])

df2=pd.get_dummies(df2,columns=df2.columns)
df = pd.concat([df2,df1],axis=1)
df['n']= df['n'].astype('int')
pd.set_option('display.max_columns', None)
# Train/test split
train_df, test_df = skl.train_test_split(df,
                                   test_size = 0.2,
                                   stratify = df['n'])
train_df.to_csv(os.path.join(opt.expPATH, "dataTrain.csv"),index=False)
test_df.to_csv(os.path.join(opt.expPATH, "dataTest.csv"),index=False)