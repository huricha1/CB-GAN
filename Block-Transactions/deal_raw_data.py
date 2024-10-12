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
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('block/'),
                    help="Experiment path")
parser.add_argument("--modelPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/' + experimentName + '/model'),
                    help="Experiment path")
opt = parser.parse_args()
pd.set_option('display.max_columns', None)
DATASETDIR = os.path.expanduser('../datasets/')
df = pd.read_csv(os.path.join(DATASETDIR, '0to999999_BlockTransaction.csv'))
df=df[818581:823276]
# print(df.columns)
df=df.drop(['transactionHash'], axis=1)
df=df.drop(['from'], axis=1)
df=df.drop(['to'], axis=1)
df=df.drop(["toCreate"], axis=1)
df=df.drop(['callingFunction'], axis=1)
df=df.drop(['eip2718type'], axis=1)
df=df.drop(['baseFeePerGas'], axis=1)
df=df.drop(['maxFeePerGas'], axis=1)
df=df.drop(['maxPriorityFeePerGas'], axis=1)
df=df.drop(['fromIsContract'], axis=1)
pd.set_option('display.max_columns', None)
# df['isError'] = (df['isError'] is 'None').replace(True,0).replace(False,1)
# df['isError']=df['isError'].replace('None','0',inplace=True)
# df['isError']=df['isError'].replace('Stack underflow','1',inplace=True)
# df['isError']=df['isError'].replace('Out of stack','1',inplace=True)
# df['isError']=df['isError'].replace('Out of gas','1',inplace=True)
df['isError']= df['isError'].astype('int')
df1=pd.DataFrame(df[['blockNumber','timestamp','value','gasLimit','gasPrice','gasUsed','isError']])
df0=pd.DataFrame(df[['toIsContract']])
df0=pd.get_dummies(df0,columns=df0.columns)
df = pd.concat([df0,df1],axis=1)

# Train/test split
train_df, test_df = skl.train_test_split(df,
                                   test_size = 0.3,
                                   stratify = df['isError'])
train_df.to_csv(os.path.join(opt.expPATH, "dataTrain.csv"),index=False)
test_df.to_csv(os.path.join(opt.expPATH, "dataTest.csv"),index=False)