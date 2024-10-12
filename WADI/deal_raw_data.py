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
parser.add_argument("--expPATH", type=str, default=os.path.expanduser('wadi/'),
                    help="Experiment path")
parser.add_argument("--modelPATH", type=str, default=os.path.expanduser('~/experiments/pytorch/' + experimentName + '/model'),
                    help="Experiment path")
opt = parser.parse_args()
pd.set_option('display.max_columns', None)
DATASETDIR = os.path.expanduser('../datasets/')
df = pd.read_csv(os.path.join(DATASETDIR, 'WADI_attackdata_labelled.csv')).drop('Unnamed: 0', axis=1)
df=df[5102:75568]
print(df.loc[df['attack']==1].shape)
print(df.shape)
# print(df[r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_LS_001_AL'])
df=df.drop(['Row'], axis=1)
df=df.drop(['Date'], axis=1)
df=df.drop(['Time'], axis=1)
df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_LS_001_AL"], axis=1)
df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_LS_002_AL"], axis=1)
df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_P_002_STATUS"], axis=1)

df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_P_003_STATUS"], axis=1)
df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_P_004_STATUS"], axis=1)
df=df.drop([r"\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_001_AL"], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_002_AL'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_001_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_002_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_001_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_002_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_004_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_005_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_009_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_004_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_101_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_201_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_301_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_401_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_501_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_SV_601_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_AIT_001_PV'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_AIT_002_PV'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_LS_001_AL'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_MV_001_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_MV_002_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_MV_003_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_P_001_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_P_002_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_P_003_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_P_004_STATUS'], axis=1)
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\PLANT_START_STOP_LOG'], axis=1)
# feature=134
df=df.drop([r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_P_006_STATUS'], axis=1)
# print(df.columns)
# print(df.shape)

df1=pd.DataFrame(df[[r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_AIT_001_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_AIT_002_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_AIT_003_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_AIT_004_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_AIT_005_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_FIT_001_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_LT_001_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_DPIT_001_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_101_CO',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_101_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_101_SP',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_201_CO',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_201_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_201_SP',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_301_CO',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_301_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_301_SP',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_401_CO',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_401_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_401_SP'
                     ,r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_501_CO',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_501_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_501_SP',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_601_CO',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_601_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIC_601_SP',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIT_001_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIT_002_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FIT_003_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FQ_101_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FQ_201_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FQ_301_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FQ_401_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FQ_501_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_FQ_601_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LT_001_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LT_002_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MCV_007_CO',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MCV_101_CO',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MCV_201_CO',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MCV_301_CO',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MCV_401_CO',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MCV_501_CO',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MCV_601_CO',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_003_SPEED',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_004_SPEED',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_PIC_003_CO',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_PIC_003_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_PIC_003_SP',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_PIT_001_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_PIT_002_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_PIT_003_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2A_AIT_001_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2A_AIT_002_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2A_AIT_003_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2A_AIT_004_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2B_AIT_001_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2B_AIT_002_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2B_AIT_003_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2B_AIT_004_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_AIT_003_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_AIT_004_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_AIT_005_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_FIT_001_PV',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\3_LT_001_PV',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\LEAK_DIFF_PRESSURE',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\TOTAL_CONS_REQUIRED_FLOW','attack']])

df2=pd.DataFrame(df[[r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_MV_001_STATUS',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_MV_002_STATUS',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_MV_003_STATUS',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_MV_004_STATUS',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_P_001_STATUS',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\1_P_005_STATUS',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_101_AH',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_101_AL',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_201_AH',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_201_AL',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_301_AH',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_301_AL',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_401_AH',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_401_AL',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_501_AH',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_501_AL',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_601_AH',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_LS_601_AL',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_003_STATUS',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_006_STATUS',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_101_STATUS',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_201_STATUS',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_301_STATUS',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_401_STATUS',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_501_STATUS',r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_MV_601_STATUS',
                     r'\\WIN-25J4RO10SBF\LOG_DATA\SUTD_WADI\LOG_DATA\2_P_003_STATUS']])

df2=pd.get_dummies(df2,columns=df2.columns)
df = pd.concat([df2,df1],axis=1)
count_ones = (df['attack'] == 1).sum().sum()
print(count_ones)
# print(df.loc[df['attack']==1].shape)
pd.set_option('display.max_columns', None)
# Train/test split
train_df, test_df = skl.train_test_split(df,
                                   test_size = 0.3,
                                   stratify = df['attack'])
train_df.to_csv(os.path.join(opt.expPATH, "dataTrain.csv"),index=False)
test_df.to_csv(os.path.join(opt.expPATH, "dataTest.csv"),index=False)