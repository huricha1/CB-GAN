import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def visual(feat,feat1):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts = ts.fit_transform(feat)
    print(x_ts.shape)  # [num, 2]
    x_min, x_max = x_ts.min(0), x_ts.max(0)
    x_ts = (x_ts - x_min) / (x_max - x_min)
    np.savetxt('real.txt', x_ts, fmt='%s', newline='\n')
    ts1 = manifold.TSNE(n_components=2, init='pca', random_state=0)
    x_ts1 = ts1.fit_transform(feat1)
    x_min, x_max = x_ts1.min(0), x_ts1.max(0)
    x_ts1 = (x_ts1 - x_min) / (x_max - x_min)
    np.savetxt('syn.txt', x_ts1, fmt='%s', newline='\n')
    x_final = np.concatenate((x_ts, x_ts1), axis=0)

    return x_final


# 设置散点形状
maker = ['o', 'o', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
# 设置散点颜色
colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']
# 图例名称
Label_Com = ['a', 'b', 'c', 'd']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }


def plotlabels(S_lowDWeights, Trure_labels, name):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data.shape)  # [num, 3]

    for index in range(2):  # 假设总共有2个类别，类别的表示为0,1
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        if(index==0):
            s1=plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)
        else:
            s2=plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index],edgecolors=colors[index], alpha=0.65)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.legend((s1, s2), ('original data', 'Synthetic data'), loc='best')
    # plt.legend(handles=s.legend_elements()[0], labels=["original data", "Synthetic data"])
    # plt.title(name, fontsize=32, fontweight='normal', pad=20)


# feat = torch.rand(128, 1024)  # 128个特征，每个特征的维度为1024
trainData=pd.read_csv("dataTrain.csv")
train_f = np.load(f'syntheticlabel100.npy', allow_pickle=False)

trainData = trainData.to_numpy()
trainData = trainData.astype(np.float32)
sc = MinMaxScaler()
trainData = sc.fit_transform(trainData)



x=trainData.shape[0]
label_test1 = [0 for index in range(x)]
label_test2 = [1 for index in range(x)]


label_test = np.array(label_test1 + label_test2 )
print(label_test)
print(label_test.shape)

fig = plt.figure(figsize=(10, 10))

plotlabels(visual(trainData,train_f), label_test, '(a)')

plt.show()