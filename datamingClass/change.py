'''
中心点横坐标（region-centroid-col）：取样中心点所在的图像的横坐标；

中心点纵坐标（region-centroid-row）：取样中心点所在的图像的纵坐标；

每个样本所含点的数量（region-pixel-count）：为9；

低密集度计数（short-line-density-5）指在通过这个区域的在任意方向上
的，长度为5的线段当中，有多少条对比度大小要低于或等于5；

高密集度计数（short-line-density-2）指在通过这个区域的在任意方向上
的，长度为5的线段当中，有多少条对比度大小要高于5；

横向像素差值的平均（vedge-mean）：指在3x3的样本中，所有的左右相
邻的两像素亮度之差的绝对值（共有6个）的平均数；

横向像素差值的标准差（vedge-sd）：上述像素差值的标准差；
纵向像素差值的平均（hedge-mean）：指在3x3的样本中，所有的上下相
邻的两像素亮度之差的绝对值（共有6个）的平均数；

纵向像素差值的标准差（hedge-sd）：上述像素差值的标准差；

整体亮度的平均数（intensity-mean）：亮度按(R + G + B)/3计算（上同），
再根据9个点的这些亮度取平均数

红分量平均（rawred-mean）：整个样本区域的红分量的平均值

蓝分量平均（rawblue-mean）：整个样本区域的绿分量的平均值

绿分量平均（rawgreen-mean）：整个样本区域的lan分量的平均值

红色超出量（exred-mean）：测量红色多于其他颜色分量的程度，按(2R -

(G + B))的公式计算

蓝色超出量（exblue-mean）：测量蓝色多于其他颜色分量的程度

绿色超出量（exgreen-mean）：测量绿色多于其他颜色分量的程度
'''

import numpy as np
import pandas as  pd
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import KFold

data = pd.read_csv('challenge.csv',header=0)
feature = data.iloc[:, :-1]
label = data.iloc[:,-1]
print(feature.head())
print(label.head())
## 标准化
scaler = StandardScaler()
feature_scaler = scaler.fit_transform(feature)
print(feature_scaler[:10])

##切分数据集
data = np.column_stack((feature_scaler,label))
print(data[:10])
kf=KFold(n_splits=5,shuffle=True)

