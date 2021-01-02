import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
import matplotlib.pyplot as plt
data = pd.read_csv('diabetes.csv',header=0)
data['class'] = data['class'].map({
    'tested_positive':0,
    'tested_negative':1
})


stmm = MinMaxScaler()
# 加载数据
feature,label = data.iloc[:,:-1],data.iloc[:,-1]
feature = stmm.fit_transform(feature)


X_train, X_test, y_train,y_test = train_test_split(feature, label, test_size=0.3, random_state=0)

clf1=RandomForestClassifier(n_estimators=71, random_state=90)
clf1.fit(X_train, y_train)

# 用交叉验证计算得分
score_pre = cross_val_score(clf1, feature, label, cv=10).mean()
score_f1 = cross_val_score(clf1, feature, label, cv=10,scoring='f1').mean()
# score_f2 = cross_val_score(clf1, feature, label, cv=10,scoring='f2').mean()
# score_auc = cross_val_score(clf1, feature, label, cv=10,scoring='auc').mean()

print(f'交叉验证Accuracy core：{score_pre}')
# print(f'交叉验证f1 score：{score_pre}')
# print(f'交叉验证auc score：{score_auc}')

print(f'交叉验证f1 score:{score_f1}')
# print(f'交叉验证f2 score:{score_f2}')
# 留出法验证
print("留出法Accuracy on test data: {:.2f}".format(clf1.score(X_test, y_test)))


# 调参，绘制学习曲线来调参n_estimators（对随机森林影响最大）
score_lt = []

# 每隔10步建立一个随机森林，获得不同n_estimators的得分
for i in range(0,200,10):
    rfc = RandomForestClassifier(n_estimators=i+1
                                ,random_state=90)
    score = cross_val_score(rfc, feature, label, cv=10).mean()
    score_lt.append(score)
score_max = max(score_lt)
print('最大得分：{}'.format(score_max),
      '子树数量为：{}'.format(score_lt.index(score_max)*10+1))

# 绘制学习曲线
x = np.arange(1,201,10)
plt.subplot(111)
plt.plot(x, score_lt, 'r-')
plt.show()