import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score


data = pd.read_csv('diabetes.csv',header=0)

stmm = MinMaxScaler()
ss = StandardScaler()
data['class'] = data['class'].map({
    'tested_positive':0,
    'tested_negative':1
})

# 加载数据
feature,label = data.iloc[:,:-1],data.iloc[:,-1]
feature = stmm.fit_transform(feature)
# feature = ss.fit_transform(feature)
X_train, X_test, y_train,y_test = train_test_split(feature, label, test_size=0.3, random_state=0)


clf1=svm.SVC()

# 网格参数寻参
param_grid = {'C': [1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
              'gamma': [0.001, 0.0001],
            'kernel':('linear','rbf',),
              }

grid_search = GridSearchCV(clf1, param_grid, n_jobs = 8, verbose=1,scoring='f1')

grid_search.fit(X_train, y_train)
print("The best parameters are %s with a score of %0.2f"
      % (grid_search.best_params_, grid_search.best_score_))#找到最佳超参数
best_parameters = grid_search.best_estimator_.get_params()
# for para, val in list(best_parameters.items()):
#     print(para, val)

model = svm.SVC(kernel=best_parameters['kernel'], C=best_parameters['C'], gamma=best_parameters['gamma'], probability=True)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
# 留出法验证
print("Accuracy on test data: {:.2f}".format(model.score(X_test, y_test)))
print("f1 on test data: {:.2f}".format(f1_score(y_test,predictions)))


# clf1.fit(X_train,y_train)
# accs=cross_val_score(clf1, feature, y=label, scoring=None,cv=10, n_jobs=1)
# print('训练集合交叉验证结果:',accs)
# print('交叉验证均值:',accs.mean())

# y_predictions1=clf1.predict(X_test)
# print("Accuracy on test data: {:.2f}".format(clf1.score(X_test, y_test)))