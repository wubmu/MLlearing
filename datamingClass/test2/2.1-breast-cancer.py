import pandas as pd
import numpy as np
from sklearn import model_selection, naive_bayes
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
data = pd.read_csv('breast-cancer.csv',header=0)
print(data.info())
print(data.describe())
# 加载数据
feature,label = data.iloc[:,:-1],data.iloc[:,-1]
feature,label = feature.values,label.values
X_train, X_test, y_train,y_test = train_test_split(feature, label, test_size=0.3, random_state=0)

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train)

model = RandomForestClassifier(max_depth=6, n_estimators=200, random_state=7)
cv = cross_val_score(model, X_train, y_train, cv=4, scoring='f1_weighted')
print(cv.mean())
model.fit(X_train,X_test)

print("Accuracy on test data: {:.2f}".format(clf.score(X_test, y_test)))

# gnb = GaussianNB()
# y_pred = gnb.fit(X_train, y_train)
# print("Number of mislabeled points out of a total %d points : %d"
#       % (X_test.shape[0], (y_test != y_pred).sum()))

def test_GaussianNB(*data, show=False):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.GaussianNB()
    cls.fit(X_train, y_train)
    print('GaussianNB Training Score: %.2f' % cls.score(X_train, y_train))
    print('GaussianNB Testing Score: %.2f' % cls.score(X_test, y_test))

def test_MultinomialNB(*data, show=False):
    X_train, X_test, y_train, y_test = data
    cls = naive_bayes.MultinomialNB()
    cls.fit(X_train, y_train)
    # print('MultinomialNB Training Score: %.2f' % cls.score(X_train, y_train))
    print('MultinomialNB Testing Score: %.2f' % cls.score(X_test, y_test))


def test_MultinomialNB_alpha(*data, show=False):
    X_train, X_test, y_train, y_test = data
    alphas = np.logspace(-2, 5, num=200)
    train_scores = []
    test_scores = []
    for alpha in alphas:
        cls = naive_bayes.MultinomialNB(alpha=alpha)
        cls.fit(X_train, y_train)
        train_scores.append(cls.score(X_train, y_train))
        test_scores.append(cls.score(X_test, y_test))

    if show:
        ## 绘图:MultinomialNB 的预测性能随 alpha 参数的影响
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(alphas, train_scores, label='Training Score')
        ax.plot(alphas, test_scores, label='Testing Score')
        ax.set_xlabel(r'$\alpha$')
        ax.set_ylabel('score')
        ax.set_ylim(0, 1.0)
        ax.set_title('MultinomialNB')
        ax.set_xscale('log')
        plt.show()

    print('MultinomialNB_alpha best train_scores %.2f' % (max(train_scores)))
    print('MultinomialNB_alpha best test_scores %.2f' % (max(test_scores)))