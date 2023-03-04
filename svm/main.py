import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.inspection import DecisionBoundaryDisplay


def add_fictive_column(X):
    fictive = np.array([-1 for _ in range(len(X))])
    return np.insert(X, len(X[0]), fictive, axis=1)


def getScoreLogRegression(param_c):
    log_regression = LogisticRegression(penalty='l2', C=param_c, max_iter=3000)
    log_regression.fit(X_train, y_train)
    return log_regression.score(X_test, y_test)


def getScoreSVM(param_c):
    svm = LinearSVC(penalty='l2', C=param_c, max_iter=2000)
    svm.fit(X_train, y_train)
    return svm.score(X_test, y_test)


def paintRelationAccuracy_C(ax, title, arr):
    ax.plot([i for i in range(1, n)], arr)
    ax.set_title(title)
    ax.set(xlabel='Параметр C', ylabel='Accuracy')


def paintGraphic(classifier, X, y, ax):

    disp = DecisionBoundaryDisplay.from_estimator(
        classifier, X, response_method='predict',
        xlabel='дисперсия вейвлет-преобразованного изображения',
        ylabel='асимметрия вейвлет-преобразованного изображения',
        ax=ax
    )
    disp.ax_.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k')


def paint(log_reg_c, svm_c):
    X = X_test[:, 2:4]

    paintRelationAccuracy_C(axis[0, 0], 'Логистическая регрессия', arrCForLogRegScore)
    log_regression = LogisticRegression(penalty='l2', C=log_reg_c, max_iter=3000)
    log_regression.fit(X, y_test)
    paintGraphic(log_regression, X, y_test, axis[1, 0])

    paintRelationAccuracy_C(axis[0, 1], 'SVM', arrCForSVMScore)
    svm = LinearSVC(penalty='l2', C=svm_c, max_iter=2000)
    svm.fit(X, y_test)
    paintGraphic(log_regression, X, y_test, axis[1, 1])

    plt.show()


figure, axis = plt.subplots(2, 2)
figure.set_figheight(8)
figure.set_figwidth(17)

# Загрузка датасета, деление на объекты и ответы
data = pd.read_csv('dataset/banknote.csv', sep=',').to_numpy()
X = np.array([data[i][:-1] for i in range(len(data))])
y = np.array([data[i][-1] for i in range(len(data))])

# Стандартизация для объектов
scaler = preprocessing.StandardScaler().fit(X)
X_scaled = scaler.transform(X)
X_scaled = add_fictive_column(X_scaled)

# Деление выборки на обучающую и контрольную
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.30, random_state=0)

arrCForLogRegScore = []
arrCForSVMScore = []
optimalCForLogReg = 1
optimalCForSVM = 1

logRegScore = getScoreLogRegression(optimalCForLogReg)
arrCForLogRegScore.append(logRegScore)

svmScore = getScoreSVM(optimalCForSVM)
arrCForSVMScore.append(svmScore)

n = 10
for i in range(2, n):
    # Логистическая регрессия
    tempLogRegScore = getScoreLogRegression(i)
    arrCForLogRegScore.append(tempLogRegScore)

    if logRegScore < tempLogRegScore:
        logRegScore = tempLogRegScore
        optimalCForLogReg = i

    # Метод опорных векторов
    tempSVMScore = getScoreLogRegression(i)
    arrCForSVMScore.append(tempSVMScore)

    if svmScore < tempSVMScore:
        svmScore = tempSVMScore
        optimalCForSVM = i

print('Логистическая регрессия: оптимальный C =', optimalCForLogReg, ', accuracy =', logRegScore)
print('SVM: оптимальный С =', optimalCForSVM, ', accuracy =', svmScore)

paint(optimalCForLogReg, optimalCForSVM)
