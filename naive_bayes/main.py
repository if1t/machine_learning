import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def load_numpy_dataset(path, separator):
    return pd.read_csv(path, sep=separator).to_numpy()


def get_X_y(dataset):
    _X = np.array([dataset[i][:-1] for i in range(len(dataset))])
    _y = np.array([dataset[i][-1] for i in range(len(dataset))])
    return _X, _y


def get_M(_X, _y):
    n = len(set(y))
    return [np.mean(_X[_y == i], axis=0) for i in range(n)]


def get_D(_X, _y):
    n = len(set(y))
    return [np.var(_X[_y == i], axis=0) for i in range(n)]


def a_laplace(x):
    n = len(set(y))
    return np.argmax([(np.log10(priori_variance(yi)) + L(x, yi, laplace)) for yi in range(n)])


def laplace(x, mean, disp):
    stdev = np.sqrt(disp)
    exponent = np.exp(- np.abs(x - mean) / stdev)
    return 1 / (2*stdev) * exponent


def a_gauss(x):
    n = len(set(y))
    return np.argmax([(np.log10(priori_variance(yi)) + L(x, yi, gauss)) for yi in range(n)])


def gauss(x, mean, disp):
    stdev = np.sqrt(disp)
    exponent = np.exp(-(np.power(x - mean, 2) / (2 * np.power(stdev, 2))))
    return (1 / (np.sqrt(2 * np.pi) * stdev)) * exponent


def L(x, cls, func):
    return np.sum(np.log10([func(x[i], M[cls], D[cls]) for i in range(len(x))]))


def priori_variance(cls):
    count_obj_cls = len([X_train[i] for i in range(len(X_train)) if y_train[i] == cls])
    return count_obj_cls / len(X_train)


def get_predicted(_X, classifier):
    return [classifier(_X[i]) for i in range(len(_X))]


def score(_X, _y, classifier):
    sum_predict = 0
    for i in range(len(_X)):
        predict = classifier(_X[i])
        if predict == _y[i]:
            sum_predict += 1

    return sum_predict / len(_X)


def paint():
    figure, axis = plt.subplots(2, 2)
    figure.set_figheight(9)
    figure.set_figwidth(17)

    paintHist(axis[0, 0], X[:, 0], 'Дисп. вейвлет-преобраз. изображ.')
    paintHist(axis[0, 1], X[:, 1], 'Ассим. вейвлет-преобраз. изображ.')
    paintHist(axis[1, 0], X[:, 2], 'Эксц. вейвлет-преобраз. изображ.')
    paintHist(axis[1, 1], X[:, 3], 'Энтропия изображ.')
    plt.show()


def paintHist(ax, _X, title):
    ax.hist(_X, bins=30)
    ax.set_title(title, size=14)
    ax.set_xlabel('значение', size=9)
    ax.set_ylabel('количество', size=9)


data = load_numpy_dataset('dataset/banknote.csv', ',')
X, y = get_X_y(data)

# Деление выборки на обучающую и контрольную
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Вычисление матожиданий и дисперсий для всех признаков объектов разных классов
M = get_M(X_train, y_train)
D = get_D(X_train, y_train)

score_g = score(X_test, y_test, a_gauss)
score_l = score(X_test, y_test, a_laplace)

print('Наивный Байес с допущением, что признаки подчиняются распределению Гаусса:', score_g)
print('Наивный Байес с допущением, что признаки подчиняются распределению Лапласа:', score_l)

paint()
