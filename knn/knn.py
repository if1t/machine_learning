import random
import numpy as np

from matplotlib import pyplot as plt
from sklearn.datasets import load_iris


# functions
def get_train_control(train_percent):  # функция для получения тестовой и контрольной выборки
    train_data = []
    control_data = []

    for i in range(len(data)):  # проход по всей выборке
        if random.random() > train_percent:  # условие для распределения рандомным образом элементов полной выборки
            control_data.append([data[i][0], data[i][1]])  # добавление i-го объекта в контрольную выборку
        else:
            train_data.append([data[i][0], data[i][1]])  # добавление i-го объекта в обучающую выборку

    return train_data, control_data


def kernel(r):  # функция Епанчикова
    K = 0
    if np.fabs(r) <= 1:
        K = 3 / 4 * (1 - r * r)

    return K


def distance(x, xi):  # вычисление расстояния в евклидовом n-мерном пространстве
    return np.sqrt((np.sum((x - xi) ** 2)))


def get_accuracy(test_result, control_data):  # вычисление процента верно классифицируемых объектов
    accuracy = sum([int(test_result[i] == control_data[i][1]) for i in range(len(control_data))]) / float(
        len(control_data))
    print('Accuracy:', np.round(accuracy, 2) * 100, '%')


def predict(xi, data, k):  # функция для определения метки класса для i-го объекта выборки
    # вычисляем расстояние до остальных объектов из обучающей выборки
    dist = [[distance(xi[0], data[i][0]), data[i][1]] for i in range(len(data))]
    # вычисляем количество ближайших k точек для объекта
    stat = [0 for i in range(3)]
    # d - объект, который содержит в индексе 0 - расстояние до обрабатываемого объекта, 1 - класс объекта до которого
    # это расстояние вычисленно
    h = sorted(dist)[k][0]  # вычисляем h динамически, чтобы для каждого элемента входило ровно k ближайших соседей

    for p in sorted(dist)[0:k]:  # среди к ближайших объектов
        #stat[p[1]] += 1  # вычисление без использования модификации с парзеновскими окнами
        stat[p[1]] += kernel(p[0] / h)  # в массиве stat в ячейке с номером класса для расстояния p прибавляем вес,
    # вес вычисляется через ядро Епанчикова, как частное от расстояния i-го объекта к текущему k ближайшему
    # соседу деленное на динамически вычисляемый шаг парзеновского окна

    metka = stat.index(max(stat))

    return metka  # возвращаем метку класса к которой объект максимально близок относительно остальных классов


def LOO(data, k_start):  # функция для поиска оптимального минимального k для данной выборки
    mn = len(data)
    optimal = 150
    LOO_logs = []  # массив для сохранения количества ошибок для k в интервале [1,k_start)
    for k in range(1, k_start):
        sum_defect = 0  # количество ошибок для текущего k
        for i in range(len(data) - 1):  # проход по выборке
            dataForSplit = data.copy()  # копируем основную выборку в новый массив
            xi = dataForSplit.pop(i)    # из которого будем удалять i-тый элемент на каждой итерации
            if predict(xi, dataForSplit, k) != xi[1]:  # если предсказанных класс для удаленного элемента отличается от правильного
                sum_defect += 1  # инкрементируем количество ошибок
        LOO_logs.append(sum_defect)  # в конце цикла записываем количество ошибок для текущего k в массив
        print('количество ошибок при k =', k, 'равно:', sum_defect)
        if sum_defect < mn:  # если количество ошибок меньше текущего минимума количества ошибок
            mn = sum_defect  # запоминаем новый минимум
            optimal = k  # запоминаем новый k параметр
    k = optimal  # в конце внешнего цикла мы знаем оптимальный параметр k
    print('optimal k =', k)

    return LOO_logs, k  # возвращаем логи с количеством ошибок для каждого k из интервала, а также оптимальное значение


def classify_knn():  # функция для классифицирования объектов из
    # контрольной выборки относительно обучающей
    test_result = []
    for xi in control_data:
        test_result.append(predict(xi, train_data, k))

    get_accuracy(test_result, control_data)


def show_LOO():  # функция для показа графика результатов поиска оптимального значения
    plt.title("LOO for found optimal k-parameter")
    plt.ylabel("Количество ошибок")
    plt.xlabel("Значение k")
    plt.plot(LOO)
    plt.plot(k, LOO[k], marker="o")
    plt.show()


# /functions
# main
iris = load_iris()

X = iris.data  # Множество объектов
Y = iris.target  # Множество отетов

data = [[X[i], Y[i]] for i in range(len(X))]  # объединение множества объектов и их классов соответственно

LOO, k = LOO(data, 149)  # вычисление оптимального параметра k - количество ближайших соседей

train_data, control_data = get_train_control(0.66)  # разделение выборки на обучающую и контрольную

classify_knn()  # классификация для контрольной выборки с оптимальным найденым k с
# итоговым вычислением accuracy

show_LOO()  # график получившихся результатов
