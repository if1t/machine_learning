import random
import time

import matplotlib.pyplot as plt
import numpy as np

from sklearn.datasets import load_iris

# constants
COLORS = ('green', 'yellow', 'magenta', 'cyan', 'red', 'blue', 'bright_red', 'bright_blue', 'bright_white')


# /constants
# functions
def set_scatter(centers, cluster, num_cluster):  # функция для отрисовки результатов кластеризации
    plt.title('Кластеризация: ' + str(num_cluster))
    for i in range(k):  # отображение найденных кластеров
        obj = np.array(cluster[i]).T
        plt.scatter(obj[x], obj[y], s=20, color=COLORS[i])

    centers_x = [center[x] for center in centers]
    centers_y = [center[y] for center in centers]
    plt.scatter(centers_x, centers_y, s=200, color='black', marker='+')  # отображение найденных центров кластеров


def distance(xi, center):  # вычисление расстояния в евклидовом n-мерном пространстве
    return np.sqrt((np.sum((xi - center) ** 2)))


def initialization_centers():  # начальная инициализация центров кластеров
    # инициализируем центры кластеров случайными объектами из выборки
    return np.array(X[np.random.choice(len(X), k, replace=False)])


def expectation(xi):  # кластеризация с текущими центрами
    dist = [distance(xi, center) for center in centers]  # вычисление расстояний до каждого центра

    return np.argmin(dist)  # возвращаем индекс(номер центра кластеров) минимального расстояния до центра


def maximization():  # перерасчет координат центров
    # вычисляем среднее арифметическое из объектов относящихся к одной группе
    return [np.mean(xi, axis=0) for xi in cluster_objects]


def quality(cluster):  # оценка качества кластеризации на основе критерия минимизации внутрикластерного расстояния
    return np.sum(
        [1 / len(cluster[i]) * np.sum([distance(cluster[i][j], centers[i]) for j in range(len(cluster[i]))]) for i in
         range(k)])


# /functions
# main
iris = load_iris()
X = iris.data  # множество объектов без меток

# какие признаки выводить на графиках
x = 2
y = 3

k = int(input('количество кластеров: '))  # количество кластеров
m = int(input('количество кластеризаций: '))  # количество кластеризаций, так как алгоритм k-means зависит от
# начальной инициализации центров кластеров

plt.ion()

clusters = [[[] for i in range(k)] for j in range(m)]  # трехмерный массив для хранения всех кластеров
arr_centers = [[] for i in range(m)]  # массив для хранения центров i-ых кластеров
quality_answers = [0 for i in range(m)]  # массив для хранения оценкци критерия внутригруппового расстояния

# формируем m кластеризаций
for num_cluster in range(m):
    centers = initialization_centers()  # инициализация центров кластеров

    while True:
        cluster_objects = [[] for i in range(k)]

        last_centers = centers.copy()
        # алгоритм Ллойда
        for xi in X:  # проход по всей выборке объектов
            number_center = expectation(xi)  # вычисление метки для i-го объекта
            cluster_objects[number_center].append(xi)  # сохранение объекта в вычисленном кластере

        centers = maximization()  # перерасчет центров
        # /алгоритм Ллойда
        # отображение найденных кластеров
        plt.clf()
        set_scatter(centers, cluster_objects, num_cluster)
        plt.draw()
        plt.gcf().canvas.flush_events()
        time.sleep(0.3)
        # /отображение найденных кластеров
        flag = True
        for i in range(k):
            for j in range(len(centers[i])):
                if centers[i][j] != last_centers[i][j]:
                    flag = False

        if flag:
            print('Кластеризация', num_cluster, 'полностью сформирована')
            # вычисляем оценку критерия внутригруппового расстояния
            quality_answers[num_cluster] = quality(cluster_objects)  # сохранения оценки для сформированного кластера
            arr_centers[num_cluster] = centers  # сохраняем сформированные центры в общий массив центров
            clusters[num_cluster] = cluster_objects  # сохраняем готовый кластер в общий массив кластеров
            break
plt.ioff()

min_index = np.argmin(quality_answers)  # вычисляем индекс кластера у которого внутригрупповое расстояние наименьшее
centers = arr_centers[min_index]  # получаем центры искомого кластера
cluster_objects = clusters[min_index]  # получаем размеченные объекты искомого кластера

print('Кластеризация с минимальным средним внутригрупповым расстоянием сформирована под индексом', min_index)
print(quality_answers)
print('Кластеризация', min_index, 'с минимальным средним внутрикластерным расстоянием:', quality_answers[min_index])

plt.gcf().clear()
set_scatter(centers, cluster_objects, min_index)
plt.show()  # выводим на график
