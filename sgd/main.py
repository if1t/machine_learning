import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

from SGD_Ridge import SGD_Ridge


def add_fictive_column(X):
    fictive = np.array([-1 for _ in range(len(X))])
    return np.insert(X, len(X[0]), fictive, axis=1)


figure, axis = plt.subplots(2, 2)
figure.set_figheight(9)
figure.set_figwidth(16)
# загрузка датасета #
data = pd.read_csv("datasets/synchronous_machine.csv", sep=";").to_numpy()

X = np.array([data[i][:-1] for i in range(len(data))])
y = np.array([data[i][-1] for i in range(len(data))])

# Добавление фиктивного признака #
X = add_fictive_column(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=63021)
lam = 0.0007

sgd = SGD_Ridge(X_train, y_train, lam)

arr_t = [np.round(1*np.round(i*0.1, 1), 2) for i in range(1, 21)]
print(arr_t)

# Вычисление оптимального значения регуляризатора #
global_min_q = 100
global_optimal_w = []
for i in range(3):
    # Инициализация начального веса #
    w = np.random.random(len(X[0]))
    print('Начальный w:', w)

    # Параметры для отрисовки #
    ax = axis[0, 1]

    # Начало алгоритма #
    arr_w = []
    arr_q = []

    local_w = []
    local_min_q = 100
    local_t = -1

    if i == 1:
        ax = axis[1, 0]
    if i == 2:
        ax = axis[1, 1]

    for j in range(len(arr_t)):
        w = sgd.SGD(w, arr_t[j])
        q = sgd.Q(X_test, y_test, w)
        print('При t =', arr_t[j], 'функционал качества =', q)
        arr_w.append(w)
        arr_q.append(q)

        if q < local_min_q:
            local_min_q = q
            local_w = w
            local_t = arr_t[j]

    ax.plot(arr_t, arr_q)
    ax.set_title(str(i+1) + ' график.')
    ax.set(xlabel='Параметр t', ylabel='Функционал качества Q')
    print('Оптимальный w:', local_w, ', при t =', local_t, 'функционал качества:', local_min_q)

    if local_min_q < global_min_q:
        global_optimal_w = local_w

# Применение линейного классификатора #
control = [sgd.predict(X_test[i], global_optimal_w) for i in range(len(X_test))]

# Отрисовка результатов #
x = np.linspace(0, len(X_test), len(X_test))
axis[0, 0].scatter(y_test, control, color='b', label='предсказанный ответ')
axis[0, 0].set_title('Соотношение ответов')
axis[0, 0].set(xlabel='Истинный', ylabel='Предсказанный')

plt.show()
