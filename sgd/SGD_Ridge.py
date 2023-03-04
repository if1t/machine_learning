import numpy as np


class SGD_Ridge(object):
    def __init__(self, X, y, lam):
        self.X = X
        self.y = y
        self.lam = lam

    def SGD(self, w, tau):
        t = tau / 1000
        h = 0.02
        # w = np.random.random(len(self.X[0]))

        norma_w = sum([w[i] ** 2 for i in range(len(w))])
        reg = (t * norma_w ** 2) / 2

        Q_now = self.Q(self.X, self.y, w) + reg

        i = 0
        k = 100
        while i < k:
            rand_i = np.random.randint(0, len(self.X))
            xi = self.X[rand_i]
            yi = self.y[rand_i]
            a = self.predict(xi, w)

            eps = self.L(a, yi)
            w = self.W(xi, yi, w, h, t)

            Q_pred = Q_now
            Q_now = self.lam*eps + (1 - self.lam)*Q_pred

            i = i + 1 if abs(Q_now - Q_pred) < 0.0001 else 0
        return w

    def predict(self, x, w):
        return np.round(np.dot(x, w), 5)

    def L(self, a, y):
        return (a - y)**2

    def Q(self, X, y, w):
        sum_L = sum([self.L(self.predict(X[i], w), y[i]) for i in range(len(X))])
        return sum_L / len(self.X)

    def W(self, x, y, w, h, t):
        return w * (1 - h * t) - h * self.delta_L(x, y, w, t)

    def delta_L(self, x, y, w, t):
        a = self.predict(x, w)
        return 2*(a - y)*x + t * w
