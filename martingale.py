import numpy as np


class PowerMartingale:
    def __init__(self, model, alarm_threshold=100, e=0.3):
        self.e = e
        self.eps = 1e-5
        self.min_p = 0.001
        self.alarm_threshold = alarm_threshold
        self.model = model

    def score(self, X):
        P = self.model.confs(X)

        M = np.zeros_like(P)
        S = np.zeros_like(P)

        first_alarm_step = None

        last_m = 1
        for i, p in enumerate(P):
            if p < self.min_p:
                p = self.min_p
            M[i] = last_m * self.e * (p ** (self.e - 1))
            if i != 0:
                S[i] = np.max([0.0, M[i-1] + S[i-1]])
            else:
                S[i] = 0.

            if S[i] >= self.alarm_threshold:
                if not first_alarm_step:
                    first_alarm_step = i
                S[i] = 0.
                M[i] = 1.

            last_m = M[i]
        return M, S, first_alarm_step
