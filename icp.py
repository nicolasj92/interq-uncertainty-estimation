import numpy as np
from sklearn.neighbors import NearestNeighbors
from scipy import stats


class ICPBase:
    def __init__(self):
        self.calibration_nc_scores = None
        self.trained = False

    def calibrate(self, X):
        self.calibration_nc_scores = self.nc_score(X)

    def predict(self, X):
        mean, _ = self.predict_and_score(X)
        return mean

    def nc_score(self, X):
        _, std = self.predict_and_score(X)
        return std

    def confs(self, X):
        if self.calibration_nc_scores is not None:
            nc_scores = self.nc_score(X)
            p = np.array([(100. - stats.percentileofscore(self.calibration_nc_scores,
                         nc_score)) / 100. for nc_score in nc_scores.tolist()])
        else:
            p = None

        return p

    def fit(self):
        raise NotImplementedError()

    def predict_and_score(self):
        raise NotImplementedError()

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()


class ICPNearestNeighborDist(ICPBase):
    def __init__(self):
        self.calibration_nc_scores = None
        self.model = NearestNeighbors(n_neighbors=1)

    def fit(self, X):
        self.model.fit(X)

    def nc_score(self, X):
        dists = self.model.kneighbors(
            X, n_neighbors=1, return_distance=True)[0][:, 0]
        return dists


class ICPMahalanobisDist(ICPBase):
    def __init__(self):
        self.calibration_c_scores = None
        self.inv_cov = None
        self.mu = None

    def fit(self, X):
        cov_matrix = np.cov(X, rowvar=False)
        self.inv_cov = np.linalg.inv(cov_matrix)
        self.mu = np.mean(X, axis=0)

    def nc_score(self, X):
        delta = X - self.mu
        D = np.sqrt(np.einsum('nj,jk,nk->n', delta, self.inv_cov, delta))
        return D


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from utils import TEP_DataLoader

    dataloader = TEP_DataLoader()
    dataloader.load_training_data()
    X, y = dataloader.get_continuous_dataset(
        length=150000, target_value="xmeas_35")
    X_train, X_valtest, y_train, y_valtest = train_test_split(
        X, y, test_size=0.30, random_state=42)

    mahalanobis = ICPMahalanobisDist()
    mahalanobis.fit(X_train)
    mahalanobis.nc_score(X_valtest)
