from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.layers import BatchNormalization, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import numpy as np
import random

from utils import set_all_seeds
from icp import ICPBase


def mse_mean(y_true, y_pred):
    mse = tf.math.squared_difference(y_true, y_pred[:, 0])
    return tf.reduce_mean(mse)


def nll_loss_regresion(y_true, y_pred):
    variance = tf.math.log(1. + tf.math.exp(y_pred[:, 1] + 1e-6))

    nll = (tf.math.log(variance) / 2) + \
        (((y_true - y_pred[:, 0]) ** 2) / (2 * variance))

    return tf.reduce_mean(nll)


def create_nn_model(input_dim, output_dim, dropout_inference=None,  loss='mse'):
    inputs = Input(input_dim)

    # defination of layers
    x = BatchNormalization()(inputs)
    # x = Dropout(hp.Float('init_dropout', 0.0, 0.1))(x)

    x = Dense(32)(x)
    x = BatchNormalization()(x)
    x = Lambda(tf.keras.activations.swish)(x)
    x = Dropout(0.2)(x, training=dropout_inference)

    x = Dense(output_dim, activation='relu')(x)

    # build the model
    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=Adam(0.03), loss=loss, metrics=[mse_mean])

    return model


class ICPRegressionMCDropout(ICPBase):
    def __init__(self, n_forward_passes, input_shape):
        self.n_passes = n_forward_passes
        self.model = create_nn_model(input_shape, 1, dropout_inference=True)

        super().__init__()

    def fit(self, X, y, batch_size=32, epochs=10):
        self.model.fit(X, y, epochs=epochs, batch_size=batch_size)
        self.trained = True

    def predict_and_score(self, X):
        mc_predictions = np.zeros((self.n_passes, len(X)))
        for i in range(self.n_passes):
            mc_predictions[i] = np.squeeze(self.model.predict(X))
        return mc_predictions.mean(axis=0), mc_predictions.std(axis=0)

    def save(self, file_prefix="./models/tep_mc_dropout_"):
        if not self.trained:
            raise Exception("Train model before saving!")

        self.model.save_weights(file_prefix + ".h5")

    def load(self, file_prefix="./models/tep_mc_dropout_"):
        self.model.load_weights(file_prefix + ".h5")
        self.trained = True


class ICPRegressionEnsemble(ICPBase):
    def __init__(self, n_members, input_shape):
        self.models = []

        for _ in range(n_members):
            model = create_nn_model(input_shape, 2, loss=nll_loss_regresion)
            self.models.append(model)

        super().__init__()

    def fit(self, X, y, batch_size=32, epochs=10):
        for model in self.models:
            set_all_seeds(random.randint(0, 100000))
            perm = np.random.permutation(len(X))
            model.fit(X[perm], y[perm], epochs=epochs, batch_size=batch_size)

        self.trained = True

    def predict_and_score(self, X):
        ensemble_pred_means = []
        ensemble_pred_vars = []
        for model in self.models:
            pred = model.predict(X)
            pred_means = pred[:, 0]
            pred_vars = pred[:, 1]
            ensemble_pred_means.append(pred_means)
            ensemble_pred_vars.append(pred_vars)

        ensemble_pred_means = np.stack(ensemble_pred_means, axis=0)
        ensemble_pred_vars = np.stack(ensemble_pred_vars, axis=0)

        mean = ensemble_pred_means.mean(axis=0)
        var = np.mean(ensemble_pred_vars +
                      (ensemble_pred_means ** 2), axis=0) - (mean ** 2)

        return mean, var

    def save(self, file_prefix="./models/tep_ensemble_"):
        if not self.trained:
            raise Exception("Train model before saving!")

        for i in range(len(self.models)):
            self.models[i].save_weights(file_prefix + str(i) + ".h5")

    def load(self, file_prefix="./models/tep_ensemble_"):
        for i in range(len(self.models)):
            self.models[i].load_weights(file_prefix + str(i) + ".h5")
        self.trained = True


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split
    from utils import TEP_DataLoader

    dataloader = TEP_DataLoader()
    dataloader.load_training_data()
    X, y = dataloader.get_continuous_dataset(
        length=150000, target_value="xmeas_31", input_features=["xmeas_1", "xmeas_2", "xmeas_10", "xmeas_11", "xmeas_14", "xmeas_16", "xmeas_18", "xmeas_20", "xmeas_25", "xmeas_33"], random_all=False)
    X_train, X_valtest, y_train, y_valtest = train_test_split(
        X, y, test_size=0.80, random_state=42)

    mc_dropout = ICPRegressionMCDropout(
        n_forward_passes=20, input_shape=X_train.shape[-1])
    mc_dropout.fit(X_train, y_train)
    mc_dropout.save()

    # ensemble = ICPRegressionEnsemble(
    #     n_members=10, input_shape=X_train.shape[-1])
    # ensemble.fit(X_train, y_train)
    # ensemble.save(file_prefix="./models/tep_ensemble_nll_")
    # ensemble.load(file_prefix='./models/tep_ensemble_nll_')
    # ensemble.calibrate(X_valtest)
