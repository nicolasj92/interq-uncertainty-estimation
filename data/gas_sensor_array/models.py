import numpy as np

from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from scipy.stats import entropy

from keras import Model
from keras.layers import Dense, Dropout, Input
from keras.utils import np_utils


def build_nn_model(num_classes, input_features, mc_dropout=False, normal_dropout=False, p_dropout=0.2):
    inputs = Input(shape=input_features)
    x = Dense(input_features, activation='relu')(inputs)
    x = Dense(32, activation='relu')(x)
    if mc_dropout:
        x = Dropout(rate=p_dropout)(x, training=True)
    elif normal_dropout:
        x = Dropout(rate=p_dropout)(x)

    x = Dense(16, activation='relu')(x)
    if mc_dropout:
        x = Dropout(rate=p_dropout)(x, training=True)
    elif normal_dropout:
        x = Dropout(rate=p_dropout)(x)

    x = Dense(8, activation='relu')(x)

    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


class Simple_NN:
    def __init__(self, input_features=128, num_classes=6, dropout=False, epochs=50):
        self.nn_model = build_nn_model(
            num_classes=num_classes, input_features=input_features, normal_dropout=dropout)
        self.num_classes = num_classes
        self.train_epochs = epochs
        self.scaler = StandardScaler()

    def fit(self, X, y, batch_size=8, validation_data=None):
        X = self.scaler.fit_transform(X)
        dummy_y = np_utils.to_categorical(y, num_classes=self.num_classes)
        if validation_data is not None:
            X_val = validation_data[0]
            y_val = np_utils.to_categorical(
                validation_data[1], num_classes=self.num_classes)
            self.nn_model.fit(X, dummy_y, batch_size=batch_size,
                              epochs=self.train_epochs, validation_data=(X_val, y_val), verbose=0)
        else:
            self.nn_model.fit(X, dummy_y, batch_size=batch_size,
                              epochs=self.train_epochs, verbose=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        return self.nn_model.predict(X)

    def predict_entropy(self, X):
        return entropy(self.predict_proba(X), axis=1)


class MCDropout_NN(Simple_NN):
    def __init__(self, num_passes=10, input_features=128, num_classes=6, epochs=50):
        self.nn_model = build_nn_model(
            num_classes=num_classes, input_features=input_features, mc_dropout=True)
        self.num_classes = num_classes
        self.scaler = StandardScaler()
        self.num_passes = num_passes
        self.train_epochs = epochs

    def fit(self, X, y, batch_size=8, validation_data=None):
        X = self.scaler.fit_transform(X)
        dummy_y = np_utils.to_categorical(y, num_classes=self.num_classes)
        if validation_data is not None:
            X_val = validation_data[0]
            y_val = np_utils.to_categorical(
                validation_data[1], num_classes=self.num_classes)
            self.nn_model.fit(X, dummy_y, batch_size=batch_size,
                              epochs=self.train_epochs, validation_data=(X_val, y_val), verbose=0)
        else:
            self.nn_model.fit(X, dummy_y, batch_size=batch_size,
                              epochs=self.train_epochs, verbose=0)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        X = self.scaler.transform(X)
        predictions = np.zeros((self.num_passes, X.shape[0], self.num_classes))
        for i in range(self.num_passes):
            predictions[i] = self.nn_model.predict(X)
        return np.mean(predictions, axis=0)

    def predict_entropy(self, X):
        return entropy(self.predict_proba(X), axis=1)


class Ensemble_NN:
    def __init__(self, num_nets=10, input_features=128, num_classes=6):
        self.models = []
        self.num_classes = num_classes
        for _ in range(num_nets):
            self.models.append(
                Simple_NN(num_classes=num_classes, input_features=input_features))

    def fit(self, X, y, validation_data=None):
        for net in self.models:
            shuffle_X, shuffle_y = shuffle(X, y)
            net.fit(shuffle_X, shuffle_y, validation_data=validation_data)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba(self, X):
        predictions = np.zeros(
            (len(self.models), X.shape[0], self.num_classes))
        for i, net in enumerate(self.models):
            predictions[i] = net.predict_proba(X)
        return np.mean(predictions, axis=0)

    def predict_entropy(self, X):
        return entropy(self.predict_proba(X), axis=1)


class RandomForestEntropy:
    def __init__(self, model):
        self.model = model

    def predict_entropy(self, X):
        all_probs = np.zeros((len(self.model.estimators_),
                             X.shape[0], self.model.n_classes_))
        for i, tree in enumerate(self.model.estimators_):
            probs = tree.predict_proba(X)
            all_probs[i] = probs

        mean_probs = np.mean(all_probs, axis=0)
        entropies = entropy(mean_probs, axis=1)
        return entropies


class GaussProcess:
    def __init__(self):
        self.model = GaussianProcessClassifier()
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_train = self.scaler.fit_transform(X)
        self.model.fit(X_train, y)

    def predict(self, X):
        X_test = self.scaler.transform(X)
        return self.model.predict(X_test)

    def predict_proba(self, X):
        X_test = self.scaler.transform(X)
        return self.model.predict_proba(X_test)


class KNearest:
    def __init__(self):
        self.model = KNeighborsClassifier()
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_train = self.scaler.fit_transform(X)
        self.model.fit(X_train, y)

    def predict(self, X):
        X_test = self.scaler.transform(X)
        return self.model.predict(X_test)

    def predict_proba(self, X):
        X_test = self.scaler.transform(X)
        return self.model.predict_proba(X_test)


class ScaledSVC:
    def __init__(self):
        self.model = SVC(probability=True)
        self.scaler = StandardScaler()

    def fit(self, X, y):
        X_train = self.scaler.fit_transform(X)
        self.model.fit(X_train, y)

    def predict(self, X):
        X_test = self.scaler.transform(X)
        return self.model.predict(X_test)

    def predict_proba(self, X):
        X_test = self.scaler.transform(X)
        return self.model.predict_proba(X_test)
