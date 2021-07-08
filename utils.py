import tensorflow as tf
import numpy as np
import random
from pyreadr import read_r


def set_all_seeds(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)


TEPDATASETTRAIN = 0
TEPDATASETTEST = 1


class TEP_DataLoader:
    def __init__(self, path_faultfree_testing="data/tep/TEP_FaultFree_Testing.RData", path_faultfree_training="data/tep/TEP_FaultFree_Training.RData", path_faulty_testing="data/tep/TEP_Faulty_Testing.RData", path_faulty_training="data/tep/TEP_Faulty_Training.RData"):
        self.path_faultfree_testing = path_faultfree_testing
        self.path_faultfree_training = path_faultfree_training
        self.path_faulty_testing = path_faulty_testing
        self.path_faulty_training = path_faulty_training

        self.testing_loaded = False
        self.training_loaded = False

    def load_testing_data(self):
        if not self.testing_loaded:
            raw_faultfree_testing = read_r(self.path_faultfree_testing)
            self.faultfree_testing = raw_faultfree_testing["fault_free_testing"]
            raw_faulty_testing = read_r(self.path_faulty_testing)
            self.faulty_testing = raw_faulty_testing["faulty_testing"]

        self.testing_loaded = True

    def load_training_data(self):
        if not self.training_loaded:
            raw_faultfree_training = read_r(self.path_faultfree_training)
            self.faultfree_training = raw_faultfree_training["fault_free_training"]
            raw_faulty_training = read_r(self.path_faulty_training)
            self.faulty_training = raw_faulty_training["faulty_training"]

        self.training_loaded = True

    def get_continuous_dataset(self, length=False, target_value="xmeas_35", dataset=TEPDATASETTRAIN, fault=False, input_features=["xmeas_1", "xmeas_2", "xmeas_10", "xmeas_11", "xmeas_14", "xmeas_16", "xmeas_18", "xmeas_20", "xmeas_25", "xmeas_33", "xmeas_31"]):
        if dataset == TEPDATASETTEST and not self.testing_loaded:
            raise Exception("Test Set not loaded!")
        if dataset == TEPDATASETTRAIN and not self.training_loaded:
            raise Exception("Training Set not loaded!")

        if dataset == TEPDATASETTRAIN and fault == False:
            data = self.faultfree_training
        elif dataset == TEPDATASETTRAIN and fault != False:
            data = self.faulty_training
        elif dataset == TEPDATASETTEST and fault == False:
            data = self.faultfree_testing
        elif dataset == TEPDATASETTEST and fault != False:
            data = self.faulty_testing
        else:
            raise Exception("Parameter combination not understood!")

        if fault != False:
            data = data[data["faultNumber"] == fault]

        length_index = length if length != False else len(data)
        data = data[:length_index]
        y = data[target_value].to_numpy()
        X = data[input_features].to_numpy()
        return X, y

    def get_runs_dataset(self, length=1, target_value="xmeas_35", dataset=TEPDATASETTRAIN, fault=False, input_features=["xmeas_1", "xmeas_2", "xmeas_10", "xmeas_11", "xmeas_14", "xmeas_16", "xmeas_18", "xmeas_20", "xmeas_25", "xmeas_33", "xmeas_31"]):
        if dataset == TEPDATASETTEST and not self.testing_loaded:
            raise Exception("Test Set not loaded!")
        if dataset == TEPDATASETTRAIN and not self.training_loaded:
            raise Exception("Training Set not loaded!")

        if dataset == TEPDATASETTRAIN and fault == False:
            data = self.faultfree_training
        elif dataset == TEPDATASETTRAIN and fault != False:
            data = self.faulty_training
        elif dataset == TEPDATASETTEST and fault == False:
            data = self.faultfree_testing
        elif dataset == TEPDATASETTEST and fault != False:
            data = self.faulty_testing
        else:
            raise Exception("Parameter combination not understood!")

        if fault != False:
            data = data[data["faultNumber"] == fault]

        runs_X = []
        runs_y = []
        for i in range(length):
            simulation_run = i + 1
            run_data = data[data["simulationRun"] == simulation_run]
            runs_y.append(run_data[target_value].to_numpy())
            runs_X.append(run_data[input_features].to_numpy())

        np_runs_y = np.stack(runs_y, axis=0)
        np_runs_X = np.stack(runs_X, axis=0)
        return np_runs_X, np_runs_y


if __name__ == "__main__":
    dataloader = TEP_DataLoader()
    dataloader.load_training_data()

    runs_X, runs_y = dataloader.get_runs_dataset(length=3, fault=5)
    print(runs_X.shape)
    print(runs_y.shape)
