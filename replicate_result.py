import tensorflow as tf
import pandas as pd
import numpy as np
import glob
import datetime
from sklearn.preprocessing import robust_scale
from tqdm.auto import tqdm
import argparse

from preprocessing import *
from AdvALSTM import *


class TestEvaluation(tf.keras.callbacks.Callback):
    def __init__(self, X_test, y_test):
        super(TestEvaluation, self).__init__()
        self.X_test, self.y_test = X_test, y_test
        
    def on_epoch_end(self, epoch, logs):
        X_test, y_test = self.X_test, self.y_test
        test_metrics = self.model.evaluate(self.X_test, self.y_test, verbose = 0)

        logs["test_loss"] = test_metrics[0]
        logs["test_acc"] = test_metrics[1]


def main():
    # Load preprocessed data
    with open('preprocessed_data.npz', 'rb') as f:
        X_train = np.load(f, allow_pickle=True)
        y_train = np.load(f, allow_pickle=True)
        X_validation = np.load(f, allow_pickle=True)
        y_validation = np.load(f, allow_pickle=True)
        X_test = np.load(f, allow_pickle=True)
        y_test = np.load(f, allow_pickle=True)

    # Create model
    model = AdvALSTM(
        units = 16, 
        epsilon = 0.1, 
        beta = 0.05, 
        learning_rate = 1E-2, 
        l2 = 0.001, 
        attention = True, 
        hinge = True,
        dropout = 0.0,
        adversarial_training = True,
        random_perturbations = False
    )

    model.fit(
        X_train,  y_train,
        validation_data = (X_validation, y_validation),
        epochs = 300,
        batch_size = 1024,
        callbacks=[TestEvaluation(X_test, y_test)]
    )

if __name__ == "__main__":
    main()