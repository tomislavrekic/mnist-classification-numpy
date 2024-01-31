from utils import SquaredError, SoftmaxCrossentropy, LossFunction, see_if_guess_is_correct
import numpy as np
from enum import Enum
import math


class EarlyStopType(Enum):
    LOSS = 1
    ACCURACY = 2


class TrainBatchReport:
    def __init__(self, loss, acc):
        self.loss = loss
        self.acc = acc


class TrainReport:
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def add_entry(self, train_loss, train_acc, val_loss, val_acc):
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)


class TrainConfig:
    def __init__(self, *, n_epochs=10, batch_size=32, learning_rate=0.001,
                 early_stop=None, early_stop_type=EarlyStopType.LOSS,
                 loss_function=SoftmaxCrossentropy()):
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.early_stop_type = early_stop_type
        self.loss_function = loss_function


class Model(object):
    def __init__(self):
        self.layers = []

    def add_layer(self, layer):
        self.layers.append(layer)

    def _add_layers(self, layers):
        self.layers = layers

    def _forward(self, x: np.ndarray) -> list:
        activations = []

        layer_input = x
        for layer in self.layers:
            layer_input = layer.forward(layer_input)
            activations.append(layer_input)

        return activations

    def _train_batch(self, x: np.ndarray, y_target: np.ndarray,
                     loss_function: LossFunction, learning_rate: float) -> TrainBatchReport:
        layer_acts = self._forward(x)
        layer_inputs = layer_acts.copy()
        layer_inputs.insert(0, x)

        y_pred = layer_inputs.pop()

        loss = loss_function.loss(y_pred, y_target)
        loss_grad = loss_function.derivative(y_pred, y_target)
        is_correct_guess = see_if_guess_is_correct(y_pred, y_target)

        for i in reversed(range(len(self.layers))):
            loss_grad = self.layers[i].backward(layer_inputs[i], loss_grad, learning_rate)

        report = TrainBatchReport(np.mean(loss), np.sum(is_correct_guess)/len(is_correct_guess))
        return report

    def train(self, x_train, y_train, x_val, y_val, *, config=TrainConfig()):
        n_epochs = config.n_epochs
        batch_size = config.batch_size
        early_stop = config.early_stop
        loss_function = config.loss_function
        early_stop_type = config.early_stop_type
        learning_rate = config.learning_rate

        train_report = TrainReport()

        if early_stop is not None:
            assert early_stop > 0
            early_stop_count = 0
            if early_stop_type == EarlyStopType.ACCURACY:
                highest_val_acc = 0
            else:
                lowest_val_loss = math.inf

        layers_backup = self.layers

        for epoch in range(n_epochs):
            epoch_loss = []
            epoch_acc = []
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = np.array([x.flatten() for x in x_train[i:i + batch_size]])
                y_batch = np.array([y for y in y_train[i:i + batch_size]])
                batch_report = self._train_batch(x_batch, y_batch, loss_function, learning_rate)
                epoch_loss.append(batch_report.loss)
                epoch_acc.append(batch_report.acc)

            train_acc = sum(epoch_acc) / len(epoch_acc)
            train_loss = sum(epoch_loss) / len(epoch_loss)

            val_pred = self.predict(x_val)
            val_loss = np.mean(loss_function.loss(val_pred, y_val))
            val_acc = np.sum(see_if_guess_is_correct(val_pred, y_val)) / val_pred.shape[0]

            train_report.add_entry(train_loss, train_acc, val_loss, val_acc)

            print(f"Epoch: {epoch + 1}, "
                  f"Train accuracy: {train_report.train_acc[-1]}, "
                  f"Train loss: {train_report.train_loss[-1]}, "
                  f"Val accuracy: {train_report.val_acc[-1]}, "
                  f"Val loss: {train_report.val_loss[-1]}")

            if early_stop is not None:
                if early_stop_type == EarlyStopType.ACCURACY:
                    if val_acc > highest_val_acc:
                        highest_val_acc = val_acc
                        layers_backup = self.layers
                        early_stop_count = 0
                    elif val_acc <= highest_val_acc:
                        early_stop_count += 1
                else:
                    if val_loss < lowest_val_loss:
                        lowest_val_loss = val_loss
                        layers_backup = self.layers
                        early_stop_count = 0
                    elif val_loss >= lowest_val_loss:
                        early_stop_count += 1

                if early_stop_count >= early_stop:
                    self._add_layers(layers_backup)
                    if early_stop_type == EarlyStopType.ACCURACY:
                        print(f"Early stop of {early_stop}, stopping training and returning model to state "
                              f"with validation accuracy of {val_acc}")
                    else:
                        print(f"Early stop of {early_stop}, stopping training and returning model to state "
                              f"with validation loss of {lowest_val_loss}")
                    break
        return train_report

    def predict(self, x):
        logits = self._forward(x)[-1]
        return logits
