from src import utils
import numpy as np
from enum import Enum
import math
from src.layer import Layer


class EarlyStopType(Enum):
    """Type of early stop function during training."""
    LOSS = 1
    ACCURACY = 2

    def __str__(self):
        """Return a string representation of the enum value."""
        return self.name.lower().capitalize()


class TrainBatchReport:
    """Container class for train batch output data."""
    def __init__(self, loss: float, acc: float):
        """TrainBatchReport Constructor

        :param loss: Mean loss of the batch
        :type loss: float
        :param acc: Mean accuracy of the batch
        :type acc: float
        """
        self.loss = loss
        self.acc = acc


class TrainReport:
    """Container class for the outputs of the training."""
    def __init__(self):
        self.train_loss = []
        self.train_acc = []
        self.val_loss = []
        self.val_acc = []

    def add_entry(self, train_loss: float, train_acc: float, val_loss: float, val_acc: float):
        """Add batch results to the training report.

        :param train_loss: Train losses for every epoch
        :type train_loss: float
        :param train_acc: Train accuracies for every epoch
        :type train_acc: float
        :param val_loss: Validation loss for every epoch
        :type val_loss: float
        :param val_acc: Validation accuracies for every epoch
        :type val_acc: float
        """
        self.train_loss.append(train_loss)
        self.train_acc.append(train_acc)
        self.val_loss.append(val_loss)
        self.val_acc.append(val_acc)


class TrainConfig:
    def __init__(self, n_epochs: int = 10, batch_size: int = 32,
                 learning_rate: float = 0.001, early_stop: int = None,
                 early_stop_type: EarlyStopType = EarlyStopType.LOSS,
                 loss_function: utils.LossFunction = utils.SoftmaxCrossentropy()):
        """

        :param n_epochs: Number of epochs for training
        :type n_epochs: int
        :param batch_size: Batch size for training
        :type batch_size: int
        :param learning_rate: Learning rate for training
        :type learning_rate: float
        :param early_stop: Early stop epochs
        :type early_stop: int
        :param early_stop_type: Type of early stop
        :type early_stop_type: EarlyStopType
        :param loss_function: Loss function to use
        :type loss_function: utils.LossFunction
        """
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.early_stop_type = early_stop_type
        self.loss_function = loss_function


class Model(object):
    def __init__(self):
        self.layers = []

    def add_layer(self, layer: Layer):
        """Append layer to the model.

        Append object of any class that derives from the base class "Layer".

        :param layer: Layer for the model
        :type layer: Layer
        """
        self.layers.append(layer)

    def _add_layers(self, layers: list[Layer]):
        """Set layers value.

        Change the layers to the given input. Used to restore the model to the previous checkpoint

        :param layers: List of defined layers
        :type layers: list[Layer]
        """
        self.layers = layers

    def _forward(self, x: np.ndarray) -> list[np.ndarray]:
        """Forward method for the model.

        Returns activations for all the layers inside the model, including the input data.

        :param x: Model input data
        :type x: np.ndarray
        :return: List of activations of all layers
        :rtype: list[np.ndarray]
        """
        activations = []

        # Add input data as the first activation
        layer_input = x
        for layer in self.layers:
            layer_input = layer.forward(layer_input)
            activations.append(layer_input)

        return activations

    def _train_batch(self, x: np.ndarray, y_target: np.ndarray,
                     loss_function: utils.LossFunction, learning_rate: float) -> TrainBatchReport:
        """Train model on a batch of data.

        This method performs a forward pass to compute layer activations, calculates
        the loss and its gradient, and then executes backpropagation to update
        the weights of the neural network.

        :param x: Input images
        :type x: np.ndarray
        :param y_target: Input Labels
        :type y_target: np.ndarray
        :param loss_function: Loss function for training
        :type loss_function: utils.LossFunction
        :param learning_rate: Train learning rate
        :type learning_rate: float
        :return: Training report for every epoch
        :rtype: TrainBatchReport
        """
        # Forward pass to compute layer activations
        layer_acts = self._forward(x)
        layer_inputs = layer_acts.copy()
        layer_inputs.insert(0, x)

        # Get predicted output from the last layer
        y_pred = layer_inputs.pop()

        # Compute loss and its gradient
        loss = loss_function.loss(y_pred, y_target)
        loss_grad = loss_function.derivative(y_pred, y_target)
        is_correct_guess = utils.see_if_guess_is_correct(y_pred, y_target)

        # Backpropagation to update weights
        for i in reversed(range(len(self.layers))):
            loss_grad = self.layers[i].backward(layer_inputs[i], loss_grad, learning_rate)

        # Create a training report
        report = TrainBatchReport(np.mean(loss), np.sum(is_correct_guess)/len(is_correct_guess))
        return report

    def reset(self):
        """Reset the state of the neural network.

        This method resets the internal state of the neural network,
        used before starting a new training session.
        """
        for i in range(len(self.layers)):
            self.layers[i].reset()

    def train(self, x_train: np.ndarray, y_train: np.ndarray,
              x_val: np.ndarray, y_val: np.ndarray, *,
              config: TrainConfig = TrainConfig()) -> TrainReport:
        """Train the neural network.

        This method trains the neural network using the provided training and
        validation data. It supports configurations for specifying the number
        of epochs, batch size, learning rate, and early stopping criteria.

        :param x_train: Training input data.
        :type x_train: np.ndarray
        :param y_train: Training target labels.
        :type y_train: np.ndarray
        :param x_val: Validation input data.
        :type x_val: np.ndarray
        :param y_val: Validation target labels.
        :type y_val: np.ndarray
        :param config: Training configuration (optional).
        :type config: TrainConfig
        :return: Training report.
        :rtype: TrainReport
        """

        # Extract configuration parameters
        n_epochs = config.n_epochs
        batch_size = config.batch_size
        early_stop = config.early_stop
        loss_function = config.loss_function
        early_stop_type = config.early_stop_type
        learning_rate = config.learning_rate

        # Display training configuration
        print(f"\nStarting training with config:\n"
              f"n_epochs = {n_epochs}, batch_size = {batch_size}, learning_rate = {learning_rate},\n "
              f"early_stop = {early_stop}, early_stop_type = {early_stop_type},\n"
              f"loss_function = {loss_function}.")

        train_report = TrainReport()

        if early_stop is not None:
            assert early_stop > 0
            early_stop_count = 0
            if early_stop_type == EarlyStopType.ACCURACY:
                highest_val_acc = 0
            else:
                lowest_val_loss = math.inf

        # Backup layers for early stopping
        layers_backup = self.layers

        # Training loop over epochs
        for epoch in range(n_epochs):
            epoch_loss = []
            epoch_acc = []
            for i in range(0, x_train.shape[0], batch_size):
                x_batch = np.array([x.flatten() for x in x_train[i:i + batch_size]])
                y_batch = np.array([y for y in y_train[i:i + batch_size]])

                # Train a batch and get the output results
                batch_report = self._train_batch(x_batch, y_batch, loss_function, learning_rate)
                epoch_loss.append(batch_report.loss)
                epoch_acc.append(batch_report.acc)

            # Compute average training loss and accuracy for the epoch
            train_acc = sum(epoch_acc) / len(epoch_acc)
            train_loss = sum(epoch_loss) / len(epoch_loss)

            # Evaluate validation set
            val_pred = self.predict(x_val)
            val_loss = np.mean(loss_function.loss(val_pred, y_val))
            val_acc = np.sum(utils.see_if_guess_is_correct(val_pred, y_val)) / val_pred.shape[0]

            # Add entry to the training report
            train_report.add_entry(train_loss, train_acc, val_loss, val_acc)

            # Print training progress
            print(f"Epoch: {epoch + 1}, "
                  f"Train accuracy: {train_report.train_acc[-1]}, "
                  f"Train loss: {train_report.train_loss[-1]}, "
                  f"Val accuracy: {train_report.val_acc[-1]}, "
                  f"Val loss: {train_report.val_loss[-1]}")

            # Early stopping logic.
            # If early stop type is "ACCURACY", then check for higher validation accuracy.
            # Else check for lowest validation loss.
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

                # Check if early stopping criteria met
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

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Generate predictions for the given input.

        :param x: Input data
        :type x: np.ndarray
        :return: Model output
        :rtype: np.ndarray
        """
        logits = self._forward(x)[-1]
        return logits
