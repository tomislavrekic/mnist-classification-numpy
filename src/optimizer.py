from itertools import permutations, product

import numpy as np
from model import Model, TrainConfig, EarlyStopType, TrainReport
from utils import LossFunction


class HyParamOptimConfig:
    """A class for configuring the HyperParamOptimizer"""
    def __init__(self, *, n_epochs: tuple[int, ...],
                 batch_size: tuple[int, ...],
                 learning_rate: tuple[float, ...],
                 early_stop: tuple[int, ...],
                 early_stop_type: tuple[EarlyStopType, ...],
                 loss_function: tuple[LossFunction, ...]):
        """HyParamOptimConfig constructor

        All inputs must be tuples. A single value of x can be expressed as a tuple as:
        tuple_x = (x,)

        :param n_epochs: Epochs to test
        :type n_epochs: tuple[int, ...]
        :param batch_size: Batch sizes to test
        :type batch_size: tuple[int, ...]
        :param learning_rate: Learning rates to test
        :type learning_rate: tuple[float, ...]
        :param early_stop: early stop n's to test
        :type early_stop: tuple[int, ...]
        :param early_stop_type: early stop types to test
        :type early_stop_type: tuple[EarlyStopType, ...]
        :param loss_function: loss functions to test
        :type loss_function: tuple[LossFunction, ...]
        """

        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop = early_stop
        self.early_stop_type = early_stop_type
        self.loss_function = loss_function

        arg_list = [self.n_epochs,
                    self.batch_size,
                    self.learning_rate,
                    self.early_stop,
                    self.early_stop_type,
                    self.loss_function]

        # Create all possible permutation of all tuples
        permuts = list(product(*[permutations(arg) for arg in arg_list]))

        # Choose the first element of every tuple permutation
        combinations = []
        for i in range(len(permuts)):
            combinations.append(tuple([permut[0] for permut in permuts[i]]))

        # Convert to set and back to remove duplicate values
        self.optim_cases = list(set(combinations))


class HyperParamOptimizer:
    def __init__(self, optim_config: HyParamOptimConfig):
        """HyperParamOptimizer constructor.

        :param optim_config: Config for Hyper-parameter Optimizer
        :type optim_config: HyParamOptimConfig
        """
        self.optim_config = optim_config

    def optimize(self, model: Model, x_train: np.ndarray, y_train: np.ndarray,
                 x_val: np.ndarray, y_val: np.ndarray) -> tuple[list[TrainConfig], list[TrainReport]]:
        """Start hyperparameter optimization.

        Starts training the given model using every config provided. Saves the training results and
        return them together with the configs.

        :param model: model to optimize on
        :type model: Model
        :param x_train: Train images
        :type x_train: np.ndarray
        :param y_train: Train labels
        :type y_train: np.ndarray
        :param x_val: Validation images
        :type x_val: np.ndarray
        :param y_val: Validation labels
        :type y_val: np.ndarray
        :return: list of train configs and train reports
        :rtype: tuple[list[TrainConfig], list[TrainReport]]
        """
        train_configs = []
        for i in range(len(self.optim_config.optim_cases)):
            args = [arg for arg in self.optim_config.optim_cases[i]]
            train_configs.append(TrainConfig(*args))

        train_reports = []
        for config in train_configs:
            train_reports.append(model.train(x_train, y_train, x_val, y_val, config=config))
            model.reset()
        return train_configs, train_reports
