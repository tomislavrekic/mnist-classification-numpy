import numpy as np


class Initializer:
    """A base class for the numpy matrix initializers.

    Initializers are used to set starting values to the weights and biases of the network layers.
    They can also be used to initialize any 1D or 2D numpy array.
    Most of the initializers found here can also be found in the Keras documentation here:
    https://keras.io/api/layers/initializers/

    Any variables needed for a specific Initializer should be passed in the "__init__".
    Values are generated in the "generate" method which takes y and y dimensions
    as a parameter.
    The generated numpy array will be of dimensions [x, y].
    """

    def __init__(self):
        pass

    def generate(self, x: int, y: int) -> np.ndarray:
        """Virtual class to be overriden. Will raise NotImplementedError."""
        raise NotImplementedError()


class RandomNormal(Initializer):
    """Random normal initializer.

     Draws samples from a normal distribution for given parameters.
     """

    def __init__(self, *, mean: float = 0.0, stddev: float = 0.05):
        super().__init__()
        self.mean = mean
        self.stddev = stddev

    def generate(self, x: int, y: int) -> np.ndarray:
        """Return the initialized numpy array

        :param x: Size of first dimension of np.ndarray
        :type x: int
        :param y: Size of second dimension of np.ndarray
        :type y: int
        :return: Initialized np.ndarray
        :rtype: np.ndarray
        """
        return np.random.normal(loc=self.mean, scale=self.stddev, size=(x, y))


class RandomUniform(Initializer):
    """Random uniform initializer.

    Draws samples from a uniform distribution for given parameters.
    """

    def __init__(self, *, low: float = -0.1, high: float = 0.1):
        super().__init__()
        self.low = low
        self.high = high

    def generate(self, x: int, y: int) -> np.ndarray:
        """Return the initialized numpy array

        :param x: Size of first dimension of np.ndarray
        :type x: int
        :param y: Size of second dimension of np.ndarray
        :type y: int
        :return: Initialized np.ndarray
        :rtype: np.ndarray
        """
        return np.random.uniform(low=self.low, high=self.high, size=(x, y))


class HeNormal(Initializer):
    """He normal initializer.

    Draws samples from a normal distribution centered on 0 with stddev = sqrt(2/(x))
    where x is the first dimension of the numpy array.
    """
    def __init__(self):
        super().__init__()

    def generate(self, x: int, y: int) -> np.ndarray:
        """Return the initialized numpy array

        :param x: Size of first dimension of np.ndarray
        :type x: int
        :param y: Size of second dimension of np.ndarray
        :type y: int
        :return: Initialized np.ndarray
        :rtype: np.ndarray
        """
        stddev = np.sqrt(2 / x)
        return np.random.normal(loc=0.0, scale=stddev, size=(x, y))


class HeUniform(Initializer):
    """He uniform initializer.

    Draws samples from a uniform distribution within [-limit,limit] where
    limit = sqrt(6/(x))
    where x is the first dimension of the numpy array.
    """

    def __init__(self):
        super().__init__()

    def generate(self, x: int, y: int) -> np.ndarray:
        """Return the initialized numpy array

        :param x: Size of first dimension of np.ndarray
        :type x: int
        :param y: Size of second dimension of np.ndarray
        :type y: int
        :return: Initialized np.ndarray
        :rtype: np.ndarray
        """
        limit = np.sqrt(6 / x)
        return np.random.uniform(low=-limit, high=limit, size=(x, y))


class GlorotNormal(Initializer):
    """Glorot normal initializer.

    Also called Xavier normal initializer.
    Draws samples from a normal distribution centered on 0 with stddev = sqrt(2/(x+y))
    where x and y are the dimensions of the numpy array.
    """

    def __init__(self):
        super().__init__()

    def generate(self, x: int, y: int) -> np.ndarray:
        """Return the initialized numpy array

        :param x: Size of first dimension of np.ndarray
        :type x: int
        :param y: Size of second dimension of np.ndarray
        :type y: int
        :return: Initialized np.ndarray
        :rtype: np.ndarray
        """
        stddev = np.sqrt(2 / (x + y))
        return np.random.normal(loc=0.0, scale=stddev, size=(x, y))


class GlorotUniform(Initializer):
    """Glorot uniform initializer.

    Also called Xavier uniform initializer.
    Draws samples from a uniform distribution within [-limit,limit] where
    limit = sqrt(6/(x+y))
    where x and y are the dimensions of the numpy array.
    """

    def __init__(self):
        super().__init__()

    def generate(self, x: int, y: int) -> np.ndarray:
        """Return the initialized numpy array

        :param x: Size of first dimension of np.ndarray
        :type x: int
        :param y: Size of second dimension of np.ndarray
        :type y: int
        :return: Initialized np.ndarray
        :rtype: np.ndarray
        """
        limit = np.sqrt(6 / (x + y))
        return np.random.uniform(low=-limit, high=limit, size=(x, y))


class Zero(Initializer):
    """Initializer that returns a np.ndarray with every value set to 0"""

    def __init__(self):
        super().__init__()

    def generate(self, x: int, y: int) -> np.ndarray:
        """Return the initialized numpy array

        :param x: Size of first dimension of np.ndarray
        :type x: int
        :param y: Size of second dimension of np.ndarray
        :type y: int
        :return: Initialized np.ndarray
        :rtype: np.ndarray
        """
        return np.zeros((x, y))
