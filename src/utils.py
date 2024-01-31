import numpy as np
import struct
from array import array
import matplotlib.pyplot as plt


def normalize(x):
    x_normalize = (x - np.min(x)) / (np.max(x) - np.min(x))
    return x_normalize


def see_if_guess_is_correct(y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
    y_pred_max = y_pred.argmax(axis=1)
    y_target_max = y_target.argmax(axis=1)
    return (y_pred_max == y_target_max).astype(int)


def one_hot(a: int, n: int) -> np.ndarray:
    """Return a one-hot vector of size "n" for a given value "a".

    For example:
    a = 3; n = 7;
    out = np.ndarray([0,0,1,0,0,0,0])

    :param a: value that the one-hot vector will represent
    :type a: int
    :param n: one-hot vector size, or number of classes
    :type n: int
    :return: one-hot vector as a numpy array
    :rtype: np.ndarray
    """
    out = np.zeros(shape=n)
    out[a] = 1
    return out


def shuffle_x_y(x, y):
    dataset_size = x.shape[0]
    shuffled_indices = np.random.permutation(dataset_size)
    x, y = x[shuffled_indices, :], y[shuffled_indices, :]
    return x, y


def train_val_splitter(x, y, ratio: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = shuffle_x_y(x, y)
    dataset_size = x.shape[0]

    limit = int(ratio * dataset_size)
    x_train, x_val = x[np.arange(0,limit), :], x[np.arange(limit, x.shape[0]), :]
    y_train, y_val = y[np.arange(0, limit), :], y[np.arange(limit, y.shape[0]), :]
    return x_train, y_train, x_val, y_val


class DataPreprocessor:
    def __init__(self, n_classes):
        self.n_classes = n_classes

    def preprocess(self, x, y):
        out_x = np.array([np.ravel(img) for img in x]) / 255.
        out_y = np.array([one_hot(label, self.n_classes) for label in y], dtype=int)
        return out_x, out_y


class DataAugmentor:
    def __init__(self, shuffle=True, translate_x=None, translate_y=None):
        self.shuffle = shuffle
        self.translate_x = translate_x
        self.translate_y = translate_y

    @staticmethod
    def _generate_translate_array(translate_limit: int, length: int) -> np.ndarray:
        # Here we create a list of all possible pixel-wise translate values
        t_range = list(range(-translate_limit, translate_limit + 1, 1))
        # Then we assign each possible translation the probability value.
        # Probability value for each is (1 / num_of_values)
        t_probs = [1. / len(t_range) for k in range(len(t_range))]
        # Create an array which hold the selected pixel-wise translation
        # value for each image in the input dataset
        t = np.random.choice(t_range, size=length, p=t_probs)
        return t

    @staticmethod
    def _translate(img, x, y):
        out = np.zeros_like(img)
        max_x, max_y = img.shape
        temp = img[-min(x, 0):max_x - max(x, 0), -min(y, 0):max_y - max(y, 0)]
        out[max(x, 0):max_x + min(x, 0), max(y, 0):max_y + min(y, 0)] = temp
        return out

    def generate(self, x: np.ndarray, y: np.ndarray, *,
                 generated_imgs_per_img: int = 1, keep_originals: bool = True) -> tuple[np.ndarray, np.ndarray]:
        assert generated_imgs_per_img > 0

        if keep_originals:
            # forward input x and y arrays to the output
            out_x = x.copy()
            out_y = y.copy()
        else:
            # Create a blank x and y array onto which we can concatenate later generated
            # augmented arrays
            out_x = np.array([], dtype=x.dtype).reshape(0, x.shape[1])
            out_y = np.array([], dtype=y.dtype).reshape(0, y.shape[1])

        for i in range(generated_imgs_per_img):
            # tx is translation on x-axis and ty is translation on y-axis
            tx = None
            ty = None
            if self.translate_x is not None and self.translate_x > 0:
                tx = self._generate_translate_array(self.translate_x, x.shape[0])
            if self.translate_y is not None and self.translate_y > 0:
                ty = self._generate_translate_array(self.translate_y, x.shape[0])

            aug_imgs = x
            aug_imgs = [img.reshape((28, 28)) for img in aug_imgs]
            if tx is not None:
                aug_imgs = [self._translate(aug_imgs[k], tx[k], 0) for k in range(x.shape[0])]
            if ty is not None:
                aug_imgs = [self._translate(aug_imgs[k], 0, ty[k]) for k in range(x.shape[0])]
            aug_imgs = np.array([np.ravel(img) for img in aug_imgs])
            out_x = np.concatenate((out_x, aug_imgs), axis=0)
            out_y = np.concatenate((out_y, y), axis=0)
        if self.shuffle:
            out_x, out_y = shuffle_x_y(out_x, out_y)
        return out_x, out_y



class LossFunction:
    """A base class for loss functions."""
    def loss(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        """Virtual function to be overriden. Return the output of loss function.

        Will raise NotImplementedError.

        :param y_pred: np.ndarray of shape [batch,n_classes] containing predicted values from the model
        :type y_pred: np.ndarray
        :param y_target: np.ndarray of shape [batch,n_classes] containing expected values in the form of a one-hot vector
        :type y_target: np.ndarray
        :return: Square error of shape [batch, n_classes]
        :rtype: np.ndarray
        """
        raise NotImplementedError()

    def derivative(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        """Virtual function to be overriden. Return the derivative of the loss function.

        Will raise NotImplementedError.

        :param y_pred: np.ndarray of shape [batch,n_classes] containing predicted values from the model
        :type y_pred: np.ndarray
        :param y_target: np.ndarray of shape [batch,n_classes] containing expected values in the form of a one-hot vector
        :type y_target: np.ndarray
        :return: Square error derivative of shape [batch, n_classes]. Used for the start of backpropagation
        :rtype: np.ndarray
        """
        raise NotImplementedError()


class SquaredError(LossFunction):
    """Square error loss function.

    Square error is calculated as:
    Error = (y_pred - y_target)^2
    """
    def loss(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        """Return a np.ndarray containing square error values of the same dimensions as input.

        :param y_pred: np.ndarray of shape [batch,n_classes] containing predicted values from the model
        :type y_pred: np.ndarray
        :param y_target: np.ndarray of shape [batch,n_classes] containing expected values in the form of a one-hot vector
        :type y_target: np.ndarray
        :return: Square error of shape [batch, n_classes]
        :rtype: np.ndarray
        """
        return (y_pred - y_target) * (y_pred - y_target)

    def derivative(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        """Return a derivative of the square error.

        Return a np.ndarray of shape [batch,n_classes] that represents the derivative of the square error function.
        Square error derivative is calculated as:
        Error = 2*(y_pred - y_target)

        :param y_pred: np.ndarray of shape [batch,n_classes] containing predicted values from the model
        :type y_pred: np.ndarray
        :param y_target: np.ndarray of shape [batch,n_classes] containing expected values in the form of a one-hot vector
        :type y_target: np.ndarray
        :return: Square error derivative of shape [batch, n_classes]. Used for the start of backpropagation
        :rtype: np.ndarray
        """
        return 2 * (y_pred - y_target)# / y_pred.shape[0]


class SoftmaxCrossentropy(LossFunction):
    """Softmax Cross-entropy loss.

    As cross-entropy loss is used to calculate the difference
    between two probability distributions, it naturally works well with softmax.
    Softmax converts logits to a probability distribution vector, and we use that
    to calculate the cross-entropy between the model's predictions and the expected
    output.
    """

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Return softmax probability distribution.

        Softmax converts a vector of real numbers into a probability distribution.
        Often used together with crossentropy and converts logits into probabilities for each class.

        :param x: Input np.ndarray of shape [batch, input_size]
        :type x: np.ndarray
        :return: Softmax np.ndarray of shape [batch, input_size]
        :rtype: np.ndarray
        """
        return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

    def loss(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        """Return cross-entropy loss.

        :param y_pred: np.ndarray of shape [batch,n_classes] containing predicted values from the model
        :type y_pred: np.ndarray
        :param y_target: np.ndarray of shape [batch,n_classes] containing expected values in the form of a one-hot vector
        :type y_target: np.ndarray
        :return: Cross-entropy loss of shape [batch, n_classes]
        :rtype: np.ndarray
        """
        y_soft = self._softmax(y_pred)
        crossentropy = - np.sum(y_target * np.log(y_soft), axis=-1)
        return crossentropy

    def derivative(self, y_pred: np.ndarray, y_target: np.ndarray) -> np.ndarray:
        """Return the derivative of the softmax cross-entropy function

        Simplification of the softmax cross-entropy derivative due to the fact that we use a one-hot vector
        during the calculation of cross-entropy. As the one-hot vector contains zeros for incorrect classes,
        they do not contribute to the loss. Derivative of softmax cross-entropy is simply subtracting 1
        from the softmax value at the index of the correct class.
        Explained in-depth here:
        https://www.michaelpiseno.com/blog/2021/softmax-gradient/

        :param y_pred: np.ndarray of shape [batch,n_classes] containing predicted values from the model
        :type y_pred: np.ndarray
        :param y_target: np.ndarray of shape [batch,n_classes] containing expected values in the form of a one-hot vector
        :type y_target: np.ndarray
        :return: Derivative of cross-entropy of shape [batch, n_classes]. Used for the start of backpropagation
        :rtype: np.ndarray
        """
        return (-y_target + self._softmax(y_pred)) # / y_pred.shape[0]


# https://www.kaggle.com/code/hojjatk/read-mnist-dataset/notebook
#
# MNIST Data Loader Class
#
class MnistDataloader(object):
    def __init__(self, training_images_filepath, training_labels_filepath,
                 test_images_filepath, test_labels_filepath):
        self.training_images_filepath = training_images_filepath
        self.training_labels_filepath = training_labels_filepath
        self.test_images_filepath = test_images_filepath
        self.test_labels_filepath = test_labels_filepath

    def read_images_labels(self, images_filepath, labels_filepath):
        labels = []
        with open(labels_filepath, 'rb') as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError('Magic number mismatch, expected 2049, got {}'.format(magic))
            labels = array("B", file.read())

        with open(images_filepath, 'rb') as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError('Magic number mismatch, expected 2051, got {}'.format(magic))
            image_data = array("B", file.read())
        images = []
        for i in range(size):
            images.append([0] * rows * cols)
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(28, 28)
            images[i][:] = img

        return images, labels

    def load_data(self):
        x_train, y_train = self.read_images_labels(self.training_images_filepath, self.training_labels_filepath)
        x_test, y_test = self.read_images_labels(self.test_images_filepath, self.test_labels_filepath)
        return (x_train, y_train), (x_test, y_test)


#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images) / cols) + 1
    plt.figure(figsize=(30, 20))
    index = 1
    for x in zip(images, title_texts):
        image = x[0]
        title_text = x[1]
        plt.subplot(rows, cols, index)
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize=15);
        index += 1