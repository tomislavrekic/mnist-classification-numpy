#https://www.kaggle.com/code/soham1024/basic-neural-network-from-scratch-in-python/notebook

import numpy as np
import matplotlib.pyplot as plt
import utils
from model import Model, TrainConfig, EarlyStopType
from layer import Layer, ReLU, Dense, Dropout, Sigmoid
from initializers import RandomNormal, HeNormal, HeUniform, GlorotNormal, GlorotUniform, RandomUniform
from os.path import join
import random

#
# Set file paths based on added MNIST Datasets
#
input_path = '../MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

#
# Load MINST dataset
#
mnist_dataloader = utils.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
(x_train_data, y_train_data), (x_test_data, y_test_data) = mnist_dataloader.load_data()

data_preprocessor = utils.DataPreprocessor(n_classes=10)

x_train, y_train = data_preprocessor.preprocess(x_train_data, y_train_data)
x_test, y_test = data_preprocessor.preprocess(x_test_data, y_test_data)

train_val_ratio = 0.8
x_train, y_train, x_val, y_val = utils.train_val_splitter(x_train, y_train, train_val_ratio)

aug = utils.DataAugmentor(8, 8)
x_train, y_train = aug.generate(x_train, y_train, generated_imgs_per_img=2, keep_originals=True)


show_images = False
if show_images:
    #
    # Show some random training and test images
    #
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, x_train.shape[0])
        images_2_show.append(x_train[r].reshape((28,28)))
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r].argmax()))

    for i in range(0, 5):
        r = random.randint(1, x_test.shape[0])
        images_2_show.append(x_test[r].reshape((28, 28)))
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r].argmax()))

    utils.show_images(images_2_show, titles_2_show)


print('X_train.shape', x_train.shape)
print('Y_train.shape', y_train.shape)
input_size = x_train.shape[1]
output_size = y_train.shape[1]

weight_init = GlorotUniform()
#weight_init = RandomNormal(mean=0.5, stddev=0.1)

network = Model()
network.add_layer(Dense(input_size, 100, weight_init=weight_init))
network.add_layer(Dropout(0.95))
network.add_layer(ReLU())
network.add_layer(Dense(100, 200, weight_init=weight_init))
network.add_layer(Dropout(0.95))
network.add_layer(ReLU())
network.add_layer(Dense(200, 200, weight_init=weight_init))
network.add_layer(Dropout(0.95))
network.add_layer(Dense(200, output_size, weight_init=weight_init))

config = TrainConfig(n_epochs=100,
                     batch_size=64,
                     learning_rate=0.001,
                     early_stop=10,
                     early_stop_type=EarlyStopType.ACCURACY,
                     loss_function=utils.SoftmaxCrossentropy())

train_report = network.train(x_train, y_train, x_val, y_val, config=config)
plt.plot(train_report.val_acc, label='val accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()


test_corrects = len(list(filter(lambda x: x == True, network.predict(x_test).argmax(axis=-1) == y_test.argmax(axis=-1))))
test_all = len(x_test)
test_accuracy = test_corrects/test_all #np.mean(test_errors)
print(f"Test accuracy = {test_corrects}/{test_all} = {test_accuracy}")