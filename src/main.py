import numpy as np
import matplotlib.pyplot as plt
import utils
from model import Model, TrainConfig, EarlyStopType
from layer import Layer, ReLU, Dense, Dropout, Sigmoid
from initializers import RandomNormal, HeNormal, HeUniform, GlorotNormal, GlorotUniform, RandomUniform
from os.path import join
import random
import optimizer

# Set file paths based on added MNIST Datasets
input_path = '../MNIST'
training_images_filepath = join(input_path, 'train-images-idx3-ubyte/train-images-idx3-ubyte')
training_labels_filepath = join(input_path, 'train-labels-idx1-ubyte/train-labels-idx1-ubyte')
test_images_filepath = join(input_path, 't10k-images-idx3-ubyte/t10k-images-idx3-ubyte')
test_labels_filepath = join(input_path, 't10k-labels-idx1-ubyte/t10k-labels-idx1-ubyte')

# Load MINST dataset
mnist_dataloader = utils.MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath,
                                         test_labels_filepath)
(x_train_data, y_train_data), (x_test_data, y_test_data) = mnist_dataloader.load_data()

# Preprocess training and test data
data_preprocessor = utils.DataPreprocessor(n_classes=10)
x_train, y_train = data_preprocessor.preprocess(x_train_data, y_train_data)
x_test, y_test = data_preprocessor.preprocess(x_test_data, y_test_data)

# Split the training data into training and validation sets
train_val_ratio = 0.8
x_train, y_train, x_val, y_val = utils.train_val_splitter(x_train, y_train, train_val_ratio)

# Apply data augmentation if enabled
use_augmentation = True
if use_augmentation:
    aug = utils.DataAugmentor(True, 5, 5)
    x_train, y_train = aug.generate(x_train, y_train, generated_imgs_per_img=1, keep_originals=True)

# Show some random training and test images if enabled
show_images = False
if show_images:
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, x_train.shape[0])
        images_2_show.append(x_train[r].reshape((28, 28)))
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r].argmax()))

    for i in range(0, 5):
        r = random.randint(1, x_test.shape[0])
        images_2_show.append(x_test[r].reshape((28, 28)))
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r].argmax()))

    utils.show_images(images_2_show, titles_2_show)

# Display shapes of training data
print('x_train.shape', x_train.shape)
print('y_train.shape', y_train.shape)

# Initialize neural network model
input_size = x_train.shape[1]
output_size = y_train.shape[1]
weight_init = GlorotNormal()

nn = Model()
nn.add_layer(Dense(input_size, 128, weight_init=weight_init))
nn.add_layer(Dropout(0.98))
nn.add_layer(ReLU())
nn.add_layer(Dense(128, 256, weight_init=weight_init))
nn.add_layer(Dropout(0.97))
nn.add_layer(ReLU())
nn.add_layer(Dense(256, output_size, weight_init=weight_init))

# Perform hyperparameter optimization if enabled
use_hyperparameter_optim = True
if use_hyperparameter_optim:
    optim_config = optimizer.HyParamOptimConfig(
        n_epochs=(15,),
        batch_size=(96,),
        learning_rate=(0.15, 0.20, 0.4),
        early_stop=(5,),
        early_stop_type=(EarlyStopType.ACCURACY,),
        loss_function=(utils.SoftmaxCrossentropy(),))

    hyp_optim = optimizer.HyperParamOptimizer(optim_config)
    configs, reports = hyp_optim.optimize(nn, x_train, y_train, x_val, y_val)

    # Find the best configuration based on validation accuracy
    best_index = None
    highest_acc = 0
    for i in range(len(reports)):
        report_val_acc = reports[i].val_acc[-1]
        if highest_acc < report_val_acc:
            highest_acc = report_val_acc
            best_index = i

    # Apply the best configuration
    if best_index is not None:
        config = configs[best_index]
        print(f"\nBest hyperparameters are:\n"
              f"n_epochs = {config.n_epochs}, batch_size = {config.batch_size}, learning_rate = {config.learning_rate},\n "
              f"early_stop = {config.early_stop}, early_stop_type = {config.early_stop_type},\n"
              f"loss_function = {config.loss_function}.")

    # Plot validation accuracy for different configurations
    for i in range(len(reports)):
        plt.plot(reports[i].val_acc, label=f'n_e {configs[i].n_epochs} b {configs[i].batch_size} '
                                           f'lr {configs[i].learning_rate} e_s {configs[i].early_stop} '
                                           f'es_t {configs[i].early_stop_type} ls_f {str(configs[i].loss_function)}')
        plt.legend(loc='best')
    plt.grid()
    plt.show()

if not use_hyperparameter_optim:
    # Training configuration if hyperparameter optimization hasn't been done
    config = TrainConfig(n_epochs=35,
                         batch_size=128,
                         learning_rate=0.05,
                         early_stop=10,
                         early_stop_type=EarlyStopType.ACCURACY,
                         loss_function=utils.SquaredError())

# Train the neural network with the selected configuration
train_report = nn.train(x_train, y_train, x_val, y_val, config=config)

# Plot validation accuracy during training
plt.plot(train_report.val_acc, label='val accuracy')
plt.legend(loc='best')
plt.grid()
plt.show()

# Test the trained model on the test set
test_out = nn.predict(x_test)
result = utils.see_if_guess_is_correct(test_out, y_test)

test_accuracy = np.sum(result) / len(result)
print(f"Test accuracy = {test_accuracy}")
