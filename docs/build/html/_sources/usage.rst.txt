.. MNIST Classification with NumPy documentation usage file.

.. _usage:

Usage
=====

This section provides guidance on how to use MNIST Classification with NumPy, particularly focusing on the usage of `main.py`.

1. **Installation:**

   If you haven't already. Follow the installation steps described in :ref:`installation` section.

2. **Run main.py:**

   Activate the virtual environment if you haven't already:

   .. code-block:: bash

      source venv/bin/activate

   To execute the main functionality of MNIST Classification with NumPy, use the provided `main.py` script. Open a terminal and run:

   .. code-block:: bash

      python main.py

   This script performs the following steps:

   - Loads the MNIST dataset
   - Preprocesses the dataset, including splitting it into train/validation/test sets
   - Optionally applies data augmentation
   - Optionally enables the preview of the dataset
   - Constructs the neural network model
   - Optionally performs hyperparameter optimization
   - Trains the model with the chosen training configuration
   - Displays performance graphs
   - Evaluates the model on the test dataset

   Note: Ensure that the necessary dependencies and dataset are set up before running `main.py`.

3. **Additional Notes:**

   - Adjust the file paths in `main.py` to point to the correct locations of your MNIST dataset files.
   - Explore the options in `main.py` to customize the training process, such as enabling data augmentation, displaying images, or performing hyperparameter optimization.
   - The script includes comments and docstrings to help you understand each step.

