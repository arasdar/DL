# Deep Learning

This repository was initially forked-of from wiseodd-hipsternet.
The credit goes to him for the most basic implementation of the DL algorithms such as optimization algorithms, and regularizations.

On top of the existing approaches, my own implemented DL algorithms were also extensively added to this repository such as NN, CNN, and RNN folders and Python Notbook files ('.ipynb') for more readability and maintainability. 

This repository will be improved, maintained, and debugged over time but is not meant for production but prototyping.
That being said, you can use it to better study and learn the theoretical foundation of deep learning in Neural Networds in practice.

#### Network Architectures

1. Convolutional NN
2. Feed Forward NN
3. Recurrent NN
4. LSTM NN
5. GRU NN

#### Optimization algorithms

1. SGD
2. Momentum SGD
3. Nesterov Momentum
4. Adagrad
5. RMSprop
6. Adam

#### Loss functions

1. Cross Entropy
2. Hinge Loss
3. Squared Loss
4. L1 Regression
5. L2 Regression

#### Regularization

1. Dropout
2. The usual L1 and L2 regularization

#### Nonlinearities

1. ReLU
2. leaky ReLU
3. sigmoid
4. tanh
5. many more.....

#### Hipster techniques

1. BatchNorm
2. Xavier weight initialization

#### Pooling

1. Max pooling
2. Average pooling

### Datasets

1. Bike sharing dataset
2. Smartwatch multisensor experimental dataset
3. Language transalation scripts
4. Other scripts

## Run

1. Install anaconda/miniconda
2. Do `conda env create`
3. Enter the env `source activate arasdar-DL-env`
4. [Optional] To install Tensorflow: `chmod +x tensorflow.sh; ./tensorflow.sh`
5. To run, you can use either the ipynb files or the python files as following:
  1. `python run_mnist.py {ff|cnn}`; `cnn` for convnet model, `ff` for the feed forward model
  2. `python run_rnn.py {rnn|lstm|gru}`; `rnn` for vanilla RNN model, `lstm` for LSTM net model, `gru` for GRU net model

## License

This repository follows the same license as the parent repository 'wiseodd-hipsternet', i.e Unlicense License <http://unlicense.org>.