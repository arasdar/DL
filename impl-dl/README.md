# Deep Learning

This repository was initially the forked-of-version from wiseodd-hipsternet.
The credit goes to him for the most basic fundemental implementation of the DL algorithms such as network architectures, optimization algorithms, and regularizations techniques.

On top of the existing approaches, my own implemented DL algorithms were also extensively added to this repository such as NN, CNN, and RNN folders and Python Notbook files ('.ipynb') for more readability and maintainability. 

This repository will be continuously improved, maintained, and debugged over time but is meant for learning and prototyping the DL approachs and NOT meant for production.

You can use this repository to better study and learn the theoretical foundation of deep learning in Neural Networds in practice such as the data preperation, building neural netowork architectures, backprp-based learning approaches, and how to run the learning session in deep NN architecture. 

This repository is including mainly the NumPy-based implementation (np-based) of DL stuff from scratch, thanks to Hipsternet/Wiseodd/Wisetodd, along with some tensorflow-based (tf-based) implementation as well.

#### Neural Network Architectures

1. Convolutional NN
2. Feed Forward NN
3. Recurrent NN
4. LSTM NN
5. GRU NN: There are modified GRUs available as well.
7. Deep CNN
8. Deep RNN
9. Deep LSTM
10. Deep GRU: Its deep modified versions are also available.

#### Optimization algorithms: Backprop-based learning

1. SGD
2. Momentum SGD
3. Nesterov Momentum
4. Adagrad
5. RMSprop
6. Adam
7. Adam_RNN

#### Loss/Error functions

1. Cross Entropy
2. Hinge Loss
3. Squared Loss
4. L1 Regression
5. L2 Regression

#### Regularization

1. Dropout
2. The usual L1 and L2 regularization
3. Dropout_SELU

#### Nonlinearities

1. ReLU
2. Leaky ReLU
3. Sigmoid
4. Tanh
5. PReLU: Parameteric ReLU or maxout network
6. Fully learable PReLU
7. Softplus: Integral-Sigmoid/Softmax
8. Softmax for multiple-output classification
9. Integ-Tanh: Integral of Tanh
10. Fully Learnable PReLU
11. ELU: Exponentioal Linear Unit
11. SELU: Scaled ELU

#### Normalization and initialization techniques

1. BatchNorm
2. Xavier weight initialization

#### Pooling

1. Max pooling
2. Average pooling

### Tensorflow-based implementations

1. CNN for mnist by Wiseodd
2. Autoencoder for mnist by Wiseodd
3. RNN by Wiseodd
3. RNN for script-generation- Deep Unsupervised Learning
4. RNN for sequence-to-sequence (seq2seq) learning for langugae translation
5. GAN for face generation on CelebA and mnist dataset
6. CNN for image classification on CIFAR dataset.
7. RNN for time-series data or historical data sequence learning

### Datasets

1. Bike sharing dataset
2. Smartwatch multisensor experimental dataset
3. Language transalation scripts
4. Mnist - NOT included but automatic download
5. CelebA - automatic download
6. CIFAR - automatic download

### Cloud computing guidance/how-to

1. Floydhub

## Run

1. Install anaconda/miniconda
2. Do `conda env create`
3. Enter the env `source activate arasdar-DL-env`
4. To run, you can use either the ipynb files or the python files as following:
  1. `python run_mnist.py {ff|cnn}`; `cnn` for convnet model, `ff` for the feed forward model
  2. `python run_rnn.py {rnn|lstm|gru}`; `rnn` for vanilla RNN model, `lstm` for LSTM net model, `gru` for GRU net model

## License

This repository follows the same license as the parent repository 'wiseodd-hipsternet', i.e Unlicense License <http://unlicense.org>.
