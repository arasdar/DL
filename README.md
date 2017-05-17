# Deep Learning

This repository was initially forked-of from wiseodd-hipsternet.
The credit goes to him for the most basic implementation of the DL algorithms such as optimization algorithms, and regularizations.

On top of the existing approaches, my own implemented DL algorithms were also extensively added to this repository such as NN, CNN, and RNN folders and Python Notbook files ('.ipynb') for more readability and maintainability. 

This repository will be improved, maintained, and debugged over time but is not meant for production but prototyping.
That being said, you can use it to better study and learn the theoretical foundation of deep learning in Neural Networds in practice.

This repository is including the implementation based on NumPy (NP) from scratch, thanks to Hipsternet/Wiseodd/Wisetodd, along with tensorflow-based (tf-based) implemented NNs.

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
2. Leaky ReLU
3. Sigmoid
4. Tanh
5. PReLU: Parameteric ReLU
6. Fully learable PReLU
7. Softplus: Integral-Sigmoid/Softmax
8. Softmax for multiple-output classification
9. Integ-Tanh: Integral of Tanh

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
