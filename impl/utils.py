import numpy as np


def exp_running_avg(running, new, gamma=.9):
    return gamma * running + (1. - gamma) * new


def accuracy(y_true, y_pred):
    return np.mean(y_pred == y_true)


def onehot(labels):
    y = np.zeros([labels.size, np.max(labels) + 1])
    y[range(labels.size), labels] = 1.
    return y


def softmax(X):
    eX = np.exp((X.T - np.max(X, axis=1)).T)
    return (eX.T / eX.sum(axis=1)).T


def sigmoid(X):
    return 1. / (1 + np.exp(-X))

# Pre-processing
def prepro(X_train, X_val, X_test):
    mean = np.mean(X_train)
    return X_train - mean, X_val - mean, X_test - mean