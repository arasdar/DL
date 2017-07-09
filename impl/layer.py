import numpy as np
import impl.utils as util
import impl.constant as c
import impl.regularization as reg
from impl.im2col import *


def fc_forward(X, W, b):
    out = X @ W + b
    cache = (W, X)
    return out, cache


def fc_backward(dout, cache):
    W, h = cache

    dW = h.T @ dout
    db = np.sum(dout, axis=0)
    dX = dout @ W.T

    return dX, dW, db

def dropout_forward(X, p_dropout):
    u = np.random.binomial(1, p_dropout, size=X.shape) / p_dropout
    out = X * u
    cache = u
    return out, cache


def dropout_backward(dout, cache):
    dX = dout * cache
    return dX


def bn_forward(X, gamma, beta, cache, momentum=.9, train=True):
    running_mean, running_var = cache

    if train:
        mu = np.mean(X, axis=0)
        var = np.var(X, axis=0)

        X_norm = (X - mu) / np.sqrt(var + c.eps)
        out = gamma * X_norm + beta

        cache = (X, X_norm, mu, var, gamma, beta)

        running_mean = util.exp_running_avg(running_mean, mu, momentum)
        running_var = util.exp_running_avg(running_var, var, momentum)
    else:
        X_norm = (X - running_mean) / np.sqrt(running_var + c.eps)
        out = gamma * X_norm + beta
        cache = None

    return out, cache, running_mean, running_var


def bn_backward(dout, cache):
    X, X_norm, mu, var, gamma, beta = cache

    N, D = X.shape

    X_mu = X - mu
    std_inv = 1. / np.sqrt(var + c.eps)

    dX_norm = dout * gamma
    dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
    dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)

    dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
    dgamma = np.sum(dout * X_norm, axis=0)
    dbeta = np.sum(dout, axis=0)

    return dX, dgamma, dbeta


def conv_forward(X, W, b, stride=1, padding=1):
    cache = W, b, stride, padding
    n_filters, d_filter, h_filter, w_filter = W.shape
    n_x, d_x, h_x, w_x = X.shape
    h_out = (h_x - h_filter + 2 * padding) / stride + 1
    w_out = (w_x - w_filter + 2 * padding) / stride + 1

    if not h_out.is_integer() or not w_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_col = im2col_indices(X, h_filter, w_filter, padding=padding, stride=stride)
    W_col = W.reshape(n_filters, -1)

    out = W_col @ X_col + b
    out = out.reshape(n_filters, h_out, w_out, n_x)
    out = out.transpose(3, 0, 1, 2)

    cache = (X, W, b, stride, padding, X_col)

    return out, cache


def conv_backward(dout, cache):
    X, W, b, stride, padding, X_col = cache
    n_filter, d_filter, h_filter, w_filter = W.shape

    db = np.sum(dout, axis=(0, 2, 3))
    db = db.reshape(n_filter, -1)

    dout_reshaped = dout.transpose(1, 2, 3, 0).reshape(n_filter, -1)
    dW = dout_reshaped @ X_col.T
    dW = dW.reshape(W.shape)

    W_reshape = W.reshape(n_filter, -1)
    dX_col = W_reshape.T @ dout_reshaped
    dX = col2im_indices(dX_col, X.shape, h_filter, w_filter, padding=padding, stride=stride)

    return dX, dW, db


def maxpool_forward(X, size=2, stride=2):
    def maxpool(X_col):
        max_idx = np.argmax(X_col, axis=0)
        out = X_col[max_idx, range(max_idx.size)]
        return out, max_idx

    return _pool_forward(X, maxpool, size, stride)


def maxpool_backward(dout, cache):
    def dmaxpool(dX_col, dout_col, pool_cache):
        dX_col[pool_cache, range(dout_col.size)] = dout_col
        return dX_col

    return _pool_backward(dout, dmaxpool, cache)


def avgpool_forward(X, size=2, stride=2):
    def avgpool(X_col):
        out = np.mean(X_col, axis=0)
        cache = None
        return out, cache

    return _pool_forward(X, avgpool, size, stride)


def avgpool_backward(dout, cache):
    def davgpool(dX_col, dout_col, pool_cache):
        dX_col[:, range(dout_col.size)] = 1. / dX_col.shape[0] * dout_col
        return dX_col

    return _pool_backward(dout, davgpool, cache)


def _pool_forward(X, pool_fun, size=2, stride=2):
    n, d, h, w = X.shape
    h_out = (h - size) / stride + 1
    w_out = (w - size) / stride + 1

    if not w_out.is_integer() or not h_out.is_integer():
        raise Exception('Invalid output dimension!')

    h_out, w_out = int(h_out), int(w_out)

    X_reshaped = X.reshape(n * d, 1, h, w)
    X_col = im2col_indices(X_reshaped, size, size, padding=0, stride=stride)

    out, pool_cache = pool_fun(X_col)

    out = out.reshape(h_out, w_out, n, d)
    out = out.transpose(2, 3, 0, 1)

    cache = (X, size, stride, X_col, pool_cache)

    return out, cache


def _pool_backward(dout, dpool_fun, cache):
    X, size, stride, X_col, pool_cache = cache
    n, d, w, h = X.shape

    dX_col = np.zeros_like(X_col)
    dout_col = dout.transpose(2, 3, 0, 1).ravel()

    dX = dpool_fun(dX_col, dout_col, pool_cache)

    dX = col2im_indices(dX_col, (n * d, 1, h, w), size, size, padding=0, stride=stride)
    dX = dX.reshape(X.shape)

    return dX

################ Original Activation functions
def relu_forward(X):
    out = np.maximum(X, 0)
    cache = X
    return out, cache

def relu_backward(dout, cache):
    dX = dout.copy()
    dX[cache <= 0] = 0
    return dX

def lrelu_forward(X, a=1e-3):
    out = np.maximum(a * X, X)
    cache = (X, a)
    return out, cache

def lrelu_backward(dout, cache):
    X, a = cache
    dX = dout.copy()
    dX[X < 0] *= a
    return dX

def sigmoid_forward(X):
    out = util.sigmoid(X)
    cache = out
    return out, cache

def sigmoid_backward(dout, cache):
    return cache * (1. - cache) * dout

def tanh_forward(X):
    out = np.tanh(X)
    cache = out
    return out, cache

def tanh_backward(dout, cache):
    dX = (1 - cache**2) * dout
    return dX

def integ_tanh_forward(X):
    out = np.log(np.cosh(X)) # ln (cosh x) + C.
    cache = X
    return out, cache

def integ_tanh_backward(dout, cache):
    X = cache
    dX = dout * np.tanh(X)
    return dX

def tanh_forward(X):
    out = np.tanh(X)
    cache = X
    return out, cache

def tanh_backward(dout, cache):
    X = cache
    dX = dout * (1 - (np.tanh(X)**2)) # dTanh = 1-tanh**2
    return dX

# Softplus: integral of sigmoid/softmax
def softplus_forward(X):
    out = np.log(1+np.exp(X))
    cache = X
    return out, cache

def softplus_backward(dout, cache):
    X = cache
    dX = dout * util.sigmoid(X)
    return dX

## Noisy ReLU by Hinton et al. (backprop)
# Wickipedia: Rectified linear units can be extended to include Gaussian noise, making them noisy ReLUs, giving[6].
# @inproceedings{nair2010rectified,
#   title={Rectified linear units improve restricted boltzmann machines},
#   author={Nair, Vinod and Hinton, Geoffrey E},
#   booktitle={Proceedings of the 27th international conference on machine learning (ICML-10)},
#   pages={807--814},
#   year={2010}
# }
# Abstract
# Restricted Boltzmann machines were developed using binary stochastic hidden units.
# These can be generalized by replacing each binary unit by an infinite number of copies that all have the same weights but have progressively more negative biases. 
# The learning and inference rules for these “Stepped Sigmoid Units” are unchanged. 
# They can be approximated efficiently by noisy, rectified linear units. 
# Compared with binary units, these units learn features that are better for object recognition on the NORB dataset and face verification on the Labeled Faces in the Wild dataset. 
# Unlike binary units, rectified linear units preserve information about relative intensities as information travels through multiple layers of feature detectors.
def noisy_relu_forward(X):
    noise = np.random.normal(loc=0.0, scale=1.0, size=None)
    out = np.maximum(0, X+noise)
    cache = (X, noise)
    return out, cache
    
def noisy_relu_backward(dout, cache):
    X, noise = cache
    dX[X < 0] = 0
    return dX

## Leaky ReLU by Andrew NG et al. (Google brain)
# Wickipedia: Leaky ReLUs allow a small, non-zero gradient when the unit is not active.
# @inproceedings{maas2013rectifier,
#   title={Rectifier nonlinearities improve neural network acoustic models},
#   author={Maas, Andrew L and Hannun, Awni Y and Ng, Andrew Y},
#   booktitle={Proc. ICML},
#   volume={30},
#   number={1},
#   year={2013}
# }
# Abstract
# DNNs acoustic models produce substantial gains in large vocabulary continuous speech recognition systems.
# Emerging work with ReLUs demonstrates additional gains in final system performance relative to more commonly used sigmoidal nonlinearities. 
# In this work, we explore the use of deep rectifier networks as acoustic models for the 300 hour Switchboard conversational speech recognition task. 
# Using simple training procedures without pretraining, networks with rectifier nonlinearities produce 2% absolute reductions in word error rates over their sigmoidal counterparts. 
# We analyze hidden layer representations to quantify differences in how ReLUs encode inputs as compared to sigmoidal units.
# Finally, we evaluate a variant of the ReLU with a gradient more amenable to optimization in an attempt to further improve deep rectifier networks.
def leaky_relu_fwd(X):
    m=1e-3 # 1e-3==0.001
    X[X < 0] *= m
    return X
def leaky_relu_bwd(X, dX):
    m=1e-3
    dX[X < 0] *= m
    return dX

## ELUs[edit] by Sepp Hochreiter et al. (LSTM with Schmidhuber)
# Wikipedia: Exponential linear units try to make the mean activations closer to zero which speeds up learning. 
# It has been shown that ELUs obtain higher classification accuracy than ReLUs.[11]
# @article{clevert2015fast,
#   title={Fast and accurate deep network learning by exponential linear units (elus)},
#   author={Clevert, Djork-Arn{\'e} and Unterthiner, Thomas and Hochreiter, Sepp},
#   journal={arXiv preprint arXiv:1511.07289},
#   year={2015}
# }
# Abstract:
# We introduce the "exponential linear unit" (ELU) which speeds up learning in deep neural networks and leads to higher classification accuracies. 
# Like rectified linear units (ReLUs), leaky ReLUs (LReLUs) and parametrized ReLUs (PReLUs), ELUs alleviate the vanishing gradient problem via the identity for positive values. 
# However, ELUs have improved learning characteristics compared to the units with other activation functions. 
# In contrast to ReLUs, ELUs have negative values which allows them to push mean unit activations closer to zero like batch normalization but with lower computational complexity.
# Mean shifts toward zero speed up learning by bringing the normal gradient closer to the unit natural gradient because of a reduced bias shift effect. 
# While LReLUs and PReLUs have negative values, too, they do not ensure a noise-robust deactivation state. 
# ELUs saturate to a negative value with smaller inputs and thereby decrease the forward propagated variation and information.
# Therefore, ELUs code the degree of presence of particular phenomena in the input, while they do not quantitatively model the degree of their absence. 
# In experiments, ELUs lead not only to faster learning, but also to significantly better generalization performance than ReLUs and LReLUs on networks with more than 5 layers. 
# On CIFAR-100 ELUs networks significantly outperform ReLU networks with batch normalization while batch normalization does not improve ELU networks. 
# ELU networks are among the top 10 reported CIFAR-10 results and yield the best published result on CIFAR-100, without resorting to multi-view evaluation or model averaging. 
# On ImageNet, ELU networks considerably speed up learning compared to a ReLU network with the same architecture, obtaining less than 10% classification error for a single crop, single model network.
# a is a hyper-parameter to be tuned and {ReLU a>=0} {exponential leak a>=0} is a constraint.
def elu_fwd(X):
    X_pos = np.maximum(0.0, X) # ReLU
    m = 1.0 # 1e-3==0.001, a==m, 0.0 <= a <= 1.0, active/passive, on/off
    X_neg = np.minimum(X, 0) # otherwise: if X<=0, Exp Leaky ReLU
    X_neg_exp = m * (np.exp(X_neg)-1) # a: slope, a>=0
    return X_pos + X_neg_exp
def elu_bwd(X, dX):
    m = 1.0 # 1e-3==0.001, a==m, 0.0 <= a <= 1.0, active/passive, on/off
    X_neg = np.minimum(X, 0) # otherwise: if X<=0, Exp Leaky ReLU
    m_neg_exp = m * np.exp(X_neg) # derivative of abs(np.exp(X_neg)-1) # a: slope, a>=0
    return dX * m_neg_exp

# # PReLU: Learning-based ReLU by He, Kaiming (ResNet with Microsoft Asia)
# Wikipedia: Parametric ReLUs take this idea further by making the coefficient of leakage into a parameter that is learned along with the other neural network parameters.[10]
# @inproceedings{he2015delving,
#  title={Delving deep into rectifiers: Surpassing human-level performance on imagenet classification},
#  author={He, Kaiming and Zhang, Xiangyu and Ren, Shaoqing and Sun, Jian},
#   booktitle={Proceedings of the IEEE international conference on computer vision},
#   pages={1026--1034},
#  year={2015}
# }
# Abbstract:
# Rectified activation units (rectifiers) are essential for state-of-the-art neural networks. 
# In this work, we study rectifier neural networks for image classification from two aspects. 
# First, we propose a Parametric Rectified Linear Unit (PReLU) that generalizes the traditional rectified unit. 
# PReLU improves model fitting with nearly zero extra computational cost and little overfitting risk. 
# Second, we derive a robust initialization method that particularly considers the rectifier nonlinearities. 
# This method enables us to train extremely deep rectified models directly from scratch and to investigate deeper or wider network architectures. 
# Based on our PReLU networks (PReLU-nets), we achieve 4.94% top-5 test error on the ImageNet 2012 classification dataset. 
# This is a 26% relative improvement over the ILSVRC 2014 winner (GoogLeNet, 6.66%). 
# To our knowledge, our result is the first to surpass human-level performance (5.1%, Russakovsky et al.) on this visual recognition challenge.
# Wikipedia: Note that for {\displaystyle a\leq 1} {\displaystyle a\leq 1}, this is equivalent to
# {\displaystyle f(x)=\max(x,ax)} f(x)=\max(x,ax)
# and thus has a relation to "maxout" networks.[10]
# Maxout nets by Ian Goodfellow (GAN inventor)
# The initialization part:
# 1- Start with the ReLU: m_pos=1.0 and m_neg=0.0
# 2- The active/on/activated neuron/unit/cell is 1.0 and the passive/off one is 0.0
# 3- Learnable units, constant units, and differentiable units/neurons/cells
# 4- Fully learnable units/cells/neurons
# 5- Biologically speaking they are NOT perfectly on/off or passive/active but leaky
def leaky_relu_fwd(X, m_neg):
    X[X < 0] *= m_neg
    return X
def leaky_relu_bwd(X, dX, m_neg):
    dX[X < 0] *= m_neg
    return dX

def leaky_relu_fwd2(X, m_neg): # m_neg: mat_1x1, a constant/leak/very small
    X_pos = np.maximum(0.0, X) # if X>0, ReLU, m_pos==1.0, constant
    X_neg = np.minimum(X, 0.0) # otherwise: if X<=0, max(mx, x)
    X_neg = np.maximum(X_neg, m_neg * X_neg) # maxout, 0 <= m_neg <= 1
    return X_pos + X_neg
def leaky_relu_bwd2(X, dX, m_neg):
    X_neg = np.minimum(X, 0.0) # otherwise: if X<=0, max(mx, x)
    X_neg = np.maximum(X_neg, m_neg * X_neg) # maxout, 0 <= m_neg <= 1
    dX[X_neg<0] *= m_neg
    return dX

def check_m(m):
    m = abs(m) # m_neg>0
    m = np.minimum(m, 1.0) # m_neg < 1.0/on/active
    m = np.maximum(0.0, m) # m_neg > 0.0/off/passive
    return m

def p_leaky_relu_fwd(X, m_neg): # m_neg: mat_1x1, a constant/leak/very small
    X_pos = np.maximum(0.0, X) # if X>0, ReLU, m_pos==1.0, constant
    m_neg = check_m(m=m_neg)
    X_neg = np.minimum(X, 0.0) # otherwise: if X<=0, max(mx, x)
    X_neg = np.maximum(X_neg, m_neg * X_neg) # maxout, 0 <= m_neg <= 1
    return X_pos + X_neg
def p_leaky_relu_bwd(X, dX, m_neg):
    X = p_leaky_relu_fwd(m_neg=m_neg, X=X)
    # dm_neg
    dX_1xm = dX.reshape(1, -1)
    X_neg = np.minimum(X, 0.0) # X < 0
    X_neg_1xm = X_neg.reshape(1, -1)
    dm_neg = dX_1xm @ X_neg_1xm.T # mat_1x1=dm_neg
    # dX ouput
    m_neg = check_m(m=m_neg)
    dX[X<0] *= m_neg
    return dX, dm_neg[0, 0]

def fp_leaky_relu_fwd(X, m_neg, m_pos): # m_neg: mat_1x1, a constant/leak/very small
    m_pos = check_m(m=m_pos)
    X_pos = np.maximum(0.0, X) # if X>0, ReLU, m_pos==1.0, constant
    X_pos = np.minimum(X * m_pos, X)
    m_neg = check_m(m=m_neg)
    X_neg = np.minimum(X, 0.0) # otherwise: if X<=0, max(mx, x)
    X_neg = np.maximum(X_neg, m_neg * X_neg) # maxout, 0 <= m_neg <= 1
    return X_pos + X_neg
def fp_leaky_relu_bwd(X, dX, m_neg, m_pos):
    X = fp_leaky_relu_fwd(m_neg=m_neg, m_pos=m_pos, X=X)
    dX_1xm = dX.reshape(1, -1)
    # dm_pos
    X_pos = np.maximum(0.0, X) # X > 0
    X_pos_1xm = X_pos.reshape(1, -1)
    dm_pos = dX_1xm @ X_pos_1xm.T # mat_1x1=dm_pos
    # dm_neg
    X_neg = np.minimum(X, 0.0) # X < 0
    X_neg_1xm = X_neg.reshape(1, -1)
    dm_neg = dX_1xm @ X_neg_1xm.T # mat_1x1=dm_neg
    # dX ouput
    m_pos = check_m(m=m_pos)
    m_neg = check_m(m=m_neg)
    dX[X<0] *= m_neg # m_neg=dX.shape
    dX[X>0] *= m_pos # m_pos=dX.shape
    return dX, dm_neg, dm_pos

def fp_leaky_relu_bwd2(each_X, each_dX, m_neg, m_pos): # each sample in X and dX
    each_X = fp_leaky_relu_fwd(m_neg=m_neg, m_pos=m_pos, X=each_X)
    # dm_neg/_pos
    each_X_pos = np.maximum(0.0, each_X) # X > 0
    dm_pos = each_dX * each_X_pos # mat_nxm=dm_pos.shape==dX.shape==X.shape
    each_X_neg = np.minimum(each_X, 0.0) # X < 0
    dm_neg = each_dX * each_X_neg # mat_nxm=dm_neg.shape==dX.shape==X.shape
    # dX
    each_dX_X_neg = each_dX.copy()
    each_dX_X_neg[each_X>0] =0
    m_neg = check_m(m=m_neg)
    each_dX_X_neg *= m_neg
#     each_dX[each_X<0] *= m_neg # m_neg.shape==dX.shape
    each_dX_X_pos = each_dX.copy()
    each_dX_X_pos[each_X<0] =0
    m_pos = check_m(m=m_pos)
    each_dX_X_pos *= m_pos
#     each_dX[each_X>0] *= m_pos # m_pos.shape==dX.shape
    return each_dX, dm_neg, dm_pos # dX.shape==dm_neg.shape==dm_pos.shape