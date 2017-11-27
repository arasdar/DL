from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


from builtins import range
from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - dropout: Scalar between 0 and 1 giving dropout strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian with standard deviation equal to   #
        # weight_scale, and biases should be initialized to zero. All weights and  #
        # biases should be stored in the dictionary self.params, with first layer  #
        # weights and biases using the keys 'W1' and 'b1' and second layer weights #
        # and biases using the keys 'W2' and 'b2'.                                 #
        ############################################################################
        
        # Based on SciPy documentation:
        #         For random samples from N(\mu, \sigma^2), use:
        #         sigma * np.random.randn(...) + mu
        #         mu = 0.
        #         sigma = weight_scale
        
        # We assume an input dimension of D, a hidden dimension of H, and perform classification over C classes.
        D, H, C = input_dim, hidden_dim, num_classes
        
        self.params = dict(
            W1=np.random.randn(D, H) * weight_scale,
            W2=np.random.randn(H, C) * weight_scale,
            b1=np.zeros((1, H)),
            b2=np.zeros((1, C))
        )

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################


    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        
        #     The architecure should be affine - relu - affine - softmax.
        y_logit, fc_cache1 = affine_forward(b=self.params['b1'], w=self.params['W1'], x=X)
        y_act, nl_cache1 = relu_forward(x=y_logit)
        X = y_act.copy() # pass to the next layer
        
        # Softmax in included in the loss function: cross entropy
        scores, fc_cache2 = affine_forward(b=self.params['b2'], w=self.params['W2'], x=X)
        
        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################        
        reg_loss = regularization(lam=self.reg, model=self.params, reg_type='l2')
        
        loss, dy = softmax_loss(x=scores, y=y)
        loss += reg_loss
        
        # Output layer backward pass
        dX, dW2, db2 = affine_backward(cache=fc_cache2, dout=dy)
        dy = dX.copy() # pass to the previous layer
        
        # hidden layer
        dy = relu_backward(cache=nl_cache1, dout=dy)
        dX, dW1, db1 = affine_backward(cache=fc_cache1, dout=dy)
        
        # gradients
        grads = dict(W1 = dW1, b1 = db1, W2 = dW2, b2 = db2)

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        # We assume an input dimension of D, a hidden dimension of H, and perform classification over C classes.
        D, H, C = input_dim, hidden_dims, num_classes
        
        # Input layer parameters: fc, bn, nl, do
        param_in = {'W1': (np.random.randn(D, H[0]) * weight_scale),
                    'b1': np.zeros((1, H[0]))}
        self.params.update(param_in)
        if self.use_batchnorm:
            param_in_bn = {'gamma1': np.ones((1, H[0])), 
                           'beta1': np.zeros((1, H[0]))}
            self.params.update(param_in_bn)

        # Hidden layers parameters: fc, bn, nl, do
        for k in range(1, self.num_layers-1): # num_layers=3
            param_h = {'W{}'.format(k+1): (np.random.randn(H[k-1], H[k]) * weight_scale),
                       'b{}'.format(k+1): np.zeros((1, H[k]))}
            self.params.update(param_h)
            if self.use_batchnorm:
                param_h_bn = {'gamma{}'.format(k+1): np.ones((1, H[k])), 
                               'beta{}'.format(k+1): np.zeros((1, H[k]))}
                self.params.update(param_h_bn)


        # Output layer parameters: fc, softmax (included in loss function)
        param_out = {'W{}'.format(self.num_layers): (np.random.randn(H[self.num_layers - 2], C) * weight_scale),
                    'b{}'.format(self.num_layers): np.zeros((1, C))}
        self.params.update(param_out)

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        caches = []
        
        # Input and hidden layers: lr + [bn] + nl + [do]
        for k in range(1, self.num_layers):
            W = self.params['W{}'.format(k)]
            b = self.params['b{}'.format(k)]
            y_affine, affine_cache = affine_forward(b=b, w=W, x=X)
            if self.use_batchnorm:
                gamma = self.params['gamma{}'.format(k)]
                beta = self.params['beta{}'.format(k)]
                y_affine, bn_cache = batchnorm_forward(beta=beta, bn_param=self.bn_params[k-1], 
                                                       gamma=gamma, x=y_affine)
            else:
                bn_cache = None
            y_nl, nl_cache = relu_forward(x=y_affine)
            if self.use_dropout:
                y_nl, do_cache = dropout_forward(dropout_param=self.dropout_param, x=y_nl)
            else:
                do_cache = None
            X = y_nl.copy() # pass to the nect layer
            cache = (affine_cache, bn_cache, nl_cache, do_cache)
            caches.append(cache) # caches[k]

        # Output layer: lr + nl(softmax) included in loss function
        W = self.params['W{}'.format(self.num_layers)] # W3
        b = self.params['b{}'.format(self.num_layers)] # W3
        scores, scores_cache = affine_forward(b=b, w=W, x=X)

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        ############################################################################        
        reg_loss = regularization(lam=self.reg, model=self.params, reg_type='l2')

        loss, dy = softmax_loss(x=scores, y=y)
        loss += reg_loss
        
        # Output layer
        dX, dW, db = affine_backward(cache=scores_cache, dout=dy)
        grads['W{}'.format(self.num_layers)] = dW # W3
        grads['b{}'.format(self.num_layers)] = db # db3
        dy = dX.copy() # pass to the previous layer
        
        # hidden layer
        for k in reversed(range(1, self.num_layers)):
            affine_cache, bn_cache, nl_cache, do_cache = caches[k-1]
            if self.use_dropout:
                dy = dropout_backward(cache=do_cache, dout=dy)
            dy = relu_backward(cache=nl_cache, dout=dy)
            if self.use_batchnorm:
                dy, dgamma, dbeta = batchnorm_backward(cache=bn_cache, dout=dy)
                grads['gamma{}'.format(k)] = dgamma
                grads['beta{}'.format(k)] = dbeta
            dX, dW, db = affine_backward(cache=affine_cache, dout=dy)
            grads['W{}'.format(k)] = dW
            grads['b{}'.format(k)] = db

        pass
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads