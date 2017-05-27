import numpy as np
import impl.loss as loss_fun
import impl.layer as l
import impl.regularization as reg
import impl.utils as util
import impl.NN as nn
import impl.pyramidnet as pyramidnet

class SPNN(nn.NN):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=False):
        
        # 1st layer: Conv layer from input
        h1, h1_cache = l.fc_forward(X, self.model['W1'], self.model['b1'])
        h1, nl_cache1 = self.forward_nonlin(h1)

        # midst layer: Pyrmidnet depth
        h2, h2_cache = l.fc_forward(h1, self.model['W2'], self.model['b2'])
        h2, bn2_cache, self.bn_caches['bn2_mean'], self.bn_caches['bn2_var'] = l.bn_forward(h2, 
                                                        self.model['gamma2'], self.model['beta2'], 
                                                        (self.bn_caches['bn2_mean'], self.bn_caches['bn2_var']), 
                                                        train=train)
        h2, nl_cache2 = self.forward_nonlin(h2)
        u2 = None # referenced before assigned ERROR!
        if train: h2, u2 = l.dropout_forward(h2, self.p_dropout)

        # last layer: FC to the output layer
        h3, h3_cache = l.fc_forward(h2, self.model['W3'], self.model['b3'])
        cache = (X, h1_cache, h2_cache, h3_cache, nl_cache1, nl_cache2, u2, bn2_cache)
        return h3, cache

    def backward(self, y_pred, y_train, cache):
        X, h1_cache, h2_cache, score_cache, nl_cache1, nl_cache2, u2, bn2_cache = cache

        # Output layer
        grad_y = self.dloss_funs[self.loss](y_pred, y_train)

        # Third layer
        dh2, dW3, db3 = l.fc_backward(grad_y, score_cache)

        # Second layer
        dh2 = l.dropout_backward(dh2, u2)
        dh2 = self.backward_nonlin(dh2, nl_cache2)
        dh2, dgamma2, dbeta2 = l.bn_backward(dh2, bn2_cache)
        dh1, dW2, db2 = l.fc_backward(dh2, h2_cache)

        # First layer
        dh1 = self.backward_nonlin(dh1, nl_cache1)
        dX_, dW1, db1 = l.fc_backward(dh1, h1_cache)

        # grad for model parameters
        grad = dict(
            W1=dW1,
            b1=db1,

            W2=dW2,
            b2=db2,
            gamma2=dgamma2,
            beta2=dbeta2,

            W3=dW3,
            b3=db3        
        )

        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            W1=np.random.randn(D, H) / np.sqrt(D / 2.),
            b1=np.zeros((1, H)),

            W2=np.random.randn(H, H) / np.sqrt(H / 2.),
            b2=np.zeros((1, H)),
            gamma2=np.ones((1, H)),
            beta2=np.zeros((1, H)),

            W3=np.random.randn(H, C) / np.sqrt(H / 2.),
            b3=np.zeros((1, C))
        )
        self.bn_caches = dict(            
            bn2_mean=np.zeros((1, H)),
            bn2_var=np.zeros((1, H))
        )