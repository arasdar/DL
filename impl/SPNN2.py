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
        h1, bn1_cache, self.bn_caches['bn1_mean'], self.bn_caches['bn1_var'] = l.bn_forward(h1, 
                                                        self.model['gamma1'], self.model['beta1'], 
                                                        (self.bn_caches['bn1_mean'], self.bn_caches['bn1_var']), 
                                                        train=train)
        h1, nl_cache1 = self.forward_nonlin(h1)
        u1 = None # referenced before assigned ERROR!
        if train: h1, u1 = l.dropout_forward(h1, self.p_dropout)

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
        cache = (X, h1_cache, h2_cache, h3_cache, nl_cache1, nl_cache2, u1, u2, bn1_cache, bn2_cache)
        return h3, cache

    def backward(self, y_pred, y_train, cache):
        X, h1_cache, h2_cache, score_cache, nl_cache1, nl_cache2, u1, u2, bn1_cache, bn2_cache = cache

        # Output layer
        grad_y = self.dloss_funs[self.loss](y_pred, y_train)

        # Third layer
        dh2, dW3, db3 = l.fc_backward(grad_y, score_cache)
        dW3 += reg.dl2_reg(self.model['W3'], self.lam)
        dh2 = self.backward_nonlin(dh2, nl_cache2)
        dh2 = l.dropout_backward(dh2, u2)
        dh2, dgamma2, dbeta2 = l.bn_backward(dh2, bn2_cache)

        # Second layer
        dh1, dW2, db2 = l.fc_backward(dh2, h2_cache)
        dW2 += reg.dl2_reg(self.model['W2'], self.lam)
        dh1 = self.backward_nonlin(dh1, nl_cache1)
        dh1 = l.dropout_backward(dh1, u1)
        dh1, dgamma1, dbeta1 = l.bn_backward(dh1, bn1_cache)

        # First layer
        _, dW1, db1 = l.fc_backward(dh1, h1_cache)
        dW1 += reg.dl2_reg(self.model['W1'], self.lam)

        grad = dict(
            W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3, gamma1=dgamma1,
            gamma2=dgamma2, beta1=dbeta1, beta2=dbeta2
        )

        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            W1=np.random.randn(D, H) / np.sqrt(D / 2.),
            W2=np.random.randn(H, H) / np.sqrt(H / 2.),
            W3=np.random.randn(H, C) / np.sqrt(H / 2.),
            b1=np.zeros((1, H)),
            b2=np.zeros((1, H)),
            b3=np.zeros((1, C)),
            gamma1=np.ones((1, H)),
            gamma2=np.ones((1, H)),
            beta1=np.zeros((1, H)),
            beta2=np.zeros((1, H))
        )

        self.bn_caches = dict(
            bn1_mean=np.zeros((1, H)),
            bn2_mean=np.zeros((1, H)),
            bn1_var=np.zeros((1, H)),
            bn2_var=np.zeros((1, H))
        )