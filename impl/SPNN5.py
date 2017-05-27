import numpy as np
import impl.loss as loss_fun
import impl.layer as l
import impl.NN as nn

class SPNN(nn.NN):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)
        
    def forward(self, X, train=False):

        # 1st layer: Input to Conv
        h1, h1_cache = l.conv_forward(X=X, W=self.model['W1'], b=self.model['b1']) 
        h1, nl1_cache = l.relu_forward(h1)

        # last layer : FC to Output
        h3 = h1.reshape([h1.shape[0], -1])
        h3, h3_cache = l.fc_forward(X=h3, W=self.model['W3'], b=self.model['b3'])

        cache = h1_cache, nl1_cache, h3_cache
        return h3, cache

    def backward(self, y, y_train, cache):

        dy = self.dloss_funs[self.loss](y, y_train)
        h1_cache, nl1_cache, h3_cache = cache

        # last layer
        dh3, dw3, db3 = l.fc_backward(dout=dy, cache=h3_cache)
        dh1 = dh3.reshape(nl1_cache.shape)

        # 1st layer
        dh1 = l.relu_backward(dout=dh1, cache=nl1_cache)
        dX, dw1, db1 = l.conv_backward(dout=dh1, cache=h1_cache)

        # grad for GD
        grad = dict(
            W1=dw1, 
            b1=db1,

            W3=dw3, 
            b3=db3
            )
        
        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            W1=np.random.randn(H, 1, 3, 3) / np.sqrt(H / 2.),
            b1=np.zeros((H, 1)),

            W3=np.random.randn(H * D, C) / np.sqrt(H * D / 2.), 
            b3=np.zeros((1, C))
            )