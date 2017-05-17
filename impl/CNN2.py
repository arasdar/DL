import numpy as np
import hipsternet.loss as loss_fun
import hipsternet.layer as l
import hipsternet.regularization as reg
import hipsternet.utils as util
import hipsternet.NN as nn

class CNN2(nn.NN):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)

    def forward(self, X, train=False):
        # Conv-1 layer forward
        h1, h1_cache = l.conv_forward(X=X, W=self.model['W1'], b=self.model['b1'])
        h2 = l.fp_leaky_relu_fwd(m_neg=self.model['m1_neg'], m_pos=self.model['m1_pos'], X=h1)
        h2 += self.model['bm1']

        # FC-1 layer forward
        # Flattening- From Conv layer to FC layer forward
        h2_flat = h2.ravel().reshape(X.shape[0], -1) # mat_1xh2, h2=Channels*Height*Width, x=mat_1xcxhxw
        h2, h2_cache = l.fc_forward(X=h2_flat, W=self.model['W2'], b=self.model['b2'])
        h3 = l.fp_leaky_relu_fwd(m_neg=self.model['m2_neg'], m_pos=self.model['m2_pos'], X=h2)
        h3 += self.model['bm2']

        # FC-2 layer forward
        y, y_cache = l.fc_forward(X=h3, W=self.model['W3'], b=self.model['b3']) # y==y_logits
        # y_prob = softmax_fwd(X=y) (included in loss/error function)

        # Output forward
        cache = X, h1, h1_cache, h2, h2_cache, h3, y_cache
        return y, cache

    def backward(self, y, y_train, cache):
        X, h1, h1_cache, h2, h2_cache, h3, y_cache = cache

        # Output layer backward
        dy = self.dloss_funs[self.loss](y, y_train) # y==y_logits

        # FC-2 layer backward
        # dy = softmax_bwd(dX=dy_prob, X=y_logits/y) (included in the loss/error function)
        dh3, dW3, db3 = l.fc_backward(dout=dy, cache=y_cache)
        
        # FC-1 layer backward
        # h3 += self.model['bm2'] # forward/fwd pass/prop
        dbm2 = np.zeros_like(dh3[0]) # initialize the shape and the value/content
        for each in dh3: # each row in dh3 which is each sample in the minibatch
            dbm2 += each
        dm2_neg = np.zeros_like(dbm2)
        dm2_pos = np.zeros_like(dbm2)
        dh2 = np.zeros_like(dh3)
        for idx in range(len(dh3)): # idx: i, row, y, or height/h
            dh2[idx], each_dm2_neg, each_dm2_pos = l.fp_leaky_relu_bwd2(each_dX=dh3[idx], each_X=h2[idx], 
                                                            m_neg=self.model['m2_neg'],
                                                            m_pos=self.model['m2_pos'])
            dm2_neg += each_dm2_neg
            dm2_pos += each_dm2_pos

        #         dh2, dm2_neg, dm2_pos = fp_leaky_relu_bwd(dX=dh3, m_neg=self.model['m2_neg'], 
        #                                                   m_pos=self.model['m2_pos'], X=h2)
        dh2_flat, dW2, db2 = l.fc_backward(dout=dh2, cache=h2_cache)

        # Flattening- From FC layer to Conv layer backward
        dh2 = dh2_flat.ravel().reshape(h1.shape)

        # Conv-1 layer
        #         dbm1 += each for each in dh2 # dh2==mat_txn, t: number of samples in a minibatch/fullbatch/batch, n: num of dim
        #         h2 += self.model['bm1'] # forward pass/prop
        dbm1 = np.zeros_like(dh2[0]) # dm1 with the shape of one sample in minibatch init with zeros
        for each in dh2:
            dbm1 += each
        dm1_neg = np.zeros_like(dbm1)
        dm1_pos = np.zeros_like(dbm1)
        dh1 = np.zeros_like(dh2)
        #         print(dm1_neg.shape, dh1.shape, len(dh2), h1.shape)
        for idx in range(len(dh2)): # idx: i, row, y, or height/h
            dh1[idx], each_dm1_neg, each_dm1_pos = l.fp_leaky_relu_bwd2(each_dX=dh2[idx], each_X=h1[idx], 
                                                            m_neg=self.model['m1_neg'], 
                                                            m_pos=self.model['m1_pos'])
            dm1_neg += each_dm1_neg
            dm1_pos += each_dm1_pos

        #         dh1, dm1_neg, dm1_pos = fp_leaky_relu_bwd(dX=dh2, m_neg=self.model['m1_neg'], 
        #                                                   m_pos=self.model['m1_neg'], X=h1)
        dX, dW1, db1 = l.conv_backward(dout=dh1, cache=h1_cache) # X is visible/input layer, dX? No use??

        # gradients for gradient descent
        grad = dict(W1=dW1, W2=dW2, W3=dW3, b1=db1, b2=db2, b3=db3, 
                    m1_neg=dm1_neg, m2_neg=dm2_neg, 
                    m1_pos=dm1_pos, m2_pos=dm2_pos, 
                    bm1=dbm1, bm2=dbm2)
        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            W1=np.random.randn(D, 1, 3, 3) / np.sqrt(D / 2.),
            W2=np.random.randn(D * 28 * 28, H) / np.sqrt(D * 14 * 14 / 2.),
            W3=np.random.randn(H, C) / np.sqrt(H / 2.),
            b1=np.zeros(shape=(D, 1)),
            b2=np.zeros(shape=(1, H)),
            b3=np.zeros(shape=(1, C)),
            m1_neg=0.0, #np.random.uniform(high=1.0, low=0.0, size=None), # 0.001 #np.ones(shape=(D, 1)), # non-linear func/prelu1
            m2_neg=0.0, #np.random.uniform(high=1.0, low=0.0, size=None), # 0.001 #np.ones(shape=(1, H)) # prelu2
            m1_pos=0.0, #np.random.uniform(high=1.0, low=0.0, size=None), # ReLU
            m2_pos=0.0, #np.random.uniform(high=1.0, low=0.0, size=None),
            bm1 = 0.0, # mx+b non-linearity/activation/logistic reg.
            bm2 = 0.0
        )
