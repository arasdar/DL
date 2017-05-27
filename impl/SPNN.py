import numpy as np
import impl.loss as loss_fun
import impl.layer as l
import impl.NN as nn
import impl.pyramidnet as pyramidnet

class SPNN(nn.NN):

    def __init__(self, D, C, H, lam=1e-3, p_dropout=.8, loss='cross_ent', nonlin='relu'):
        super().__init__(D, C, H, lam, p_dropout, loss, nonlin)
        
    def forward(self, X, train=False):

        # 1st layer -  convolution to change the size from x_txcxhxw to h_txhxwxd, c==channels of image, d=depth/num_units
        h1_txhxwxd_logit, h1_txhxwxd_logit_cache = l.conv_forward(b=self.model['b1'], #b_1xh
                                                    padding=1, #padding=true means image size stays the same: 'SAME'
                                                    stride=1, # stride one means include all and no jump
                                                    W=self.model['W1'], # kernel size cx3x3xd for all layers
                                                    X=X) # input image in SPNN for spatial PNN        # 2nd layer - adding to the linear layer output
        #         dX, dw1, db1 = l.conv_backward(dout=dh1_txhxwxd_logit, cache=h1_txhxwxd_logit_cache)

        # 1st layer - nonlinearity-relu
        h1_txhxwxd_act, h1_txhxwxd_act_cache = l.relu_forward(h1_txhxwxd_logit)
        #         dh1_txhxwxd_logit = l.relu_backward(cache=h1_txhxwxd_act_cache, dout=dh1_txhxwxd_act)

        # midst layer: Pyrmidnet as core layer
        h1_txhxwxd_act, h1_txhxwxd_act_pyrmidnet_cache = pyramidnet.fwd(
                    b1=self.model['b11'], b21=self.model['b21'], b22=self.model['b22'], 
                      w1=self.model['W11'], w21=self.model['W21'], w22=self.model['W22'], 
                      X=h1_txhxwxd_act)
        #         dh1_txhxwxd_act, dw11, db11, dw21, db21, dw22, db22 = pyrmidnet_bwd(cache=h1_txhxwxd_act_pyrmidnet_cache, 
        #                                                                           dy_txhxwxd_act=dh1_txhxwxd_act)

        # midst layer: Pyrmidnet as core layer 2
        h1_txhxwxd_act, h1_txhxwxd_act_pyrmidnet_cache_2 = pyramidnet.fwd(
                    b1=self.model['b11_2'], b21=self.model['b21_2'], b22=self.model['b22_2'], 
                      w1=self.model['W11_2'], w21=self.model['W21_2'], w22=self.model['W22_2'], 
                      X=h1_txhxwxd_act)
        #         dh1_txhxwxd_act, dw11_2, db11_2, dw21_2, db21_2, dw22_2, db22_2 = pyrmidnet_bwd(cache=h1_txhxwxd_act_pyrmidnet_cache_2, 
        #                                                                           dy_txhxwxd_act=dh1_txhxwxd_act)

        # midst layer: Pyrmidnet as core layer 3
        h1_txhxwxd_act, h1_txhxwxd_act_pyrmidnet_cache_3 = pyramidnet.fwd(
                    b1=self.model['b11_3'], b21=self.model['b21_3'], b22=self.model['b22_3'], 
                      w1=self.model['W11_3'], w21=self.model['W21_3'], w22=self.model['W22_3'], 
                      X=h1_txhxwxd_act)
        #         dh1_txhxwxd_act, dw11_3, db11_3, dw21_3, db21_3, dw22_3, db22_3 = pyrmidnet_bwd(cache=h1_txhxwxd_act_pyrmidnet_cache_3, 
        #                                                                           dy_txhxwxd_act=dh1_txhxwxd_act)

        # last layer : FC layer -  fully connected to the output layer (visible layer)
        # n=hxwxd flattened
        h1_txn_act = h1_txhxwxd_act.reshape([h1_txhxwxd_act_cache.shape[0], -1])
        #         dh1_txhxwxd_act = dh1_txn_act.reshape(h1_txhxwxd_act_cache.shape)

        y_tx10_logit, y_tx10_logit_cache = l.fc_forward(X=h1_txn_act, W=self.model['W2'], b=self.model['b2'])
        #         dh1_txn_act, dw2, db2 = l.fc_backward(dout=dy_tx10_logit, cache=y_tx10_logit_cache)

        # Output
        cache = h1_txhxwxd_logit_cache, h1_txhxwxd_act_cache, h1_txhxwxd_act_pyrmidnet_cache, h1_txhxwxd_act_pyrmidnet_cache_2, h1_txhxwxd_act_pyrmidnet_cache_3, y_tx10_logit_cache
        #         h1_txhxwxd_logit_cache, h1_txhxwxd_act_cache, h1_txhxwxd_act_pyrmidnet_cache, h1_txhxwxd_act_pyrmidnet_cache_2, h1_txhxwxd_act_pyrmidnet_cache_3, y_tx10_logit_cache = cache

        return y_tx10_logit, cache

    def backward(self, y, y_train, cache):

        #         # Output
        #         cache = h1_txhxwxd_logit_cache, h1_txhxwxd_act_cache, h1_txhxwxd_act_pyrmidnet_cache, h1_txhxwxd_act_pyrmidnet_cache_2, y_tx10_logit_cache
        h1_txhxwxd_logit_cache, h1_txhxwxd_act_cache, h1_txhxwxd_act_pyrmidnet_cache, h1_txhxwxd_act_pyrmidnet_cache_2, h1_txhxwxd_act_pyrmidnet_cache_3, y_tx10_logit_cache = cache

        # Output layer backward
        dy_tx10_logit = self.dloss_funs[self.loss](y, y_train) # y==y_logits

        #         y_tx10_logit, y_tx10_logit_cache = l.fc_forward(X=h1_txn_act, W=self.model['W2'], b=self.model['b2'])
        dh1_txn_act, dw2, db2 = l.fc_backward(dout=dy_tx10_logit, cache=y_tx10_logit_cache)

        #         # last layer : FC layer -  fully connected to the output layer (visible layer)
        #         # n=hxwxd flattened
        #         h1_txn_act = h1_txhxwxd_act.reshape([h1_txhxwxd_act_cache.shape[0], -1])
        dh1_txhxwxd_act = dh1_txn_act.reshape(h1_txhxwxd_act_cache.shape)

        #         # midst layer: Pyrmidnet as core layer
        #         h1_txhxwxd_act, h1_txhxwxd_act_pyrmidnet_cache = pyrmidnet_fwd(
        #                     b1=self.model['b11'], b21=self.model['b21'], b22=self.model['b22'], 
        #                       w1=self.model['w11'], w21=self.model['w21'], w22=self.model['w22'], 
        #                       X=h1_txhxwxd_act)
        dh1_txhxwxd_act, dw11, db11, dw21, db21, dw22, db22 = pyramidnet.bwd(cache=h1_txhxwxd_act_pyrmidnet_cache, 
                                                                          dy_txhxwxd_act=dh1_txhxwxd_act)
        #         # midst layer: Pyrmidnet as core layer 2
        #         h1_txhxwxd_act, h1_txhxwxd_act_pyrmidnet_cache_2 = pyramidnet.fwd(
        #                     b1=self.model['b11_2'], b21=self.model['b21_2'], b22=self.model['b22_2'], 
        #                       w1=self.model['W11_2'], w21=self.model['W21_2'], w22=self.model['W22_2'], 
        #                       X=h1_txhxwxd_act)
        dh1_txhxwxd_act, dw11_2, db11_2, dw21_2, db21_2, dw22_2, db22_2 = pyramidnet.bwd(cache=h1_txhxwxd_act_pyrmidnet_cache_2, 
                                                                          dy_txhxwxd_act=dh1_txhxwxd_act)

        #         # midst layer: Pyrmidnet as core layer 3
        #         h1_txhxwxd_act, h1_txhxwxd_act_pyrmidnet_cache_2 = pyramidnet.fwd(
        #                     b1=self.model['b11_2'], b21=self.model['b21_2'], b22=self.model['b22_2'], 
        #                       w1=self.model['W11_2'], w21=self.model['W21_2'], w22=self.model['W22_2'], 
        #                       X=h1_txhxwxd_act)
        dh1_txhxwxd_act, dw11_3, db11_3, dw21_3, db21_3, dw22_3, db22_3 = pyramidnet.bwd(cache=h1_txhxwxd_act_pyrmidnet_cache_3, 
                                                                          dy_txhxwxd_act=dh1_txhxwxd_act)

        #         # 1st layer - nonlinearity-relu
        #         h1_txhxwxd_act, h1_txhxwxd_act_cache = l.relu_forward(h1_txhxwxd_logit)
        dh1_txhxwxd_logit = l.relu_backward(cache=h1_txhxwxd_act_cache, dout=dh1_txhxwxd_act)

        #         # 1st layer -  convolution to change the size from x_txcxhxw to h_txhxwxd, c==channels of image, d=depth/num_units
        #         h1_txhxwxd_logit, h1_txhxwxd_logit_cache = l.conv_forward(b=self.model['b1'], #b_1xh
        #                                                     padding=1, #padding=true means image size stays the same: 'SAME'
        #                                                     stride=1, # stride one means include all and no jump
        #                                                     W=self.model['W1'], # kernel size cx3x3xd for all layers
        #                                                     X=X) # input image in SPNN for spatial PNN        # 2nd layer - adding to the linear layer output
        dX, dw1, db1 = l.conv_backward(dout=dh1_txhxwxd_logit, cache=h1_txhxwxd_logit_cache)

        # grad for GD
        grad = dict(
            # Input layer to Conv: 1st layer in SPNN: Conv layer from the input
            W1=dw1, 
            b1=db1,
            #             W1=np.random.randn(H, 1, 3, 3) / np.sqrt(H / 2.),
            #             b1=np.zeros((H, 1)),

            # Pyrmidnet layer: midst layer
            W11=dw11, 
            b11=db11,
            #             W11=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            #             b11=np.zeros((H, 1)),
            W21=dw21, 
            b21=db21,
            #             W21=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            #             b21=np.zeros((H, 1)), # 1st layer Conv the input
            W22=dw22, 
            b22=db22, # 1st layer in SPNN: Conv layer from the input
            #             W22=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            #             b22=np.zeros((H, 1)), # 1st layer Conv the input

            # Pyrmidnet layer: midst layer 2
            W11_2=dw11_2, 
            b11_2=db11_2,
            #             W11_2=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            #             b11_2=np.zeros((H, 1)),
            W21_2=dw21_2, 
            b21_2=db21_2,
            #             W21_2=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            #             b21_2=np.zeros((H, 1)), # 1st layer Conv the input
            W22_2=dw22_2, 
            b22_2=db22_2, # 1st layer in SPNN: Conv layer from the input
            #             W22_2=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            #             b22_2=np.zeros((H, 1)), # 1st layer Conv the input

            # Pyrmidnet layer: midst layer 3
            W11_3=dw11_3, 
            b11_3=db11_3,
            #             W11_2=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            #             b11_2=np.zeros((H, 1)),
            W21_3=dw21_3, 
            b21_3=db21_3,
            #             W21_2=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            #             b21_2=np.zeros((H, 1)), # 1st layer Conv the input
            W22_3=dw22_3, 
            b22_3=db22_3, # 1st layer in SPNN: Conv layer from the input
            #             W22_2=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            #             b22_2=np.zeros((H, 1)), # 1st layer Conv the input

            #             # FC to output layer: last layer in SPNN: FC layer to the output 
            W2=dw2, 
            b2=db2
            #             W2=np.random.randn(H * D, C) / np.sqrt(H * D / 2.), 
            #             b2=np.zeros((1, C))
            )
        
        return grad

    def _init_model(self, D, C, H):
        self.model = dict(
            #             # Input layer to Conv: 1st layer in SPNN: Conv layer from the input
            #             W1=dw1, 
            #             b1=db1,
            W1=np.random.randn(H, 1, 3, 3) / np.sqrt(H / 2.),
            b1=np.zeros((H, 1)),

            #             # Pyrmidnet layer: midst layer
            #             W11=dw11, 
            #             b11=db11,
            W11=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            b11=np.zeros((H, 1)),
            #             W21=dw21, 
            #             b21=db21,
            W21=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            b21=np.zeros((H, 1)), # 1st layer Conv the input
            #             W22=dw22, 
            #             b22=db22, # 1st layer in SPNN: Conv layer from the input
            W22=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            b22=np.zeros((H, 1)), # 1st layer Conv the input

            #             # Pyrmidnet layer: midst layer 2
            #             W11_2=dw11_2, 
            #             b11_2=db11_2,
            W11_2=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            b11_2=np.zeros((H, 1)),
            #             W21_2=dw21_2, 
            #             b21_2=db21_2,
            W21_2=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            b21_2=np.zeros((H, 1)), # 1st layer Conv the input
            #             W22_2=dw22_2, 
            #             b22_2=db22_2, # 1st layer in SPNN: Conv layer from the input
            W22_2=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            b22_2=np.zeros((H, 1)), # 1st layer Conv the input

            #             # Pyrmidnet layer: midst layer 3
            #             W11_2=dw11_2, 
            #             b11_2=db11_2,
            W11_3=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            b11_3=np.zeros((H, 1)),
            #             W21_2=dw21_2, 
            #             b21_2=db21_2,
            W21_3=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            b21_3=np.zeros((H, 1)), # 1st layer Conv the input
            #             W22_2=dw22_2, 
            #             b22_2=db22_2, # 1st layer in SPNN: Conv layer from the input
            W22_3=np.random.randn(H, H, 3, 3) / np.sqrt(H / 2.),
            b22_3=np.zeros((H, 1)), # 1st layer Conv the input

            #             #             # FC to output layer: last layer in SPNN: FC layer to the output 
            #             W2=dw2, 
            #             b2=db2
            W2=np.random.randn(H * D, C) / np.sqrt(H * D / 2.), 
            b2=np.zeros((1, C))
            )