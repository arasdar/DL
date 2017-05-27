import numpy as np
import impl.layer as l

def fwd(X, w, b):

    # 1st layer: Linear layer of Pyramid
    h1, h1_cache = l.conv_forward(X=X, w=w, b=b)
    y = h1.copy()

    # 2nd layer: Non-linear layer of Pyramid
    h1, nl1_cache = l.relu_forward(h1)
    h2, h2_cache = l.conv_forward(X=h1, w=w, b=b)
    y += h2

    cache = h1_cache, nl1_cache, h2_cache
    return y, chache

def bwd(dy, cache):

    h1_cache, nl1_cache, h2_cache = cache

    # 2nd layer
    dh1, dw2, db2 = l.conv_backward(dout=dy, cache=h2_cache)
    dh1 = l.relu_backward(dout=dh1, cache=nl1_cache)

    #         # 2nd layer - adding to spnn output
    #         y_txhxwxd_logit += h22_txhxwxd_logit
    dh22_txhxwxd_logit = dy_txhxwxd_logit

    #         # 2nd layer - 2nd convolution
    #         h22_txhxwxd_logit, h22_txhxwxd_logit_cache = l.conv_forward(b=b22, #b_1xh
    #                                                     padding=1, #padding=true means image size stays the same: 'SAME'
    #                                                     stride=1, # stride one means include all and no jump
    #                                                     W=w22, # kernel size 3x3 for all layers
    #                                                     X=h21_txhxwxd_act) # input image in SPNN for spatial PNN

    #         # 2nd layer - non-linearity
    #         h21_txhxwxd_act, h21_txhxwxd_act_cache = l.relu_forward(h21_txhxwxd_logit)
    dh21_txhxwxd_logit = l.relu_backward(cache=h21_txhxwxd_act_cache, dout=dh21_txhxwxd_act)

    #         # 2nd layer -  non-linear term --> wf(wx)=f(x)
    #         # 2nd layer - 1st convolution
    #         h21_txhxwxd_logit, h21_txhxwxd_logit_cache = l.conv_forward(b=b21, #b_1xh
    #                                                     padding=1, #padding=true means image size stays the same: 'SAME'
    #                                                     stride=1, # stride one means include all and no jump
    #                                                     W=w21, # kernel size 3x3 for all layers
    #                                                     X=x_txcxhxw) # input image in SPNN for spatial PNN
    dx21_txcxhxw, dw21, db21 = l.conv_backward(dout=dh21_txhxwxd_logit, cache=h21_txhxwxd_logit_cache)

    #         # 1nd layer - adding to spnn output
    #         y_txhxwxd_logit = h1_txhxwxd_logit
    dh1_txhxwxd_logit = dy_txhxwxd_logit

    #         # 1st layer -  linear term --> wx=x
    #         h1_txhxwxd_logit, h1_txhxwxd_logit_cache = l.conv_forward(b=b1, #b_1xh
    #                                                     padding=1, #padding=true means image size stays the same: 'SAME'
    #                                                     stride=1, # stride one means include all and no jump
    #                                                     W=w1, # kernel size cx3x3xd for all layers
    #                                                     X=x_txcxhxw) # input image in SPNN for spatial PNN        # 2nd layer - adding to the linear layer output
    dx1_txcxhxw, dw1, db1 = l.conv_backward(dout=dh1_txhxwxd_logit, cache=h1_txhxwxd_logit_cache)
    
    # summation of both dX
    dX = dx21_txcxhxw + dx1_txcxhxw

    #         def pyrmidnet_fwd(X, w1, b1, w21, b21, w22, b22):
    return dX, dw1, db1, dw21, db21, dw22, db22