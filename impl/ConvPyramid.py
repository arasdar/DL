import numpy as np
import impl.layer as l

def fwd(X, w1, b1, w21, b21, w22, b22):

    # 1st layer: Linear layer of Pyramid
    h1, h1_cache = l.conv_forward(X=X, w=w1, b=b1)

    # 2nd layer: Non-linear layer of Pyramid
    h2, h21_cache = l.conv_forward(X=X, w=w21, b=b21)
    h2, nl2_cache = l.relu_forward(h2)
    h2, h22_cache = l.conv_forward(X=h2, w=w22, b=b22)

    # Output: sum NOT concat
    y = h1 + h2

    cache = h1_cache, h21_cache, nl2_cache, h22_cache
    return y, cache

def bwd(dy, cache):

    h1_cache, h21_cache, nl2_cache, h22_cache = cache

    # 2nd layer
    dh2, dw22, db22 = l.conv_backward(dout=dy, cache=h22_cache)
    dh2 = l.relu_backward(dout=dh2, cache=nl2_cache)
    dh2, dw21, db21 = l.conv_backward(dout=dh2, cache=h21_cache)

    # 1st layer
    dh1, dw1, db1 = l.conv_backward(dout=dy, cache=h1_cache)

    # Input: sum NOT concat
    dX = dh1 + dh2
    return dX, dw1, db1, dw21, db21, dw22, db22