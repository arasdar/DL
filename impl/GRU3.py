import numpy as np
import impl.loss as loss_fun
import impl.layer as l
import impl.regularization as reg
import impl.utils as util
import impl.RNN as rnn

class GRU3(rnn.RNN):

    def __init__(self, D, H, char2idx, idx2char):
        super().__init__(D, H, char2idx, idx2char)

    def forward(self, X, h_old, train=True):
        m = self.model
        Wh, Wy = m['Wh'], m['Wy']
        bh, by = m['bh'], m['by']

        X_one_hot = np.zeros(self.D)
        X_one_hot[X] = 1.
        X_one_hot = X_one_hot.reshape(1, -1)

        # input: concat: [h, x]
        X = np.column_stack((h_old, X_one_hot))
        hh, hh_cache = l.fc_forward(X, Wh, bh)

        # gate: h_prob
        hz, hz_sigm_cache = l.sigmoid_forward(hh)

        # signal: h_pred
        hh, hh_tanh_cache = l.tanh_forward(hh)

        # output: h_next and y_pred
        h = h_old + hz * (hh - h_old)
        y, y_cache = l.fc_forward(h, Wy, by)

        cache = h_old, X, hh_cache, hz, hz_sigm_cache, hh, hh_tanh_cache, h, y_cache

        if not train:
            y = util.softmax(y)

        return y, h, cache

    def backward(self, y_pred, y_train, dh_next, cache):
        h_old, X, hh_cache, hz, hz_sigm_cache, hh, hh_tanh_cache, h, y_cache = cache

        dy = loss_fun.dcross_entropy(y_pred, y_train)

        # output: h_next and y_pred
        dh, dWy, dby = l.fc_backward(dy, y_cache)
        dh += dh_next
        dh_old1 = (1. - hz) * dh

        # signal: h_pred
        dhh = hz * dh
        dhh = l.tanh_backward(dhh, hh_tanh_cache)

        # gate: h_prob
        dhz = (hh - h_old) * dh
        dhz = l.sigmoid_backward(dhz, hz_sigm_cache)

        # input
        dhh += dhz
        dX, dWh, dbh = l.fc_backward(dhh, hh_cache)
        dh_old2 = dX[:, :self.H]

        # concat: [h, x]
        dh_next = dh_old1 + dh_old2

        grad = dict(Wh=dWh, Wy=dWy, bh=dbh, by=dby)

        return grad, dh_next

    def _init_model(self, D, C, H):
        Z = H + D

        self.model = dict(
            Wh=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wy=np.random.randn(H, D) / np.sqrt(D / 2.),
            bh=np.zeros((1, H)),
            by=np.zeros((1, D))
        )