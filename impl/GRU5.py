import numpy as np
import impl.loss as loss_fun
import impl.layer as l
import impl.regularization as reg
import impl.utils as util
import impl.RNN as rnn

class GRU5(rnn.RNN):

    def __init__(self, D, H, char2idx, idx2char):
        super().__init__(D, H, char2idx, idx2char)

    def forward(self, X, h_old, train=True):
        m = self.model
        Wz, Wh, Wy = m['Wh'], m['Wh'], m['Wy']
        bz, bh, by = m['bh'], m['bh'], m['by']

        X_one_hot = np.zeros(self.D)
        X_one_hot[X] = 1.
        X_one_hot = X_one_hot.reshape(1, -1)

        # concat: [h, x]
        X = np.column_stack((h_old, X_one_hot))

        # gate: h_prob
        hz, hz_cache = l.fc_forward(X, Wz, bz)
        hz, hz_sigm_cache = l.sigmoid_forward(hz)

        # signal: h_pred
        hh, hh_cache = l.fc_forward(X, Wh, bh)
        hh, hh_tanh_cache = l.tanh_forward(hh)

        # h_next and y_pred
        h = X + hz * (hh - X)
        y, y_cache = l.fc_forward(h, Wy, by)

        cache = X, hz, hz_cache, hz_sigm_cache, hh, hh_cache, hh_tanh_cache, h, y_cache

        if not train:
            y = util.softmax(y)

        return y, h, cache

    def backward(self, y_pred, y_train, dh_next, cache):
        X, hz, hz_cache, hz_sigm_cache, hh, hh_cache, hh_tanh_cache, h, y_cache = cache

        dy = loss_fun.dcross_entropy(y_pred, y_train)

        # h_next and y_pred
        dh, dWy, dby = l.fc_backward(dy, y_cache)
        dh += dh_next
        dh_old1 = (1. - hz) * dh

        # signal: h_pred
        dhh = hz * dh
        dhh = l.tanh_backward(dhh, hh_tanh_cache)
        dX, dWh, dbh = l.fc_backward(dhh, hh_cache)
        dh_old2 = dX[:, :self.H]

        # gate: h_prob
        dhz = (hh - h_old) * dh
        dhz = l.sigmoid_backward(dhz, hz_sigm_cache)
        dX, dWz, dbz = l.fc_backward(dhz, hz_cache)
        dh_old3 = dX[:, :self.H]

        # concat: [h, x]
        dh_next = dh_old1 + dh_old2 + dh_old3

        grad = dict(Wz=dWz, Wh=dWh, Wy=dWy, bz=dbz, bh=dbh, by=dby)

        return grad, dh_next

    def _init_model(self, D, C, H):
        Z = H + D

        self.model = dict(
            Wz=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wh=np.random.randn(Z, H) / np.sqrt(Z / 2.),
            Wy=np.random.randn(H, D) / np.sqrt(D / 2.),
            bz=np.zeros((1, H)),
            bh=np.zeros((1, H)),
            by=np.zeros((1, D))
        )