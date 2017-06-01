import numpy as np
import impl.loss as loss_fun
import impl.layer as l
import impl.regularization as reg
import impl.utils as util
import impl.NN as nn

class RNN(nn.NN):

    def __init__(self, D, H, char2idx, idx2char):
        self.D = D
        self.H = H
        self.char2idx = char2idx
        self.idx2char = idx2char
        self.vocab_size = len(char2idx)
        super().__init__(D, D, H, None, None, loss='cross_ent', nonlin='relu')

    def initial_state(self):
        return np.zeros((1, self.H))

    def forward(self, X, h, train=True):
        Wxh, Whh, Why = self.model['Wxh'], self.model['Whh'], self.model['Why']
        bh, by = self.model['bh'], self.model['by']

        X_one_hot = np.zeros(self.D)
        X_one_hot[X] = 1.
        X_one_hot = X_one_hot.reshape(1, -1)

        hprev = h.copy()

        h, h_cache = l.tanh_forward(X_one_hot @ Wxh + hprev @ Whh + bh)
        y, y_cache = l.fc_forward(h, Why, by)

        cache = (X_one_hot, Whh, h, hprev, y, h_cache, y_cache)

        if not train:
            y = util.softmax(y)

        return y, h, cache

    def backward(self, y_pred, y_train, dh_next, cache):
        X, Whh, h, hprev, y, h_cache, y_cache = cache

        # Softmax gradient
        dy = loss_fun.dcross_entropy(y_pred, y_train)

        # Hidden to output gradient
        dh, dWhy, dby = l.fc_backward(dy, y_cache)
        dh += dh_next
        dby = dby.reshape((1, -1))

        # tanh
        dh = l.tanh_backward(dh, h_cache)

        # Hidden gradient
        dbh = dh
        dWhh = hprev.T @ dh
        dWxh = X.T @ dh
        dh_next = dh @ Whh.T

        grad = dict(Wxh=dWxh, Whh=dWhh, Why=dWhy, bh=dbh, by=dby)

        return grad, dh_next

    def train_step(self, X_train, y_train, h):
        ys = []
        caches = []
        loss = 0.

        # Forward
        for x, y in zip(X_train, y_train):
            y_pred, h, cache = self.forward(x, h, train=True)
            loss += loss_fun.cross_entropy(self.model, y_pred, y, lam=0)
            ys.append(y_pred)
            caches.append(cache)

        loss /= X_train.shape[0]

        # Backward
        dh_next = np.zeros((1, self.H))
        grads = {k: np.zeros_like(v) for k, v in self.model.items()}

        for t in reversed(range(len(X_train))):
            grad, dh_next = self.backward(ys[t], y_train[t], dh_next, caches[t])

            for k in grads.keys():
                grads[k] += grad[k]

        for k, v in grads.items():
            grads[k] = np.clip(v, -5., 5.)

        return grads, loss, h

    def sample(self, X_seed, h, size=100):
        chars = [self.idx2char[X_seed]]
        idx_list = list(range(self.vocab_size))
        X = X_seed

        for _ in range(size - 1):
            prob, h, _ = self.forward(X, h, train=False)
            idx = np.random.choice(idx_list, p=prob.ravel())
            chars.append(self.idx2char[idx])
            X = idx

        return ''.join(chars)

    def _init_model(self, D, C, H):
        self.model = dict(
            Wxh=np.random.randn(D, H) / np.sqrt(D / 2.),
            Whh=np.random.randn(H, H) / np.sqrt(H / 2.),
            Why=np.random.randn(H, D) / np.sqrt(C / 2.),
            bh=np.zeros((1, H)),
            by=np.zeros((1, D))
        )