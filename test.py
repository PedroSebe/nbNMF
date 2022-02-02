import numpy as np
from nbNMF import nbNMF

def test_data(n, p, k, phi=100, seed=1234):
    rng = np.random.default_rng(seed)
    W = rng.random((n,k))
    H = rng.random((k,p))
    mean = W @ H
    p = phi/(mean+phi)
    return W, H, rng.negative_binomial(phi, p)

W, H, X = test_data(20, 50, 5, 5)
model = nbNMF(5)
model.fit(X)
model.fit_transform(X)
model.inverse_transform(model.transform(X))
model.score(X)