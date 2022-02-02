import numpy as np
import scipy.stats
import warnings
from initialization import _initialize_nmf
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning

class nbNMF(TransformerMixin, BaseEstimator):
    def __init__(self, n_components, init="nndsvda", max_iter=300, tol=1e-4):
        self.n_components = n_components
        self.n_components_ = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y=None):
        W, H, phi, n_iter, error = nbNMF_optimize(X, self.n_components, self.init, self.max_iter, self.tol)
        self.components_ = H
        self.phi_ = phi
        self.reconstruction_err_ = error
        if n_iter == self.max_iter:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence." % self.max_iter,
                ConvergenceWarning
            )
        return self
    
    def transform(self, X, y=None):
        W, H, phi, n_iter, error = nbNMF_optimize(X, self.n_components, self.init, self.max_iter, self.tol,
                                                    w_only=True, H=self.components_, phi=self.phi_)
        return W
    
    def fit_transform(self, X, y=None):
        W, H, phi, n_iter, error = nbNMF_optimize(X, self.n_components, self.init, self.max_iter, self.tol)
        self.components_ = H
        self.phi_ = phi
        self.reconstruction_err_ = error
        if n_iter == self.max_iter:
            warnings.warn(
                "Maximum number of iterations %d reached. Increase "
                "it to improve convergence." % self.max_iter,
                ConvergenceWarning
            )
        return W
    
    def inverse_transform(self, W):
        return W @ self.components_

    def score(self, X, y=None):
        WH = self.transform(X) @ self.components_
        return scipy.stats.nbinom.logpmf(X, self.phi_, self.phi_/(WH+self.phi_)).sum()

def update_w(X, W, H, phi):
    WH = W @ H
    num = (X/WH) @ H.T
    den = ((X+phi)/(WH+phi)) @ H.T
    return W * num / den

def update_phi(X, mean, phi):
    p = phi/(phi+mean)
    ratio = (X-1)/phi
    # calculate first and second derivatives
    fp = np.log(p) + 1 - p + np.log(1+ratio)
    fpp = ( (1-p)**2 - ratio/(1+ratio) ) / phi
    fp, fpp = fp.sum(), fpp.sum()
    # regular Newton-Raphson suggests we do phi -= fp/fpp.
    # running Newton-Raphson on log(phi) to avoid negative
    # overdispersion means we do the following multiplicative
    # update instead:
    phi *= np.exp(-fp/(fp+fpp*phi))
    return phi

def nbNMF_optimize(X, n_components, init="nndsvda", max_iter=300, tol=1e-4, w_only=False, H=None, phi=None):
    # initialization
    if w_only:
        if H is None:
            raise ValueError("When optimizing for W only (e.g. in nbNMF.transform), the components matrix H has to be supplied.")
        if phi is None:
            raise ValueError("When optimizing for W only (e.g. in nbNMF.transform), the overdispersion phi has to be supplied.")
        W = np.ones((X.shape[0],n_components))
    else:
        W, H = _initialize_nmf(X, n_components=n_components, init=init)
        phi = 10.0
    error_at_init = -scipy.stats.nbinom.logpmf(X, phi, phi/(W@H+phi)).sum()
    previous_error = error_at_init

    for n_iter in range(1, max_iter + 1):
        # update parameters
        W = update_w(X, W, H, phi)
        if not w_only:
            H = update_w(X.T, H.T, W.T, phi).T
            phi = update_phi(X, W @ H, phi)
        # test for convergence every 10 iterations
        if n_iter % 10 == 0:
            error = -scipy.stats.nbinom.logpmf(X, phi, phi/(W@H+phi)).sum()
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    return W, H, phi, n_iter, error