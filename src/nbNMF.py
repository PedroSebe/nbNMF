import numpy as np
import scipy.stats
import warnings
from .initialization import _initialize_nmf
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.exceptions import ConvergenceWarning

class nbNMF(TransformerMixin, BaseEstimator):
    def __init__(self, n_components, init="nndsvda", max_iter=300, tol=1e-4, random_state=None, alpha_W=0.0, alpha_H="same", l1_ratio=0.0):
        self.n_components = n_components
        self.init = init
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.alpha_W = alpha_W
        self.alpha_H = alpha_H
        self.l1_ratio = l1_ratio
    
    def fit(self, X, y=None, W=None, H=None, phi=None):
        validate_input(X)
        self.n_features_ = X.shape[1]
        self.n_components_ = self.n_components
        if self.alpha_H == "same":
            self.alpha_H = self.alpha_W
        W, H, phi, n_iter, error = nbNMF_optimize(
            X, self.n_components, self.alpha_W, self.alpha_H, self.l1_ratio,
            self.init, self.max_iter, self.tol, self.random_state, W, H, phi)
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
        check_is_fitted(self, ["components_", "phi_"])
        validate_input(X)
        W, H, phi, n_iter, error = nbNMF_optimize(X, self.n_components, self.alpha_W, self.alpha_H, self.l1_ratio, self.init, self.max_iter, self.tol, self.random_state,
                                                    w_only=True, H=self.components_, phi=self.phi_)
        if X.shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        return W
    
    def fit_transform(self, X, y=None, W=None, H=None, phi=None):
        validate_input(X)
        self.n_features_ = X.shape[1]
        self.n_components_ = self.n_components
        if self.alpha_H == "same":
            self.alpha_H = self.alpha_W
        W, H, phi, n_iter, error = nbNMF_optimize(
            X, self.n_components, self.alpha_W, self.alpha_H, self.l1_ratio,
            self.init, self.max_iter, self.tol, self.random_state, False,
            W, H, phi)
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
        check_is_fitted(self, ["components_", "phi_"])
        check_array(W)
        return W @ self.components_

    def score(self, X, y=None):
        check_is_fitted(self, ["components_", "phi_"])
        validate_input(X)
        WH = self.transform(X) @ self.components_
        return scipy.stats.nbinom.logpmf(X, self.phi_, self.phi_/(WH+self.phi_)).sum()
    
    def get_params(self, deep=True):
        return dict(
            n_components = self.n_components,
            init = self.init,
            max_iter = self.max_iter,
            tol = self.tol
        )
    
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

def update_w(X, W, H, phi, l1=0.0, l2=0.0):
    WH = W @ H
    num = (X/WH) @ H.T
    den = ((X+phi)/(WH+phi)) @ H.T
    den += l1 + l2*W
    return W * num / den

def validate_input(X):
    check_array(X)
    if (X.min() < 0) or ((X % 1).min() > 1e-4):
      raise ValueError("The nbNMF method only works for count data (non-negative integers).")

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

def nbNMF_optimize(X, n_components, alpha_W=0.0, alpha_H=0.0, l1_ratio=0.0, init="nndsvda", max_iter=500, tol=1e-9, random_state=None, w_only=False, W=None, H=None, phi=None):
    # define regularization
    l1_W = alpha_W*l1_ratio
    l2_W = alpha_W*(1-l1_ratio)
    l1_H = alpha_H*l1_ratio
    l2_H = alpha_H*(1-l1_ratio)
    # initialization
    if w_only:
        if H is None:
            raise ValueError("When optimizing for W only (e.g. in nbNMF.transform), the components matrix H has to be supplied.")
        if phi is None:
            raise ValueError("When optimizing for W only (e.g. in nbNMF.transform), the overdispersion phi has to be supplied.")
        W = np.ones((X.shape[0],n_components))
    elif init != "custom":
        W, H = _initialize_nmf(X, n_components=n_components, init=init, random_state=random_state)
        phi = 10.0
    error_at_init = -scipy.stats.nbinom.logpmf(X, phi, phi/(W@H+phi)).sum()
    previous_error = error_at_init
    error = error_at_init

    for n_iter in range(1, max_iter + 1):
        # update parameters
        W = update_w(X, W, H, phi, l1_W, l2_W)
        if not w_only:
            H = update_w(X.T, H.T, W.T, phi, l1_H, l2_H).T
            phi = update_phi(X, W @ H, phi)
        # test for convergence every 10 iterations
        if n_iter % 10 == 0:
            error = -scipy.stats.nbinom.logpmf(X, phi, phi/(W@H+phi)).sum()
            #print(f"n={n_iter}\tprogress={100*(previous_error-error)/(error_at_init-error)*100:.3f},\tphi={phi:.3f}")
            if (previous_error - error) < tol * (error_at_init-error):
                break
            previous_error = error

    return W, H, phi, n_iter, error