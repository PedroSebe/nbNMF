import unittest
from nbNMF.nbNMF import nbNMF
from nbNMF.initialization import _initialize_nmf
from sklearn.exceptions import ConvergenceWarning
import numpy as np

def simulate_train_data(n, p, k, phi=100, seed=1234):
    rng = np.random.default_rng(seed)
    W = rng.random((n,k))
    H = rng.random((k,p))
    mean = W @ H
    p = phi/(mean+phi)
    return W, H, rng.negative_binomial(phi, p)

class TestNMF(unittest.TestCase):
    def test_fit_transform(self):
        print()
        n, p, k, phi = 1000, 100, 5, 10.0
        W_true, H_true, X = simulate_train_data(n, p, k, phi)
        nmf = nbNMF(5)
        W = nmf.fit_transform(X)
        assert W.min() > 0
        assert nmf.components_.min() > 0
    
    def test_inverse_transform(self):
        print()
        n, p, k, phi = 1000, 100, 5, 10.0
        W_true, H_true, X = simulate_train_data(n, p, k, phi)
        nmf = nbNMF(5)
        W = nmf.fit_transform(X)
        X_reconstructed = nmf.inverse_transform(W)
        assert X_reconstructed.min() > 0
    
    def test_convergence_warning(self):
        with self.assertWarns(ConvergenceWarning):
            A = np.ones((4, 4))
            nbNMF(2, max_iter=1).fit(A)

    def test_initialize_nn_output(self):
        # Test that initialization does not return negative values
        rng = np.random.mtrand.RandomState(42)
        data = np.abs(rng.randn(10, 10))
        for init in ("random", "nndsvd", "nndsvda", "nndsvdar"):
            W, H = _initialize_nmf(data, 10, init=init, random_state=0)
            assert not ((W < 0).any() or (H < 0).any())

    def test_initialize_close(self):
        # Test NNDSVD error
        # Test that _initialize_nmf error is less than the standard deviation of
        # the entries in the matrix.
        rng = np.random.mtrand.RandomState(42)
        A = np.abs(rng.randn(10, 10))
        W, H = _initialize_nmf(A, 10, init="nndsvd")
        error = np.linalg.norm(np.dot(W, H) - A)
        sdev = np.linalg.norm(A - A.mean())
        assert error <= sdev

    def test_initialize_variants(self):
        # Test NNDSVD variants correctness
        # Test that the variants 'nndsvda' and 'nndsvdar' differ from basic
        # 'nndsvd' only where the basic version has zeros.
        rng = np.random.mtrand.RandomState(42)
        data = np.abs(rng.randn(10, 10))
        W0, H0 = _initialize_nmf(data, 10, init="nndsvd")
        Wa, Ha = _initialize_nmf(data, 10, init="nndsvda")
        War, Har = _initialize_nmf(data, 10, init="nndsvdar", random_state=0)

        for ref, evl in ((W0, Wa), (W0, War), (H0, Ha), (H0, Har)):
            np.testing.assert_allclose(evl[ref != 0], ref[ref != 0])

    def test_nmf_transform_custom_init(self):
        # Smoke test that checks if NMF.transform works with custom initialization
        random_state = np.random.RandomState(0)
        W, H, A = simulate_train_data(6, 5, 4)
        n_components = 4
        avg = np.sqrt(A.mean() / n_components)
        H_init = np.abs(avg * random_state.randn(n_components, 5))
        W_init = np.abs(avg * random_state.randn(6, n_components))
        phi_init = np.abs(random_state.gamma(2.0, 1.0))

        m = nbNMF(n_components=n_components, init="custom", random_state=0)
        m.fit_transform(A, W=W_init, H=H_init, phi=phi_init)
        m.transform(A)

    def test_regularization(self):
        n, p, k, phi = 1000, 100, 5, 10.0
        W_true, H_true, X = simulate_train_data(n, p, k, phi)
        for alpha_W in [0.0, 5.0, 100.0]:
            for alpha_H in [0.0, 5.0, 100.0]:
                for l1_ratio in [0.0, 0.50, 1.0]:
                    nmf = nbNMF(5, alpha_W=alpha_W, alpha_H=alpha_H, l1_ratio=l1_ratio)
                    W = nmf.fit_transform(X)
                    assert W.min() >= 0
                    assert nmf.components_.min() >= 0
    

if __name__ == "__name__":
    unittest.main()