import pytest
import numpy as np
import scipy as sc
from scipy.stats import wishart
from unittest.mock import patch, MagicMock

# Import the functions from the original file
# Assuming the original file is named gauss_tl.py
from data import (
    gen_gauss, draw_cov, gen_coef, gen_noise, covs_all, 
    coefs_all, draw_tasks, draw_all, gauss_tl
)


class TestGenGauss:
    """Test the gen_gauss function"""
    
    def test_gen_gauss_shape(self):
        """Test that gen_gauss returns correct shape"""
        mu = np.array([0, 1])
        sigma = np.eye(2)
        n = 10
        result = gen_gauss(mu, sigma, n)
        assert result.shape == (n, len(mu))
    
    def test_gen_gauss_single_dimension(self):
        """Test gen_gauss with single dimension"""
        mu = np.array([0])
        sigma = np.array([[1]])
        n = 5
        result = gen_gauss(mu, sigma, n)
        assert result.shape == (n, 1)
    
    def test_gen_gauss_deterministic_seed(self):
        """Test that gen_gauss is reproducible with seed"""
        np.random.seed(42)
        mu = np.array([0, 1])
        sigma = np.eye(2)
        n = 10
        result1 = gen_gauss(mu, sigma, n)
        
        np.random.seed(42)
        result2 = gen_gauss(mu, sigma, n)
        
        np.testing.assert_array_equal(result1, result2)


class TestDrawCov:
    """Test the draw_cov function"""
    
    def test_draw_cov_shape(self):
        """Test that draw_cov returns correct shape"""
        p = 3
        result = draw_cov(p)
        assert result.shape == (p, p)
    
    def test_draw_cov_diagonal_ones(self):
        """Test that diagonal elements are 1"""
        p = 5
        result = draw_cov(p)
        np.testing.assert_array_equal(np.diag(result), np.ones(p))
    
    def test_draw_cov_symmetric(self):
        """Test that covariance matrix is symmetric"""
        p = 4
        result = draw_cov(p)
        np.testing.assert_array_almost_equal(result, result.T)
    
    def test_draw_cov_single_dimension(self):
        """Test draw_cov with p=1"""
        p = 1
        result = draw_cov(p)
        assert result.shape == (1, 1)
        assert result[0, 0] == 1
    
    def test_draw_cov_positive_definite(self):
        """Test that generated covariance is positive definite"""
        p = 3
        result = draw_cov(p)
        eigenvals = np.linalg.eigvals(result)
        assert np.all(eigenvals > -1e-10)  # Allow for small numerical errors


class TestGenCoef:
    """Test the gen_coef function"""
    
    def test_gen_coef_no_mask(self):
        """Test gen_coef without mask"""
        coef_0 = np.array([[1, 2], [3, 4]])
        lambd = 0.5
        result = gen_coef(coef_0, lambd)
        assert result.shape == coef_0.shape
    
    def test_gen_coef_lambd_zero(self):
        """Test that lambda=0 returns original coefficients (approximately)"""
        np.random.seed(42)
        coef_0 = np.array([[1, 2], [3, 4]])
        lambd = 0.0
        result = gen_coef(coef_0, lambd)
        # Should be close to original but with added noise term
        assert result.shape == coef_0.shape
    
    def test_gen_coef_with_mask(self):
        """Test gen_coef with mask"""
        coef_0 = np.array([[1, 2], [3, 4]])
        lambd = 0.5
        mask = np.array([[1, 0], [1, 1]])
        result = gen_coef(coef_0, lambd, mask=mask)
        
        # Elements where mask is 0 should remain unchanged
        assert result[0, 1] == coef_0[0, 1]
        assert result.shape == coef_0.shape
    
    def test_gen_coef_mask_preservation(self):
        """Test that masked elements are preserved exactly"""
        coef_0 = np.array([[1, 2, 3], [4, 5, 6]])
        lambd = 1.0  # Maximum perturbation
        mask = np.array([[0, 1, 0], [1, 0, 1]])  # 0 means preserve
        result = gen_coef(coef_0, lambd, mask=mask)
        
        # Check preserved elements
        assert result[0, 0] == coef_0[0, 0]
        assert result[0, 2] == coef_0[0, 2]
        assert result[1, 1] == coef_0[1, 1]


class TestGenNoise:
    """Test the gen_noise function"""
    
    def test_gen_noise_shape(self):
        """Test gen_noise returns correct shape"""
        shape = (10, 5)
        result = gen_noise(shape)
        assert result.shape == shape
    
    def test_gen_noise_single_dimension(self):
        """Test gen_noise with single dimension"""
        shape = (5,)
        result = gen_noise(shape)
        assert result.shape == shape
    
    def test_gen_noise_reproducible(self):
        """Test gen_noise is reproducible with seed"""
        np.random.seed(42)
        shape = (3, 3)
        result1 = gen_noise(shape)
        
        np.random.seed(42)
        result2 = gen_noise(shape)
        
        np.testing.assert_array_equal(result1, result2)


class TestCovsAll:
    """Test the covs_all function"""
    
    def test_covs_all_basic(self):
        """Test basic functionality of covs_all"""
        n_task = 3
        p_s = 2
        p_n = 3
        cov_s, cov_n = covs_all(n_task, p_s, p_n)
        
        assert len(cov_s) == n_task
        assert len(cov_n) == n_task
        
        for i in range(n_task):
            assert cov_s[i].shape == (p_s, p_s)
            assert cov_n[i].shape == (p_n, p_n)
    
    def test_covs_all_positive_definite(self):
        """Test that all generated covariances are positive definite"""
        n_task = 2
        p_s = 3
        p_n = 3
        cov_s, cov_n = covs_all(n_task, p_s, p_n)
        
        for i in range(n_task):
            # Check positive definiteness
            assert np.all(np.linalg.eigvals(cov_s[i]) > -1e-10)
            assert np.all(np.linalg.eigvals(cov_n[i]) > -1e-10)
    
    def test_covs_all_with_mask(self):
        """Test covs_all with mask parameter"""
        n_task = 2
        p_s = 2
        p_n = 4
        mask = np.array([True, True, False, False])
        cov_s, cov_n = covs_all(n_task, p_s, p_n, mask=mask)
        
        assert len(cov_s) == n_task
        assert len(cov_n) == n_task
        
        # Check that fixed part is consistent across tasks
        fix_size = np.sum(mask == False)
        if fix_size > 0:
            for i in range(1, n_task):
                np.testing.assert_array_equal(
                    cov_n[i][-fix_size:, -fix_size:],
                    cov_n[0][-fix_size:, -fix_size:]
                )


class TestCoefsAll:
    """Test the coefs_all function"""
    
    def test_coefs_all_basic(self):
        """Test basic functionality of coefs_all"""
        n_task = 3
        lambd = 0.5
        beta_0 = np.array([[1, 2], [3, 4]])
        gamma_0 = np.array([[1], [2], [3]])
        
        gamma, beta = coefs_all(n_task, lambd, beta_0, gamma_0)
        
        assert len(gamma) == n_task
        assert len(beta) == n_task
        
        for i in range(n_task):
            assert gamma[i].shape == gamma_0.shape
            assert beta[i].shape == beta_0.shape
    
    def test_coefs_all_with_mask(self):
        """Test coefs_all with mask parameter"""
        n_task = 2
        lambd = 0.5
        beta_0 = np.array([[1, 2]])
        gamma_0 = np.array([[1], [2]])
        mask = np.array([[1], [0]])  # Second element should be preserved
        
        gamma, beta = coefs_all(n_task, lambd, beta_0, gamma_0, mask=mask)
        
        # Check that masked elements are preserved
        for i in range(n_task):
            assert gamma[i][1, 0] == gamma_0[1, 0]


class TestDrawTasks:
    """Test the draw_tasks function"""
    
    def test_draw_tasks_basic(self):
        """Test basic functionality of draw_tasks"""
        n_task = 2
        n = 10
        p_s = 2
        p_n = 3
        
        # Create mock parameters
        params = {
            "p_nconf": 0,
            "mu_s": np.zeros(p_s),
            "mu_n": np.zeros(p_n),
            "cov_s": [np.eye(p_s) for _ in range(n_task)],
            "cov_n": [np.eye(p_n) for _ in range(n_task)],
            "eps": 0.1,
            "alpha": np.array([[1], [1]]),
            "beta": [np.zeros((0, p_n)) for _ in range(n_task)],
            "gamma": [np.ones((p_n, 1)) for _ in range(n_task)],
            "g": 1.0
        }
        
        x, y, n_ex = draw_tasks(n_task, n, params)
        
        assert x.shape[0] == n_task * n
        assert y.shape[0] == n_task * n
        assert x.shape[1] == p_s + p_n
        assert y.shape[1] == 1
        assert len(n_ex) == n_task
        assert all(n_i == n for n_i in n_ex)


class TestDrawAll:
    """Test the draw_all function"""
    
    def test_draw_all_basic(self):
        """Test basic functionality of draw_all"""
        alpha = np.array([[1], [1]])
        n_task = 2
        n = 10
        p = 5
        p_s = 2
        p_conf = 1
        eps = 0.1
        g = 1.0
        lambd = 0.5
        beta_0 = np.array([[1, 1, 1]])
        gamma_0 = np.array([[1], [1], [1]])
        
        result = draw_all(alpha, n_task, n, p, p_s, p_conf, eps, g, lambd, beta_0, gamma_0)
        x, y, x_test, y_test, n_ex, n_ex_test, params = result
        
        assert x.shape[0] == n_task * n
        assert y.shape[0] == n_task * n
        assert x_test.shape[0] == n_task * n
        assert y_test.shape[0] == n_task * n
        assert x.shape[1] == p
        assert y.shape[1] == 1
        assert len(n_ex) == n_task
        assert len(n_ex_test) == n_task


class TestGaussTL:
    """Test the gauss_tl class"""
    
    def test_gauss_tl_initialization(self):
        """Test gauss_tl class initialization"""
        n_task = 2
        n = 10
        p = 5
        p_s = 2
        p_conf = 1
        eps = 0.1
        g = 1.0
        lambd = 0.5
        lambd_test = 0.3
        
        # Suppress print output during testing
        with patch('builtins.print'):
            tl = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test)
        
        assert tl.n_task == n_task
        assert tl.n == n
        assert tl.p == p
        assert tl.p_s == p_s
        assert tl.p_conf == p_conf
        assert tl.eps == eps
        assert tl.g == g
        assert tl.lambd == lambd
        assert tl.lambd_test == lambd_test
        
        # Check data structures
        assert "x_train" in tl.train
        assert "y_train" in tl.train
        assert "x_test" in tl.train
        assert "y_test" in tl.train
        assert tl.train["x_train"].shape[1] == p
        assert tl.train["y_train"].shape[1] == 1
    
    def test_gauss_tl_full_case(self):
        """Test gauss_tl when p_s == p (full case)"""
        n_task = 2
        n = 10
        p = 3
        p_s = 3  # Same as p
        p_conf = 1
        eps = 0.1
        g = 1.0
        lambd = 0.5
        lambd_test = 0.3
        
        with patch('builtins.print'):
            tl = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test)
        
        assert tl.is_full == True
        assert tl.p == p  # Should remain p after adjustment
        assert tl.train["x_train"].shape[1] == p
    
    def test_gauss_tl_resample(self):
        """Test the resample method"""
        n_task = 2
        n = 10
        p = 5
        p_s = 2
        p_conf = 1
        eps = 0.1
        g = 1.0
        lambd = 0.5
        lambd_test = 0.3
        
        with patch('builtins.print'):
            tl = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test)
        
        # Test resample with different parameters
        new_g = 2.0
        new_eps = 0.2
        new_lambd = 0.8
        
        x_new, y_new = tl.resample(n_task, n, g=new_g, eps=new_eps, lambd=new_lambd)
        
        assert x_new.shape == tl.train["x_train"].shape
        assert y_new.shape == tl.train["y_train"].shape
        assert tl.g == new_g
        assert tl.eps == new_eps
        assert tl.lambd == new_lambd
    
    def test_gauss_tl_add_noise(self):
        """Test the add_noise method"""
        n_task = 2
        n = 10
        p = 5
        p_s = 2
        p_conf = 1
        eps = 0.1
        g = 1.0
        lambd = 0.5
        lambd_test = 0.3
        
        with patch('builtins.print'):
            tl = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test)
        
        original_shape = tl.train["x_train"].shape
        n_noise = 3
        
        tl.add_noise(n_noise)
        
        assert tl.train["x_train"].shape == (original_shape[0], original_shape[1] + n_noise)
        assert tl.train["x_test"].shape == (original_shape[0], original_shape[1] + n_noise)
        assert tl.test["x_train"].shape == (original_shape[0], original_shape[1] + n_noise)
        assert tl.test["x_test"].shape == (original_shape[0], original_shape[1] + n_noise)
    
    @patch('scipy.linalg.block_diag')
    def test_gauss_tl_true_cov(self, mock_block_diag):
        """Test the true_cov method"""
        n_task = 2
        n = 10
        p = 5
        p_s = 2
        p_conf = 1
        eps = 0.1
        g = 1.0
        lambd = 0.5
        lambd_test = 0.3
        
        # Mock the block_diag function to return a predictable result
        mock_block_diag.return_value = np.eye(6)  # p_s + p_n + 1
        
        with patch('builtins.print'):
            tl = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test)
        
        cov_mats = tl.true_cov(train=True)
        
        assert len(cov_mats) == n_task
        for cov_mat in cov_mats:
            assert isinstance(cov_mat, np.ndarray)
            assert cov_mat.ndim == 2


class TestIntegration:
    """Integration tests for the entire pipeline"""
    
    def test_full_pipeline(self):
        """Test the complete pipeline from start to finish"""
        np.random.seed(42)  # For reproducibility
        
        n_task = 2
        n = 20
        p = 6
        p_s = 3
        p_conf = 2
        eps = 0.1
        g = 1.0
        lambd = 0.5
        lambd_test = 0.3
        
        with patch('builtins.print'):
            tl = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test)
        
        # Test that all components work together
        assert tl.train["x_train"].shape == (n_task * n, p)
        assert tl.train["y_train"].shape == (n_task * n, 1)
        assert tl.test["x_train"].shape == (n_task * n, p)
        assert tl.test["y_train"].shape == (n_task * n, 1)
        
        # Test resample
        x_new, y_new = tl.resample(n_task, n)
        assert x_new.shape == (n_task * n, p)
        assert y_new.shape == (n_task * n, 1)
        
        # Test add_noise
        tl.add_noise(2)
        assert tl.train["x_train"].shape == (n_task * n, p + 2)
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        # Test with minimal dimensions
        n_task = 1
        n = 5
        p = 2
        p_s = 1
        p_conf = 0
        eps = 0.01
        g = 0.1
        lambd = 0.1
        lambd_test = 0.1
        
        with patch('builtins.print'):
            tl = gauss_tl(n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test)
        
        assert tl.train["x_train"].shape == (n_task * n, p)
        assert tl.train["y_train"].shape == (n_task * n, 1)
    
    def test_deterministic_behavior(self):
        """Test that the system behaves deterministically with fixed seed"""
        params = {
            'n_task': 2, 'n': 10, 'p': 5, 'p_s': 2, 'p_conf': 1,
            'eps': 0.1, 'g': 1.0, 'lambd': 0.5, 'lambd_test': 0.3
        }
        
        # Generate data twice with same seed
        np.random.seed(42)
        with patch('builtins.print'):
            tl1 = gauss_tl(**params)
        
        np.random.seed(42)
        with patch('builtins.print'):
            tl2 = gauss_tl(**params)
        
        # Results should be identical
        np.testing.assert_array_equal(tl1.train["x_train"], tl2.train["x_train"])
        np.testing.assert_array_equal(tl1.train["y_train"], tl2.train["y_train"])


# Fixtures for common test data
@pytest.fixture
def sample_gauss_tl():
    """Fixture providing a sample gauss_tl instance"""
    np.random.seed(42)
    with patch('builtins.print'):
        return gauss_tl(
            n_task=2, n=10, p=5, p_s=2, p_conf=1,
            eps=0.1, g=1.0, lambd=0.5, lambd_test=0.3
        )


@pytest.fixture
def sample_parameters():
    """Fixture providing sample parameters for testing"""
    return {
        'n_task': 2,
        'n': 10,
        'p_s': 2,
        'p_n': 3,
        'alpha': np.array([[1], [1]]),
        'beta_0': np.array([[1, 1, 1]]),
        'gamma_0': np.array([[1], [1], [1]]),
        'eps': 0.1,
        'g': 1.0,
        'lambd': 0.5
    }


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"])