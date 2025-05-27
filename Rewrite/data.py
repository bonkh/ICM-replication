import numpy as np
from scipy.stats import wishart

# Generate multivariate Gaussian samples
def gen_gauss(mu, sigma, n):
  return np.random.multivariate_normal(mu, sigma, n)

# Generate covariance matrix 
def draw_cov(p):
  scale = np.random.normal(0,1,(p,p))
  scale = np.dot(scale.T, scale)

  # Create a Wishart distributed covariance matrix
  if p == 1:
    cov = scale
  else:
    cov = wishart.rvs(df=p, scale=scale)

  #Normalize covariance matrix
  for i in range(p):
    for j in range(p):
      if i == j: continue
      cov[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])

  np.fill_diagonal(cov,1) 
  return cov

# Generate task-specific coefficients for the non-causal features
def gen_coef(coef_0, lambd, mask = None):
  """
  Generate coefficients for the non-causal features.
  coef_0: Initial coefficients
  lambd: Regularization parameter
  mask: Mask for coefficients (optional)
  """

  if mask is None:
    return (1-lambd)*coef_0 + lambd*np.random.normal(0,1,coef_0.shape)

  mask_compl = ((mask+1)%2).astype(bool)
  draw = np.random.normal(0,1,coef_0.shape)
  ret = (1-lambd)*coef_0 + lambd*draw
  ret[mask_compl] = coef_0[mask_compl]
  return ret

# Generate noise samples that follow a normal distribution
def gen_noise(shape):
  return np.random.normal(0,1,shape)

# Generate covariance matrices for all tasks
def covs_all(n_task, p_s, p_n, mask = None):
  cov_s, cov_n = [], []
  fix = -1
  ref = None

  # If a mask is provided, find the fixed features
  if not mask is None:
    fix_mask = np.where(mask == False)[0]
    if len(fix_mask) > 0:
      fix = fix_mask.size
      ref = draw_cov(fix)
  
  for k in range(n_task):
    # Generate covariance matrix for invariant features
    cov_s.append(draw_cov(p_s))

    # Generate covariance matrix for non-invariant features
    cov_n_k = draw_cov(p_n)

    # If a mask is provided, fill the fixed features with the reference covariance
    if fix > 0:
      cov_n_k[-fix:, -fix:] = ref 

    # Ensure the covariance matrix is positive definite
    # As fixed variables may lead to non-positive definite matrices
    # Which is required for covariance matrices
    eig = np.linalg.eig(cov_n_k)
    if not np.all(eig[0]>0):
      pd = False
      max_iter = 100
      it = 0
      while (not pd) and it<max_iter:
        it += 1
        samp = np.random.normal(0,1,(fix, p_n-fix))
        if np.any(np.array(samp.shape) == 1):
          samp = samp.flatten()
        
        if fix == 1:
          cov_n_k[-fix,0:p_n-fix] = samp.flatten()
          cov_n_k[0:p_n-fix, -fix] = samp.flatten()
        elif fix == p_n-1:
          cov_n_k[-fix:,p_n-fix-1] = samp.flatten()
          cov_n_k[p_n-fix-1, -fix:] = samp.flatten()
        else:
          cov_n_k[-fix:,0:p_n-fix] = samp
          cov_n_k[0:p_n-fix, -fix:] = samp.T

        pd = np.all(np.linalg.eig(cov_n_k)[0]>0) 

    cov_n.append(cov_n_k)

  return cov_s, cov_n

# Generate coefficients for non-invariant features across all tasks
def coefs_all(n_task, lambd, beta_0, gamma_0, mask = None):
  beta, gamma = [], []
  for k in range(n_task):
    gamma.append(gen_coef(gamma_0, lambd, mask = mask))
    beta.append(gen_coef(beta_0, lambd))
  return gamma, beta

# Generate data for all tasks
def draw_tasks(n_task, n, params):
  p_nconf = params['p_nconf'] # Number of confounding features
  mu_s = params['mu_s'] # Mean for invariant features
  mu_n = params['mu_n'] # Mean for non-invariant features
  cov_s = params['cov_s'] # Covariance for invariant features
  cov_n = params['cov_n'] # Covariance for non-invariant features
  eps = params['eps'] # Noise level
  alpha = params['alpha'] # Coefficients for invariant features
  beta = params['beta'] # Coefficients for non-invariant features
  gamma = params['gamma'] # Coefficients for non-invariant features across tasks
  g = params['g'] # Scaling factor for non-invariant features
  x, y, n_ex = [], [], [] # Combined features, target and number of samples per tasks

  for k in range(n_task):
    # Generate invariant features
    xs_k = (gen_gauss(mu_s, cov_s[k], n)) 

    # Generate target feature
    eps_draw = gen_noise((n,1))
    y_k     = np.dot(xs_k, alpha) + eps*eps_draw

    # Generate non-invariant features 
    gamma_k = gamma[k]
    noise_k = (g*gen_gauss(mu_n, cov_n[k], n))
    xn_k = np.dot(y_k, gamma_k.T) + noise_k
    beta_k  = beta[k]
    if p_nconf > 0:
      xn_k += np.dot(xs_k[:,p_nconf:], beta_k)

    # Create dataset for task k
    x.append(np.concatenate([xs_k, xn_k], 1))
    y.append(y_k)
    n_ex.append(n)

  return np.concatenate(x,0), np.concatenate(y,0), n_ex

# Master function to draw all tasks and return the generated data
def draw_all(alpha, n_task, n, p, p_s, p_conf, eps, g, lambd,beta_0, gamma_0, mask = None):
  p_n = p - p_s
  mu_s = np.zeros(p_s)
  mu_n = np.zeros(p_n)
  cov_s, cov_n = covs_all(n_task, p_s, p_n, mask = mask)
  gamma, beta = coefs_all(n_task, lambd, beta_0, gamma_0, mask = mask)
  params = {'mu_s':mu_s, 'mu_n' : mu_n, 'cov_s' : cov_s, 'cov_n' : cov_n, 
            'eps':eps, 'g':g,
            'alpha' : alpha, 'beta': beta, 'gamma': gamma, 'p_nconf' : (p_s-p_conf)}
  x, y, n_ex = draw_tasks(n_task, n, params)
  # x_test, y_test, n_ex_test = draw_tasks(n_task, n, params)

  return x, y, n_ex, params
  # return x, y, x_test, y_test, n_ex, n_ex_test, params

# Class to create synthetic data for experiments
class gauss_tl(object):
  """
  Class for synthetic data experiments.
  """

  # Initialize the class with parameters for the synthetic data generation.
  # Use at the start of experiment
  def __init__(self, n_task, n, p, p_s, p_conf, eps, g, lambd, lambd_test, mask = None):

    # Handle the case where all features are invariant (by adding one more feature)
    if p_s == p:
      p = p+1
      self.is_full = True
    else:
      self.is_full = False

    alpha = gen_coef(np.random.normal(0,1,(p_s,1)),0)

    p_n = p - p_s
    gamma_0 = np.random.normal(0,1,(p_n,1))
    beta_0 = np.random.normal(0,1,(p_conf,p_n))

    # Generate data for training 
    x, y, n_ex, params = draw_all(alpha, n_task, n, p, p_s, p_conf, 
                                  eps, g, lambd, beta_0, gamma_0, mask = mask)

    # Generate data for testing (diff lambda_test)
    xt, yt, n_ext, params_test = draw_all(alpha, n_task, n, p, p_s, p_conf, 
                                          eps, g, lambd_test, beta_0, gamma_0, mask = mask)

    # If the number of features is full (all invariant), remove the last feature
    if self.is_full:
      x = x[:,0:-1]
      x_test = x_test[:,0:-1]
      xt = xt[:,0:-1]
      x_tt = x_tt[:,0:-1]
      self.p = p-1
      self.alpha = alpha[:-1]
    else:
      self.p = p
      self.alpha = alpha

    # Store parameters 
    self.p_s = p_s
    self.p_conf = p_conf
    self.train = {}
    self.train['x_train'] = x
    self.train['y_train'] = y
    self.train['n_ex'] = np.array(n_ex)
    self.train['cov_s'] = params['cov_s']
    self.train['cov_n'] = params['cov_n']
    self.train['eps'] = params['eps']
    self.train['gamma'] = params['gamma']
    self.train['beta'] = params['beta']
    self.lambd = lambd
    self.lambd_test = lambd_test
    self.g = g
    self.eps = eps

    self.test = {}
    self.test['x_test'] = xt
    self.test['y_test'] = yt
    self.test['n_ex'] = np.array(n_ext)
    self.test['cov_s'] = params_test['cov_s']
    self.test['cov_n'] = params_test['cov_n']
    self.test['eps'] = params_test['eps']
    self.test['gamma'] = params_test['gamma']
    self.test['beta'] = params_test['beta']
    self.gamma_0 = gamma_0
    self.beta_0 = beta_0
    self.n_task = n_task
    self.n = n

  # Use when resampling for each repeat
  # Data resampled is different from the one used in the initialization
  # With differnt causal relationships 
  def resample(self,g = None, lambd = None, eps = None, noise = 0, mask = None):
    if g is None: g = self.g
    if eps is None: eps = self.eps
    if lambd is None: lambd = self.lambd

    # Generate new coefficients for the invariant features
    alpha = gen_coef(np.random.normal(0,1,(self.p_s,1)),0)

    xt, yt, n_ext, params = draw_all(alpha, 
                                                  self.n_task, self.n, 
                                                  self.p, self.p_s, self.p_conf, 
                                                  eps, g,lambd, 
                                                  self.beta_0, self.gamma_0, 
                                                  mask = mask)

    
    xt_test, yt_test, n_ext, params_test = draw_all(alpha, 
                                                  self.n_task, self.n, 
                                                  self.p, self.p_s, self.p_conf, 
                                                  eps, g,self.lambd_test, 
                                                  self.beta_0, self.gamma_0, 
                                                  mask = mask)
    
    # Add noise feature(s) to the features if specified
    if noise> 0:
      xt = np.append(xt, np.random.normal(0,1,(xt.shape[0],noise)),1)

    self.alpha = alpha
    self.train['x_train'] = xt
    self.train['y_train'] = yt
    self.train['n_ex'] = np.array(n_ext)
    self.train['cov_s'] = params['cov_s']
    self.train['cov_n'] = params['cov_n']
    self.train['eps'] = params['eps']
    self.train['gamma'] = params['gamma']
    self.train['beta'] = params['beta']

    self.test['x_test'] = xt_test
    self.test['y_test'] = yt_test
    self.test['cov_s'] = params_test['cov_s']
    self.test['cov_n'] = params_test['cov_n']
    self.test['eps'] = params_test['eps']
    self.test['gamma'] = params_test['gamma']
    self.test['beta'] = params_test['beta']
    self.lambd = lambd
    self.g = g
    self.eps = eps

    return xt, yt

