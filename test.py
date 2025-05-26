import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import wishart
import seaborn as sns

def gen_coef(coef_0, lambd, mask = None):
  if not mask is None:
    mask_compl = ((mask+1)%2).astype(bool)
    draw = np.random.normal(0,1,coef_0.shape)
    ret = (1-lambd)*coef_0 + lambd*draw
    ret[mask_compl] = coef_0[mask_compl]
    return ret
  else:
    return (1-lambd)*coef_0 + lambd*(np.random.normal(0,1,coef_0.shape) + np.random.normal(0,1))

# First, let's implement the necessary helper functions
def gen_gauss(mu, sigma, n):
    """Generate multivariate Gaussian samples"""
    return np.random.multivariate_normal(mu, sigma, n)

def gen_noise(shape):
    """Generate Gaussian noise"""
    return np.random.normal(0, 1, shape)

def draw_cov(p):
    """Generate a random covariance matrix"""
    scale = np.random.normal(0, 1, (p, p))
    scale = np.dot(scale.T, scale)
    if p == 1:
        cov = scale
    else:
        cov = wishart.rvs(df=p, scale=scale)
    
    # Normalize covariance matrix
    for i in range(p):
        for j in range(p):
            if i == j: 
                continue
            cov[i,j] = cov[i,j]/np.sqrt(cov[i,i]*cov[j,j])
    
    np.fill_diagonal(cov, 1) 
    return cov

def draw_tasks(n_task, n, params):
    """
    Generate synthetic multi-task data with confounding relationships
    
    Parameters:
    - n_task: number of tasks
    - n: number of samples per task
    - params: dictionary with generation parameters
    """
    
    p_nconf = params['p_nconf']
    mu_s = params['mu_s']
    mu_n = params['mu_n']
    cov_s = params['cov_s']
    cov_n = params['cov_n']
    eps = params['eps']
    alpha = params['alpha']
    beta = params['beta']
    gamma = params['gamma']
    g = params['g']
    
    x, y, n_ex = [], [], []

    for k in range(n_task):
        print(f"\n=== Generating Task {k+1} ===")
        
        # Step 1: Generate signal features
        xs_k = gen_gauss(mu_s, cov_s[k], n)
        print(f"Signal features shape: {xs_k.shape}")
        print(f"Signal features mean: {np.mean(xs_k, axis=0)}")
        
        # Step 2: Generate target variable
        eps_draw = gen_noise((n, 1))
        y_k = np.dot(xs_k, alpha)# + eps * eps_draw
        print(f"Target variable shape: {y_k.shape}")
        print(f"Target variable mean: {np.mean(y_k):.3f}")
        
        # Step 3: Generate noise/confounding features
        gamma_k = gamma[k]
        noise_k = g * gen_gauss(mu_n, cov_n[k], n)
        xn_k = np.dot(y_k, gamma_k) + noise_k
        print(f"Noise features shape: {xn_k.shape}")
        print(f"Gamma coefficients for task {k+1}: {gamma_k.flatten()}")
        
        # Step 4: Add confounding from signal features (if applicable)
        beta_k = beta[k]
        if p_nconf > 0:
            xn_k += np.dot(xs_k[:, p_nconf:], beta_k)
            print(f"Added confounding from signal features")
            print(f"Beta coefficients: {beta_k}")
        
        # Step 5: Combine features
        x_k = np.concatenate([xs_k, xn_k], 1)
        print(f"Final feature matrix shape: {x_k.shape}")
        
        x.append(x_k)
        y.append(y_k)
        n_ex.append(n)

    return np.concatenate(x, 0), np.concatenate(y, 0), n_ex

def demonstrate_draw_tasks(config=None):
    """Demonstrate draw_tasks with a config-driven setup"""
    
    print("=" * 60)
    print("DRAW_TASKS FUNCTION DEMONSTRATION")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Use passed config or default to experiment-style config
    if config is None:
        config = {
            'n_task': 7,
            'n': 4000,
            'p': 6,
            'p_s': 3,
            'p_conf': 1,
            'eps': 1,
            'g': 1,
            'lambd': 0.5,
            'lambd_test': 0.99,
            'max_l': 100,
            'n_ul': 100
        }
    
    n_task = config['n_task']
    n = config['n']
    p_s = config['p_s']
    p_n = config['p'] - p_s
    p_conf = config['p_conf']
    eps = config['eps']
    g = config['g']
    p_nconf = p_s - p_conf

    print(f"Setup:")
    print(f"- Number of tasks: {n_task}")
    print(f"- Samples per task: {n}")
    print(f"- Signal features (p_s): {p_s}")
    print(f"- Noise features (p_n): {p_n}")
    print(f"- Confounding signal features (p_conf): {p_conf}")
    
    # Generate parameters
    mu_s = np.zeros(p_s)
    mu_n = np.zeros(p_n)
    cov_s = [draw_cov(p_s) for _ in range(n_task)]
    cov_n = [draw_cov(p_n) for _ in range(n_task)]
    print(f"\nCovariance matrices for signal features:", cov_s)
    print(f"Covariance matrices for noise features:", cov_n)
    
    # Causal coefficients (signal → target), same across tasks
    # alpha = gen_coef(np.random.normal(0,1,(p_s,1)),0)
    alpha = np.array([[1], [1], [0.1]])  # Small, controlled coefficients

    print(f"\nTrue causal coefficients (alpha): {alpha.flatten()}")

    # Task-specific confounding (target → noise)
    gamma = [np.random.uniform(-1, 1, (1, p_n)) for _ in range(n_task)]

    # Task-specific signal → noise confounding
    beta = [np.random.uniform(-0.5, 0.5, (p_conf, p_n)) for _ in range(n_task)]
    
    # Assemble parameter dictionary
    params = {
        'p_nconf': p_nconf,
        'mu_s': mu_s,
        'mu_n': mu_n,
        'cov_s': cov_s,
        'cov_n': cov_n,
        'eps': eps,
        'alpha': alpha,
        'beta': beta,
        'gamma': gamma,
        'g': g
    }
    
    print(f"\nConfounding structure:")
    for k in range(n_task):
        print(f"Task {k+1} - Target→Noise gamma: {gamma[k].flatten()}")
    
    # Generate the data
    X, Y, n_ex = draw_tasks(n_task, n, params)
    
    print(f"\n" + "="*40)
    print("RESULTS:")
    print(f"Combined feature matrix shape: {X.shape}")
    print(f"Combined target vector shape: {Y.shape}")
    print(f"Samples per task: {n_ex}")
    
    # Split for downstream use
    task_data = []
    start = 0
    for samples in n_ex:
        end = start + samples
        task_data.append((X[start:end, :], Y[start:end]))
        start = end
    
    return X, Y, n_ex, params, task_data

def analyze_confounding(X, Y, n_ex, params):
    """Analyze the confounding relationships in generated data"""
    
    print(f"\n" + "="*60)
    print("CONFOUNDING ANALYSIS")
    print("="*60)
    
    n = n_ex[0]
    p_s = len(params['mu_s'])
    
    # Split into tasks
    task1_X, task1_Y = X[:n, :], Y[:n]
    task2_X, task2_Y = X[n:, :], Y[n:]
    
    for task_idx, (task_X, task_Y) in enumerate([(task1_X, task1_Y), (task2_X, task2_Y)]):
        print(f"\nTask {task_idx + 1} Analysis:")
        
        # Split features
        signal_features = task_X[:, :p_s]
        noise_features = task_X[:, p_s:]
        
        # Correlation analysis
        print(f"Signal features → Target correlations:")
        for i in range(p_s):
            corr = np.corrcoef(signal_features[:, i], task_Y.flatten())[0, 1]
            print(f"  Signal {i+1}: {corr:.3f}")
        
        print(f"Noise features → Target correlations:")
        for i in range(noise_features.shape[1]):
            corr = np.corrcoef(noise_features[:, i], task_Y.flatten())[0, 1]
            print(f"  Noise {i+1}: {corr:.3f}")
        
        # Expected correlations based on gamma coefficients
        gamma_k = params['gamma'][task_idx]
        print(f"Expected noise correlations (from gamma): {gamma_k.flatten()}")

def visualize_relationships(X, Y, n_ex, config=None, show_interactions=False):
    """
    Visualize the relationships between features and target for multi-task data.
    
    Parameters:
    - X: Feature matrix (n_samples, n_features)
    - Y: Target vector (n_samples,)
    - n_ex: List of number of examples per task
    - config: Dict with keys 'p' (total features), 'p_s' (signal), 'p_n' (noise)
    - show_interactions: If True, show pairwise feature interaction plots
    """
    
    if config is None:
        config = {'p': X.shape[1], 'p_s': X.shape[1] // 2}

    p = config['p']
    p_s = config['p_s']
    p_n = p - p_s
    n_task = len(n_ex)
    split_indices = np.cumsum([0] + n_ex)

    # Set up color palette for tasks
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_task, 3)))
    
    # ========== 2. Feature-Target Relationships Grid ==========
    # Determine grid layout for features
    n_cols = min(4, p)  # Max 4 columns for readability
    n_rows = int(np.ceil(p / n_cols))
    
    fig2, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig2.suptitle("Feature-Target Relationships Across All Tasks", fontsize=16, fontweight='bold')
    
    # Handle single row/column cases
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    for feat_idx in range(p):
        row = feat_idx // n_cols
        col = feat_idx % n_cols
        
        if row >= n_rows or col >= n_cols:
            break
            
        ax = axes[row, col]
        
        # Plot all tasks on the same subplot for comparison
        for task_idx in range(n_task):
            start, end = split_indices[task_idx], split_indices[task_idx + 1]
            task_X, task_Y = X[start:end], Y[start:end]
            
            # Scatter plot with transparency
            ax.scatter(task_X[:, feat_idx], task_Y, alpha=0.3, 
                      color=colors[task_idx], s=1, label=f'Task {task_idx+1}')
            
            # Add regression line
            x_data = task_X[:, feat_idx].flatten()
            y_data = task_Y.flatten()
            
            if len(x_data) > 1 and not np.all(x_data == x_data[0]):
                try:
                    z = np.polyfit(x_data, y_data, 1)
                    p_reg = np.poly1d(z)
                    x_line = np.linspace(x_data.min(), x_data.max(), 100)
                    ax.plot(x_line, p_reg(x_line), "--", alpha=0.8, 
                           color=colors[task_idx], linewidth=1.5)
                except np.linalg.LinAlgError:
                    pass
        
        # Feature labeling
        is_signal = feat_idx < p_s
        feature_type = "Signal" if is_signal else "Noise"
        feature_name = f"Signal_{feat_idx+1}" if is_signal else f"Noise_{feat_idx-p_s+1}"
        
        ax.set_xlabel(f'{feature_name}')
        ax.set_ylabel('Target')
        ax.set_title(f'{feature_name} ({feature_type})')
        ax.grid(True, alpha=0.3)
        
        # Add legend only to first subplot
        if feat_idx == 0:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Hide unused subplots
    for feat_idx in range(p, n_rows * n_cols):
        row = feat_idx // n_cols
        col = feat_idx % n_cols
        if row < n_rows and col < n_cols:
            axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.show()
    

# Run the demonstration
if __name__ == "__main__":
    # Generate and analyze the data
    X, Y, n_ex, params, task_data = demonstrate_draw_tasks()
    
    # Analyze confounding relationships
    analyze_confounding(X, Y, n_ex, params)
    
    # Create visualizations (uncomment if you want to see plots)
    visualize_relationships(X, Y, n_ex)
    
    print(f"\n" + "="*60)
    print("KEY INSIGHTS FROM draw_tasks:")
    print("="*60)
    print("1. Signal features have CAUSAL relationship with target")
    print("   - Same alpha coefficients across all tasks")
    print("   - Direct influence: signal → target")
    print("\n2. Noise features have CONFOUNDING relationship with target")
    print("   - Different gamma coefficients per task")
    print("   - Reverse causation: target → noise")
    print("   - Appear predictive but are actually effects!")
    print("\n3. Multi-task structure:")
    print("   - Shared causal mechanism (alpha)")
    print("   - Task-specific confounding patterns (gamma, beta)")
    print("   - Tests algorithm's ability to distinguish causation from correlation")
    
    print(f"\n4. Data shape verification:")
    print(f"   - Total samples: {X.shape[0]} (2 tasks × 100 samples)")
    print(f"   - Total features: {X.shape[1]} (3 signal + 2 noise)")
    print(f"   - Tasks can be identified by sample ranges: [0:{n_ex[0]}], [{n_ex[0]}:{sum(n_ex)}]")