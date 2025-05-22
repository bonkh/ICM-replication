import numpy as np
import scipy as sp 
from sklearn import linear_model
import itertools

def mat_hsic(X,n_samples_per_task):

    # print(n_samples_per_task)
    task_boundaries = np.cumsum(n_samples_per_task)
    domains = np.zeros((np.sum(n_samples_per_task),np.sum(n_samples_per_task)))
    #Here the man of domains matrix can be 900x900 = num of samples x num of samples
    currentIndex = 0

    for i in range(n_samples_per_task.size):
        domains[currentIndex : task_boundaries[i], currentIndex:task_boundaries[i]] = np.ones((n_samples_per_task[i], n_samples_per_task[i]))
        currentIndex = task_boundaries[i]

        # So the result here is 
        # 111000000
        # 111000000
        # 111000000
        # 000111000
        # 000111000
        # 000111000
        # 000000111
        # 000000111
        # 000000111
    return domains


def np_getDistances(x,y):

    K = (x[:,:, np.newaxis] - y.T)
    result = np.linalg.norm(K,axis = 1)
    return result 

def numpy_GetKernelMat(X,sX):

	Kernel = (X[:,:, np.newaxis] - X.T).T
	Kernel = np.exp( -1./(2*sX) * np.linalg.norm(Kernel, axis=1))

	return Kernel


def numpy_HsicGammaTest(X,Y, sigma_X, sigma_Y, domain_kernel = 0):
    """
    Args:
        - sigma_X: kernel parameter for X
        - sigma_Y: kernel parameter for Y 
        - domain_kernel: Domain martrix for Y 

    """
    
    n = X.T.shape[1]
    
    kernel_X = numpy_GetKernelMat(X, sigma_X)

    kernel_Y = domain_kernel

    coef = 1./n

    # The formula can be founded there https://proceedings.neurips.cc/paper_files/paper/2007/file/d5cfead94f5350c12c322b5b664544c1-Paper.pdf
    HSIC = (coef**2) * np.sum(kernel_X * kernel_Y) + coef**4 * np.sum(
                kernel_X)*np.sum(kernel_Y) - 2* coef**3 * np.sum(np.sum(kernel_X,axis=1)*np.sum(kernel_Y, axis=1))

    #Get sums of Kernels
    KXsum = np.sum(kernel_X)
    KYsum = np.sum(kernel_Y)

    #Get stats for gamma approx

    xMu = 1./(n*(n-1))*(KXsum - n)
    yMu = 1./(n*(n-1))*(KYsum - n)
    V1 = coef**2*np.sum(kernel_X * kernel_X) + coef**4*KXsum**2 - 2*coef**3*np.sum(np.sum(kernel_X,axis=1)**2)
    V2 = coef**2*np.sum(kernel_Y * kernel_Y) + coef**4*KYsum**2 - 2*coef**3*np.sum(np.sum(kernel_Y,axis=1)**2)

    meanH0 = (1. + xMu*yMu - xMu - yMu)/n
    varH0 = 2.*(n-4)*(n-5)/(n*(n-1.)*(n-2.)*(n-3.))*V1*V2

    #Parameters of the Gamma
    a = meanH0**2/varH0
    b = n * varH0/meanH0

    return n*HSIC, a, b


def levene_pval(Residual,nEx, numR):

    prev = 0 
    n_ex_cum = np.cumsum(nEx)

    for j in range(numR):

        r1 = Residual[prev:n_ex_cum[j],:]

        if j == 0:
            residTup = (r1,)

        else:
            residTup = residTup + (r1,)

        prev = n_ex_cum[j]

    return residTup




def split_train_valid(x, y, n_samples_per_task_list, valid_split=0.2):

    task_boundaries = np.append(0, np.cumsum(n_samples_per_task_list).astype(int))
# n_ex_cumsum
    # print(task_boundaries)
    # Example: [0, 300, 600, 900].
    # This list show the boundaries for each tasks

    n_samples_per_task_train_set, n_samples_per_task_valid_set = [], []
    train_x, train_y, valid_x, valid_y = [], [], [], []

    for i in range(len(n_samples_per_task_list)): # Means for each task
        n_train_task = int((1 - valid_split) * n_samples_per_task_list[i])
        # print(n_train_task)
        # print(x[0.0])

        train_x.append( x[task_boundaries[i] : task_boundaries[i] + n_train_task])
        train_y.append( y[task_boundaries[i] : task_boundaries[i] + n_train_task])

        valid_x.append( x[task_boundaries[i] + n_train_task : task_boundaries[i + 1]])
        valid_y.append( y[task_boundaries[i] + n_train_task : task_boundaries[i + 1]])
        
        n_samples_per_task_train_set.append(n_train_task)
        n_samples_per_task_valid_set.append(n_samples_per_task_list[i] - n_train_task)

    train_x = np.concatenate(train_x, axis=0)
    valid_x = np.concatenate(valid_x, axis=0)
    train_y = np.concatenate(train_y, axis=0)
    valid_y = np.concatenate(valid_y, axis=0)

    n_samples_per_task_train_set = np.array(n_samples_per_task_train_set)
    n_samples_per_task_valid_set = np.array(n_samples_per_task_valid_set)

    return train_x, train_y, valid_x, valid_y, n_samples_per_task_train_set, n_samples_per_task_valid_set



def full_search(train_x, train_y, valid_x, valid_y, n_samples_per_task, n_samples_per_task_valid, 
                use_hsic, alpha, return_n_best=None):
    """
    Perform Algorithm 1, search over all possible subsets of features. 

    Args:
        - use_hsic: whether to use HSIC. If not, Levene test is used.
        - alpha: level for the statistical test of equality of distributions 
          (HSIC or Levene).
        return_n_best: return top n subsets (in terms of test statistic). 
          Default returns only the best subset. 

    """
    num_tasks = len(n_samples_per_task)
   
    best_subset = []
    accepted_sets = []
    accepted_mse = []
    all_sets = []
    all_pvals = []

    best_mse = 1e10

    rang = np.arange(train_x.shape[1])
    current_inter = np.arange(train_x.shape[1])

    # Step 1: Statistical test on residuals from mean predictor
    # It was consider as a baseline, with using no feature
    #Get numbers for the mean
    pred_valid = np.mean(train_y)
    residual = valid_y - pred_valid

    if use_hsic:
        valid_dom = mat_hsic(valid_y, n_samples_per_task_valid)
        ls = np_getDistances(residual, residual)
        sx = 0.5 * np.median(ls.flatten())

        stat, a, b = numpy_HsicGammaTest(residual, valid_dom,
                                               sx, 1, 
                                               domain_kernel = valid_dom)
        pvals = 1. - sp.stats.gamma.cdf(stat, a, scale=b)
    else:
        residTup = levene_pval(residual, n_samples_per_task, num_tasks)
        pvals = sp.stats.levene(*residTup)[1]

    # If residuals are independent, save the current baseline MSE

    if (pvals > alpha):
        mse_current  = np.mean((valid_y - pred_valid) ** 2)
        if mse_current < best_mse:
            best_mse = mse_current
            best_subset = []
            accepted_sets.append([])
            accepted_mse.append(mse_current)
    
    all_sets.append([])
    all_pvals.append(pvals)

    # Step 2: Loop over all subsets of features

    for i in range(1, rang.size + 1):
        for s in itertools.combinations(rang, i):
            currentIndex = rang[np.array(s)]
            regr = linear_model.LinearRegression()
            
            #Train regression with given subset on training data
            regr.fit(train_x[:, currentIndex], 
                     train_y.flatten())

            #Compute mse for the validation set
            pred = regr.predict(
              valid_x[:, currentIndex])[:,np.newaxis]

            #Compute residual
            residual = valid_y - pred

            if use_hsic:
                valid_dom = mat_hsic(valid_y, n_samples_per_task_valid)
                ls = np_getDistances(residual, residual)
                sx= 0.5 * np.median(ls.flatten())
                stat, a, b = numpy_HsicGammaTest(
                    residual, valid_dom, sx, 1, domain_kernel = valid_dom)
                pvals = 1.- sp.stats.gamma.cdf(stat, a, scale=b)
            else:
                residTup = levene_pval(residual, n_samples_per_task_valid, num_tasks)
                pvals = sp.stats.levene(*residTup)[1]
            
            all_sets.append(s)
            all_pvals.append(pvals)
                                                                            
            if (pvals > alpha):
                mse_current = np.mean((pred - valid_y) ** 2)
                if mse_current < best_mse: 
                    best_mse = mse_current
                    best_subset = s
                    current_inter = np.intersect1d(current_inter, s)
                    accepted_sets.append(s)
                    accepted_mse.append(mse_current)

 
    if len(accepted_sets) == 0:
        all_pvals = np.array(all_pvals).flatten()
        sort_pvals = np.argsort(all_pvals)
        idx_max = sort_pvals[-1]
        best_subset = all_sets[idx_max]
        accepted_sets.append(best_subset)

    if return_n_best:
        return [np.array(s) for s in accepted_sets[-return_n_best:]]
    else:
        return np.array(best_subset)
    
     
def subset(x, y, n_samples_per_task_list, delta, valid_split, use_hsic = False, 
           return_n_best = None):

    """
    Run Algorithm 1 for full subset search. 

    Args:
        - x: train features. Shape [n_samples, n_features].
        - y: train labels. Shape [n_examples, 1].
        - n_samples_per_task_list: list with number of examples per task (should be ordered in 
          train_x and train_y). Shape: [n_tasks]
        - delta: Significance level of statistical test.
        - use_hsic: use HSIC? If False, Levene is used. 
        return_n_best: number of subsets to return. 
    """

    train_x, train_y, valid_x, valid_y, n_ex_train, n_ex_valid = split_train_valid(x, y, n_samples_per_task_list, valid_split)

    # print(n_ex_valid)
    
    subset = full_search(train_x, train_y, valid_x, valid_y,
                         n_ex_train, n_ex_valid, use_hsic, 
                         delta, return_n_best = return_n_best)

    return subset

# TODO: Implement HSIC check function/Levene check function
def greedy_search(train_x, train_y, valid_x, valid_y, n_samples_per_task, n_samples_per_task_valid, 
                  use_hsic, alpha):
    """
    Perform Algorithm 2, greedy search over subsets of features.
    Args:
        - use_hsic: whether to use HSIC. If not, Levene test is used.
        - alpha: level for the statistical test of equality of distributions 
          (HSIC or Levene).
    """

    # Initialization
    num_tasks = len(n_samples_per_task)
    num_predictors = train_x.shape[1]

    # List tracks: p-values and MSE
    all_pvals = []
    all_mse = []

    # Current subset track: selected, accepted subset
    selected = np.zeros(num_predictors)

    # Limit for iterations
    n_iters = 10 * num_predictors
    stay = 1

    pow_2 = np.array([2**i for i in np.arange(num_predictors)])

    ind = 0
    bins = []

    # Step 1: Statistical test on residuals from mean predictor (Only use Levene)
    pred_baseline = np.mean(train_y)
    residual_baseline = valid_y - pred_baseline

    residTup = levene_pval(residual_baseline, n_samples_per_task, num_tasks)
    levene = sp.stats.levene(*residTup)

    binary = np.sum(pow_2 * selected)
    bins.append(binary)
    all_pvals.append(levene[1])

    if all_pvals[-1] > alpha:
        accepted_subset = np.array([])
    else:
        accepted_subset = None

    # Step 2: Loop over subsets of features
    while stay:
        # p-values, statistics, and MSE for each predictor
        pvals_a = np.zeros(num_predictors)
        statistic_a = 1e10 * np.ones(num_predictors)
        mse_a = np.zeros(num_predictors)

        for p in range(num_predictors):
            current_subset = np.sort(np.where(selected == 1)[0])
            regr = linear_model.LinearRegression()

            # If the predictor is not selected, add it to the current subset
            if selected[p] == 0:
                subset_add = np.append(current_subset, p).astype(int) 
                regr.fit(train_x[:, subset_add], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_add])[:, np.newaxis]
                mse_current = np.mean((pred - valid_y) ** 2)
                residual = valid_y - pred

                residTup = levene_pval(residual, n_samples_per_task_valid, n_samples_per_task_valid.size)
                levene = sp.stats.levene(*residTup)

                pvals_a[p] = levene[1]
                statistic_a[p] = levene[0]
                mse_a[p] = mse_current

            # If the predictor is already selected, remove it from the current subset
            elif selected[p] == 1:
                acc_rem = np.copy(selected)
                acc_rem[p] = 0
                subset_rem = np.sort(np.where(acc_rem == 1)[0])

                if subset_rem.size == 0:
                    continue

                regr.fit(train_x[:, subset_rem], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_rem])[:, np.newaxis]
                mse_current = np.mean((pred - valid_y) ** 2)
                residual = valid_y - pred

                residTup = levene_pval(residual, n_samples_per_task_valid, n_samples_per_task_valid.size)
                levene = sp.stats.levene(*residTup)

                pvals_a[p] = levene[1]
                statistic_a[p] = levene[0]
                mse_a[p] = mse_current

        accepted = np.where(pvals_a > alpha)

        # If the p-values are all above alpha, we choose the best MSE subset
        if accepted[0].size > 0:
            best_mse = np.amin(mse_a[np.where(pvals_a > alpha)])
            selected[np.where(mse_a == best_mse)] = (selected[np.where(mse_a == best_mse)] + 1) % 2

            accepted_subset = np.sort(np.where(selected == 1)[0])
            binary = np.sum(pow_2 * selected)

            if binary in bins:
                stay = 0
            bins.append(binary)

        # Else we fall back on using test statistics to choose the best subset
        else:   
            best_pval_arg = np.argmin(statistic_a)
            selected[best_pval_arg] = (selected[best_pval_arg] + 1) % 2
            binary = np.sum(pow_2 * selected)

            if binary in bins:
                stay = 0
            bins.append(binary)

        if ind > n_iters:
            stay = 0
        ind += 1

    if accepted_subset is None:
        all_pvals = np.array(all_pvals).flatten()
        max_pvals = np.argsort(all_pvals)[-1]
        accepted_subset = np.where(np.array(list(bin(bins[max_pvals])[2:].zfill(num_predictors)), dtype=int) == 1)[0]

    return np.array(accepted_subset)


def greedy_subset(x, y, n_ex, delta, valid_split, use_hsic = False):

    """
    Run Algorithm 2 for greedy subset search. 

    Args:
        x: train features. Shape [n_examples, n_features].
        y: train labels. Shape [n_examples, 1].
        n_ex: list with number of examples per task (should be ordered in 
          train_x and train_y). Shape: [n_tasks]
        delta: Significance level of statistical test.
        use_hsic: use HSIC? If False, Levene is used. 
        return_n_best: number of subsets to return. 
    """
    train_x, train_y, valid_x, valid_y, n_ex_train, n_ex_valid = split_train_valid(x, y, n_ex, valid_split)
    subset = greedy_search(train_x, train_y, valid_x, valid_y, n_ex_train, 
                           n_ex_valid, use_hsic, delta)

    return np.array(subset)


def lightGBM_test(x, y, n_ex, delta, valid_split, use_hsic = False):
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score
    from sklearn.metrics import mean_squared_error
    import matplotlib.pyplot as plt

    # 3. Train LightGBM model
    train_x, train_y, valid_x, valid_y, n_ex_train, n_ex_valid = split_train_valid(x, y, n_ex, valid_split)
    lgb_train = lgb.Dataset(train_x, train_y)
    lgb_eval = lgb.Dataset(valid_x, valid_y, reference=lgb_train)

    params = {
        'objective': 'regression',
        'metric': 'rmse',
        'verbosity': -1
    }

    model = lgb.train(params, lgb_train, valid_sets=[lgb_eval], num_boost_round=100)
    y_pred = model.predict(valid_x)
    print("Light GBM predictions:")
    print("MSE:", mean_squared_error(valid_y, y_pred))

    # # 5. Plot feature importance
    # lgb.plot_importance(model, max_num_features=10, importance_type='gain')  # 'split' or 'gain'
    # plt.title("Feature Importance")
    # plt.show()

