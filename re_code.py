import numpy as np
import scipy as sp 
from sklearn import linear_model
import itertools

def full_search(train_x, train_y, valid_x, valid_y, n_samples_per_tasks, n_samples_per_task_valid,
                use_hsic, alpha, return_n_best=None):
    num_tasks = len(n_samples_per_tasks)
    # task_boundaries = np.cumsum(n_samples_per_tasks)

    best_subset = []
    accepted_subsets = []
    accepted_mse = []
    all_sets = []
    all_pvals = []

    best_mse = 1e10

    feature_list = np.arange(train_x.shape[1] + 1)
    current_iter = np.array(train_x.shape[1])

    #                           1. Baseline 
    pred_valid = np.mean(train_y)
    residual = valid_y - pred_valid 

    if use_hsic:
        # ??????????????????
        # could make into function for readability 
        valid_dom = mat_hsic(valid_y, n_samples_per_task_valid)
        ls = np_getDistances(residual, residual)
        sx = 0.5 * np.median(ls.flatten())

        stat, a, b = numpy_HsicGammaTest(residual, valid_dom, sx, 1, domain_kernel = valid_dom)
        pvals = 1. - sp.stats.gamma.cdf(stat, a, scale=b)

    else:
        residTup = levene_pval(residual, n_samples_per_task, num_task)
        pvals = sp.stats.levene(*residTup)[1]
    
    if (pvals > alpha):
        mse_current = np.mean((valid_y - pred_valid) ** 2)
        if mse_current < best_mse:
            best_mse = mse_current
            best_subset = []
            accepted_subsets.append([])
            accepted_mse.append([])

    all_sets.append([])
    all_pvals.append([])

    # 2. Subset search 
    for i in range(1, feature_list.size):
        for s in itertools.combinations(feature_list, i):
            subset_features = feature_list[np.array(s)]
            regr = linear_model.LinearRegression()

            regr.fit(train_x[:, subset_features], train_y.flatten())

            pred = regr.predict(valid_x[:, subset_features])[:, np.newaxis]

            residual = valid_y - pred 

            if use_hsic:
                valid_dom = mat_hsic(valid_y, n_samples_per_task_valid)
                ls = np_getDistance(residual, residual)
                sx = 0.5 * np.median(ls.flatten())
                stat, a, b = numpy_HsicGammaTest(residual, valid_dom, sx, 1, domain_kernel = valid_dom)
                pval = 1. - sp.stats.gamma.cdf(stat, a, scale=b)
            else:
                residTup = levene_pval(residual, n_samples_per_task_valid, num_tasks)
                pval = sp.stats.levene(*residTup)[1]

            all_sets.append(s)
            all_pvals.append(pval)

            if pval > alpha:
                mse_current = np.mean((valid_y - pred)**2)
                if mse_currrent < best_mse:
                    best_mse = mse_current
                    best_subset = s 
                    accepted_subsets.append(s)
                    accepted_mse.append(mse_current)
                    current_iter = np.intersect1d(current_iter, s)

    if len(accepted_subsets) == 0:
        all_pvals = np.array(all_pvals).flatten()
        sort_pvals = np.argsort(all_pvals)
        idx_max= sort_pvals[-1]
        best_subset = all_sets[idx_max]
        accepted_subsets.append(best_subset)
    
    if return_n_best:
        return [np.array(s) for s in accepted_subsets[-return_n_best:]], [np.array(mse) for mse in accepted_mse[-return_n_best:]], np.array(pvals[sort_pvals[:-return_n_best]])
    return np.array(best_subset), np.array(best_mse), np.array(sort_pvals[idx_max])


def greedy_search(train_x, train_y, valid_x, valid_y, n_samples_per_tasks, n_samples_per_task_valid, use_hsic, alpha):
    num_predictors = train_x.shape[1]
    selected = np.zeros(num_predictors)
    accepted_subsets = []

    all_sets = []
    all_pvals = []

    n_iters = 10 * num_predictors
    stay = 1 

    pow_2 = np.array([2**i for i in np.arange(num_predictors)])
    ind = 0
    prev_stat = 0 

    bins = []

    pred = np.mean(train_y)
    mse_current = np.mean((valid_y - pred) ** 2)
    residual = valid_y - pred

    residTup = levene_pval(residual, n_samples_per_task_valid, num_tasks)
    pval = sp.stats.levene(*residTup)[1]
    all_sets.append([])
    all_pvals.append(pval)

    if pval > alpha:
        accepted_subsets = np.array([])
        # Doesn't track pvals, best mse, best subset 

    while (stay > 0):
        pval_a = np.zeros(num_predictors)
        statistic_a = np.zeros(num_predictors)
        mse_a = np.zeros(num_predictors)

        for p in range(num_predictors):
            current_subset = np.sort(np.where(sellected == 1)[0])
            regr = linear_model.LinearRegression()

            if selected[p]==0:
                subset_add = np.append(current_subset, p).astype(int)
                regr.fit(trian_x[:, subset_add], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_add])[:, np.newaxis]
                mse_current = np.mean((valid_y - pred) ** 2)
                residual = valid_y - pred 

                residTyp = levene_pval(residual, n_samples_per_task_valid, n_samples_per_task_valid.size)
                levene_pval = sp.stats.levene(*residTup)

                pvals_a[p] = levene_pval[1]
                statistic_a[p] = levene_pval[0]
                mse_a[p] = mse_current

                all_sets.append(subset_add)
                all_pvals.append(levene_pval[1])

            else:
                acc_rem = np.copy(selected)
                acc_rem[p] = 0

                subset_rem = np.sortnp(np.where(acc_rem == 1)[0])

                if subset_rem.size == 0:
                    continue

                regr = linear_model.LinearRegression()
                regr.fit(train_x[:, subset_rem], train_y.flatten())
                pred = regr.predict(valid_x[:, subset_rem])[:, np.newaxis]
                mse_current = np.mean((valid_y - pred) ** 2)
                residual = valid_y - pred

                residTup = levene_pval(residual, n_samples_per_task_valid, n_samples_per_task_valid.size)
                levene_pval = sp.stats.levene(*residTup)

                pvals_a[p] = levene_pval[1]
                statistic_a[p] = levene_pval[0]
                mse_a[p] = mse_current

                all_sets.append(subset_rem)
                all_pvals.append(levene_pval[1])

        accepted = np.where(pvals_a > alpha)

        if accepted[0].size > 0:

            best_mse = np.amin(mse_a[np.where(pvals_a > alpha)])
            selected[np.where(mse_a == best_mse)[0]] = (selected[np.where(mse_a == best_mse)[0]] + 1) % 2

            accepted_subsets = np.sort(np.where(selected == 1)[0])

            binary = np.sum(pow_2 * selected)
            if (bins == binary).any():
                stay = 0
            bins.append(binary)

        else:
            best_pvals_arg = np.argmin(statistic_a)
            selected[best_pvals_arg] = (selected[best_pvals_arg + 1] % 2)
            binary = np.sum(pow_2 * selected)

            if (bins == binary).any():
                stay = 0
            bins.append(binary)

        if ind > n_iters:
            stay = 0
        ind += 1
    
    if accepted_subsets is None:
        all_pvals = np.array(all_pvals).flatten()
        max_pvals = np.argsort(all_pvals)[-1]
        accepted_subsets = np.sort(all_sets[max_pvals])

    return np.array(accepted_subsets)






