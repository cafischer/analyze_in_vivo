import numpy as np
from scipy.stats import chisquare
from scipy.optimize import curve_fit
import matplotlib.pyplot as pl


def chi_square_goodness_of_fit(observed, expected, n_intervals, n_estimated_parameters):
    chi_squared = np.sum((observed - expected)**2 / expected)
    dof = n_intervals -n_estimated_parameters -1

    return chi_squared, dof

def exp_dist(x, rate):
    return rate * np.exp(-x * rate)


if __name__ == '__main__':
    # parameters
    test_points = np.arange(1, 10 + 1, 1)

    # generate observed data
    n_samples = 200
    rate_data = 0.8
    #data = n_samples * exp_dist(test_points, rate_data)# + 0.01*np.random.rand(len(test_points))
    data = np.array([279, 20, 8, 5, 0, 1, 1, 0, 0, 0])
    n_samples = np.sum(data)

    # fit scale of exponential distribution
    p_opt, _ = curve_fit(exp_dist, test_points, data/float(n_samples))
    rate_est = p_opt[0]
    #rate_est = 1.0

    n_estimated_parameters = 1

    # compute expected values
    expected = exp_dist(test_points, rate_est) * n_samples

    # compute observed values
    observed = data

    # compute statistic
    idxs_enough_data = observed >= 4
    observed = observed[idxs_enough_data]
    expected = expected[idxs_enough_data]
    test_points = test_points[idxs_enough_data]

    chisquared, p = chisquare(observed, expected, n_estimated_parameters)

    print 'rate true: ', rate_data
    print 'rate est.: ', rate_est
    print 'chi-squared: %.2f' % chisquared
    print 'p-val: %.5f' % p

    pl.figure()
    pl.bar(test_points, observed, width=test_points[1] - test_points[0])
    pl.plot(test_points, expected, 'or')
    pl.show()