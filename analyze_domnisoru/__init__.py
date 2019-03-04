import scipy.stats as st
import numpy as np
import matplotlib.pyplot as pl


def perform_kde(x, sigma):
    kde = st.gaussian_kde(x, bw_method=np.sqrt(sigma**2 / np.cov(x)))
    return kde


def evaluate_kde(x, kde):
    pdf_kde = kde.pdf(x)
    return pdf_kde


if __name__ == '__main__':
    sigma = 2
    x = np.array([2, 10])
    kde = perform_kde(x, sigma=sigma)
    x_test = np.arange(0, 12, 0.01)

    assert np.sqrt(kde.covariance) == sigma
    assert np.allclose(evaluate_kde(x_test, kde),
                          st.norm.pdf(x_test, loc=x[0], scale=sigma)/2. + st.norm.pdf(x_test, loc=x[1], scale=sigma)/2.)

    pl.figure()
    pl.plot(x_test, evaluate_kde(x_test, kde))
    pl.plot(x_test, st.norm.pdf(x_test, loc=x[0], scale=sigma)/2. + st.norm.pdf(x_test, loc=x[1], scale=sigma)/2.)
    pl.show()