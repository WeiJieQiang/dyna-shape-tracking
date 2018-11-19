print(__doc__)


import numpy as np
from matplotlib import pyplot as plt

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

#np.random.seed(0)


def f(x):
    """The function to predict."""
    return 5 + x * np.sin(x)

# ----------------------------------------------------------------------

alpha = 2*np.pi
X = np.atleast_2d([0, .1*alpha, .2*alpha, .3*alpha, .4*alpha, .5*alpha, .6*alpha, .7*alpha, .8*alpha, .9*alpha, alpha]).T

# Observations
y = f(X).ravel()
dy = 0 + 2.0 * np.random.random(y.shape)
#noise = np.random.normal(0, dy)
y += dy

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.atleast_2d(np.linspace(0, alpha, 100)).T

# Instantiate a Gaussian Process model
#kernel = C(1.0, (1e-3, 1e3))*RBF(0.1, (1e-5, 1e5)) 
kernel = 1* RBF(length_scale=0.1, length_scale_bounds=(1e-3, 1e3))#RBF(0.1, (1e-5, 1e5))*C(1.0, (1e-3, 1e3))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

# Fit to data using Maximum Likelihood Estimation of the parameters
gp.fit(X, y)

# Make the prediction on the meshed x-axis (ask for MSE as well)
y_pred, sigma = gp.predict(x, return_std=True)

# Plot the function, the prediction and the 95% confidence interval based on
# the MSE

plt.figure()
plt.plot(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x) + 5$')
plt.plot(X, y, 'r.', markersize=10, label=u'Observations')
plt.plot(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='upper left')


plt.figure()
plt.polar(x, f(x), 'r:', label=u'$f(x) = x\,\sin(x) + 5$')
plt.polar(X, y, 'r.', markersize=10, label=u'Observations')
plt.polar(x, y_pred, 'b-', label=u'Prediction')
plt.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([y_pred - 1.9600 * sigma,
                        (y_pred + 1.9600 * sigma)[::-1]]),
         alpha=.5, fc='b', ec='None', label='95% confidence interval')
plt.xlabel('$x$')
plt.ylabel('$f(x)$')
plt.ylim(-10, 20)
plt.legend(loc='best')
