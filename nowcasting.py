import numpy as np
import pandas as pd


class BVARGLP(object):

    def __init__(self,
                 data,
                 lags,
                 hyperpriors=1,  # TODO Set Boolean
                 vc=10e6,
                 pos=None,  # TODO Implement based on the variable names
                 mnpsi=1,  # TODO Set Boolean
                 mnalpha=0,  # TODO Set Boolean
                 sur=1,  # TODO Set Boolean
                 noc=1,  # TODO Set Boolean
                 fcast=1,  # TODO Set Boolean
                 hz=8,
                 mcmc=0,  # TODO Set Boolean
                 ndraws=20000,
                 ndrawsdiscard=10000,  # TODO set to 'ndraws/2'
                 mcmcconst=1,
                 mcmcfcast=1,  # TODO Set Boolean
                 mcmcstorecoef=1):  # TODO Set Boolean
        # TODO rewrite documentation according to variable usage
        """
        This class implements the Bayesian VAR from Giannone, Lenza and Primiceri (2012), hence the name GLP
        :param hyperpriors: 0 = no priors on hyperparameters
                            1 = reference priors on hyperparameters (default)
                            [NOTE: hyperpriors on psi calibrated for data expressed in
                            4 x logs, such as 4 x log(GDP). Thus if interest rate is in
                            percentage, divide by 100]
        :param vc: prior variance in the MN prior for the coefficients multiplying
                    the contant term (Default: vc=10e6)
        :param pos: position of the variables that enter the VAR in first  # TODO this should change to variable name, not variable position
                     differences and for which one might want to set the prior mean
                     on the coefficient on the first own lag in the MN prior and the
                     prior mean of the sum-of-coefficients prior to 0 (instead of 1)
                     (Default: pos=[])
        :param mnpsi: 0 = diagonal elements of the scale matrix of the IW prior on
                       the covariance of the residuals NOT treated as
                       hyperparameters (set to the residual variance of an AR(1))
                       1 = diagonal elements of the scale matrix of the IW prior on
                       the covariance of the residuals treated as
                       hyperparameters (default)
        :param mnalpha:  0 = Lag-decaying parameter of the MN prior set to 2 and
                         NOT treated as hyperparameter (default)
                         1 = Lag-decaying parameter of the MN prior treated as
                         hyperparameter
        :param sur: 0 = single-unit-root prior is OFF
                    1 = single-unit-root prior is ON and its std is treated as an
                    hyperparameter (default)
        :param noc: 0 = no-cointegration (sum-of coefficients) prior is OFF
                    1 = no-cointegration (sum-of coefficients) is ON and its std is
                    treated as an hyperparameter (default)
        :param fcast: 0 = does not generate forecasts at the posterior mode
                      1 = generates forecasts at the posterior mode (default)
        :param hz: longest horizon at which the code generates forecasts
                   (default: maxhz=8)
        :param mcmc: 0 = does not run the MCMC (default)
                     1 = runs the MCMC after the maximization
        :param ndraws: number of draws in the MCMC (default: Ndraws=20000)
        :param ndrawsdiscard: number of draws initially discarded to allow convergence
                              in the in the MCMC (default=Ndraws/2)
        :param mcmcconst: scaling constant for the MCMC (should be calibrated to achieve
                          an acceptance rate of approx 25%) (default: MCMCconst=1)
        :param mcmcfcast: 0 = does not generate forecasts when running the MCMC
                          1 = generates forecasts while running the MCMC
                          (for each draw of the hyperparameters the code takes a
                          draw of the VAR coefficients and shocks, and generates
                          forecasts at horizons hz) (default).
        :param mcmcstorecoef: 0 = does not store the MCMC draws of the VAR
                              coefficients and residual covariance matrix
                              1 = stores the MCMC draws of the VAR coefficients and
                              residual covariance matrix (default)
        """

        self.data = data
        self.lags = lags
        self.hyperpriors = hyperpriors
        self.vc = vc
        self.pos = pos
        self.mnalpha = mnalpha
        self.mnpsi = mnpsi
        self.sur = sur
        self.noc = noc
        self.fcast = fcast
        self.hz = hz
        self.mcmc = mcmc
        self.ndrwas = ndraws
        self.ndrwasdiscard = ndrawsdiscard
        self.mcmccosnt = mcmcconst
        self.mcmcfcast = mcmcfcast
        self.mcmcstorecoef = mcmcstorecoef

        self.TT = data.shape[0]  # Time-series sample size without lags
        self.n = data.shape[1]  # Number of variables in the VAR
        self.k = self.n * self.lags + 1  # Number of coefficients on each equation

        self._set_priors()
        self._regressor_matrix_ols()
        self._minnesota_prior_mean()
        self._minimization()

    def _set_priors(self):
        # Sets up the default choices for the priors of the BVAR of Giannone, Lenza and Primiceri (2012)
        if self.hyperpriors == 1:
            # hyperprior mode
            mode_lambda = 0.2
            mode_miu = 1
            mode_theta = 1

            # hyperprior sds
            sd_lambda = 0.4
            sd_miu = 1
            sd_theta = 1

            # scale and shape of the IG on psi/(d-n-1)
            scalePSI = 0.02 ** 2

            priorcoef = pd.DataFrame(index=['lambda', 'miu', 'theta', 'alpha', 'beta'],
                                     columns=['r_k', 'r_theta', 'PSI'])

            priorcoef.loc['lambda', 'r_k'], priorcoef.loc['lambda', 'r_theta'] = self._gamma_coef(mode_lambda, sd_lambda)
            priorcoef.loc['miu', 'r_k'], priorcoef.loc['miu', 'r_theta'] = self._gamma_coef(mode_miu, sd_miu)
            priorcoef.loc['theta', 'r_k'], priorcoef.loc['theta', 'r_theta'] = self._gamma_coef(mode_theta, sd_theta)
            priorcoef.loc['alpha', 'PSI'] = scalePSI
            priorcoef.loc['beta', 'PSI'] = scalePSI

            self.priorcoef = priorcoef

        else:
            self.priorcoef = None

    def _regressor_matrix_ols(self):
        # TODO purpose is to construct the SS matrix
        # Constructs the matrix of regressors

        n = self.n
        lags = self.lags
        data = self.data

        x = np.zeros((self.TT, self.k))
        x[:, 0] = 1

        for i in range(1, self.lags + 1):
            x[:, 1 + (i - 1) * n: i * n + 1] = data.shift(i).values

        y0 = data.iloc[:lags, :].mean().values
        x = x[lags:, :]
        y = data.values[lags:, :]

        self.T = y.shape[0]  # Sample size after lags

        # OLS for AR(1) residual variance of each equation
        SS = np.zeros(self.n)

        for i in range(self.n):
            y_reg = y[1:, i]
            x_reg = np.hstack((np.ones((self.T - 1, 1)), y[:-1, i].reshape((-1, 1))))
            ar1 = OLS1(y_reg, x_reg)
            SS[i] = ar1.sig2hatols

        self.SS = SS

    def _minnesota_prior_mean(self):
        b = np.zeros((self.k, self.n))
        diagb = np.ones(self.n)
        # TODO Set to zero the prior mean on the first own lag for variables selected in the vector pos
        # TODO diagb(pos) = 0
        b[1:self.n+1, :] = np.diag(diagb)
        self.b = b

    def _minimization(self):
        # Starting values for the minimization
        lambda0 = 0.2  # std of MN prior
        theta0 = 1  # std of SUR prior
        miu0 = 1  # std NOC prior
        alpha0 = 2  # lag-decaying parameter of the MN prior
        psi0 = self.SS

        # Bounds for the minimization step
        lambda_min = 0.0001
        lambda_max = 5

        alpha_min = 0.1
        alpha_max = 5

        theta_min = 0.0001
        theta_max = 50

        miu_min = 0.0001
        miu_max = 50

        PSI_min = self.SS / 100
        PSI_max = self.SS * 100

        # Transforming inputs to unbounded and builds the initial guess
        x0 = np.array([-np.log((lambda_max - lambda0) / (lambda0 - lambda_min))])

        if self.mnpsi == 1:
            inpsi = -np.log((PSI_max - psi0) / (psi0 - PSI_min))
            x0 = np.concatenate((x0, inpsi))
        else:
            # TODO this looks redundant
            inpsi = None

        if self.sur == 1:
            intheta = np.array([-np.log((theta_max - theta0) / (theta0 - theta_min))])
            x0 = np.concatenate((x0, intheta))
        else:
            # TODO this looks redundant
            intheta = None

        if self.noc == 1:
            inmiu = np.array([-np.log((miu_max - miu0) / (miu0 - miu_min))])
            x0 = np.concatenate((x0, inmiu))
        else:
            # TODO this looks redundant
            inmiu = None

        if self.mnalpha == 1:
            inalpha = np.array([-np.log((alpha_max - alpha0) / (alpha0 - alpha_min))])
            x0 = np.concatenate((x0, inalpha))
        else:
            # TODO this looks redundant
            inalpha = None

        # initial guess for the inverse Hessian
        H0 = 10 * np.eye(len(x0))

        # Maximization of the posterior of the hyperparameters

    @staticmethod
    def _gamma_coef(mode, sd):
        k = (2 + mode ** 2 / sd ** 2 + np.sqrt((4 + mode ** 2 / sd ** 2) * mode ** 2 / sd ** 2)) / 2
        theta = np.sqrt(sd ** 2 / k)
        return k, theta


class OLS1(object):
    # TODO Documentation (Simple OLS regression)

    def __init__(self, y, x):
        self.x = x
        self.y = y

        nobsy = y.shape[0]
        nobs, nvar = x.shape
        assert nobsy == nobs, 'x and y must have the same number of lines'

        self.nobs = nobs
        self.nvar = nvar

        self.XX = x.T @ x
        self.invXX = np.linalg.inv(self.XX)
        self.bhatols = self.invXX @ (x.T @ y)
        self.yhatols = x @ self.bhatols
        self.resols = y - self.yhatols
        self.sig2hatols = (self.resols.T @ self.resols) / (nobs - nvar)
        self.sigbhatols = self.sig2hatols * self.invXX
        self.r2 = np.var(self.yhatols) / np.var(y)
