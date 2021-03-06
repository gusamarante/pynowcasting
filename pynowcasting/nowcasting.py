import numpy as np
import pandas as pd
from tqdm import tqdm
import numpy.ma as ma
from scipy.special import gammaln
from pykalman import KalmanFilter
from pynowcasting.pycsminwel import csminwel


class BVARGLP(object):

    def __init__(self, data, lags, hz=8, vc=10e6, stationary_prior=None, crit=1e-16,
                 hyperpriors=True, mnpsi=True, mnalpha=False, sur=True, noc=True,
                 fcast=False, mcmc=False, ndraws=20000, ndrawsdiscard=None, mcmcconst=1,
                 mcmcfcast=True, mcmcstorecoef=True, verbose=False):
        """
        This class implements the Bayesian VAR from Giannone, Lenza and Primiceri (2012), hence the name GLP. The main
        idea of the models is to use multiple priors, each with their own hyperprior, in order to generate a shrinkage
        behaviour.

        This class only accepts data with a quarterly frequency and with no missign data.

        @param hyperpriors: False = no priors on hyperparameters
                            True = reference priors on hyperparameters (default)
                            [NOTE: hyperpriors on psi calibrated for data expressed in
                            4 x logs, such as 4 x log(GDP). Thus if interest rate is in
                            percentage, divide by 100]

        @param vc: prior variance in the MN prior for the coefficients multiplying
                   the contant term (Default: vc=10e6)

        @param stationary_prior: names of the variables that enter the VAR in first
                                 differences and for which one might want to set the prior mean
                                 on the coefficient on the first own lag in the MN prior and the
                                 prior mean of the sum-of-coefficients prior to 0 (instead of
                                 the typical 1)

        @param mnpsi: False = diagonal elements of the scale matrix of the IW prior on
                      the covariance of the residuals NOT treated as
                      hyperparameters (set to the residual variance of an AR(1))
                      True = diagonal elements of the scale matrix of the IW prior on
                      the covariance of the residuals treated as hyperparameters (default)

        @param mnalpha:  False = Lag-decaying parameter of the MN prior set to 2 and
                         NOT treated as hyperparameter (default)
                         True = Lag-decaying parameter of the MN prior treated as
                         hyperparameter

        @param sur: False = single-unit-root prior is OFF
                    True = single-unit-root prior is ON and its std is treated as an
                    hyperparameter (default)

        @param noc: False = no-cointegration (sum-of coefficients) prior is OFF
                    True = no-cointegration (sum-of coefficients) is ON and its std is
                    treated as an hyperparameter (default)

        @param fcast: False = does not generate forecasts at the posterior mode
                      True = generates forecasts at the posterior mode (default)

        @param hz: number of quarters for which it generates forecasts (default: hz=8)

        @param mcmc: False = does not run the MCMC (default)
                     True = runs the MCMC after the maximization

        @param ndraws: number of draws in the MCMC (default: Ndraws=20000)

        @param ndrawsdiscard: number of draws initially discarded to allow convergence
                              in the in the MCMC (default=Ndraws/2)

        @param mcmcconst: scaling constant for the MCMC (should be calibrated to achieve
                          an acceptance rate of approx 25%) (default: MCMCconst=1)

        @param mcmcfcast: False = does not generate forecasts when running the MCMC
                          True = generates forecasts while running the MCMC
                          (for each draw of the hyperparameters the code takes a
                          draw of the VAR coefficients and shocks, and generates
                          forecasts at horizons hz) (default).

        @param mcmcstorecoef: False = does not store the MCMC draws of the VAR
                              coefficients and residual covariance matrix
                              True = stores the MCMC draws of the VAR coefficients and
                              residual covariance matrix (default)

        @param verbose: Prints relevant information during the estimation.

        @param crit: value for convergence criteria
        """

        assert data.index.inferred_freq == 'Q', "input 'data' must be quarterly and recognized by pandas."

        self.data = data
        self.lags = lags
        self.hyperpriors = hyperpriors
        self.vc = vc
        self.stationary_prior = stationary_prior

        if stationary_prior is None:
            self.pos = None
        else:
            self.pos = [self.data.columns.get_loc(var) for var in stationary_prior]

        self.mnalpha = mnalpha
        self.mnpsi = mnpsi
        self.sur = sur
        self.noc = noc
        self.fcast = fcast
        self.hz = hz
        self.mcmc = mcmc
        self.ndraws = ndraws
        self.ndrwasdiscard = int(ndraws/2) if ndrawsdiscard is None else ndrawsdiscard
        self.mcmccosnt = mcmcconst
        self.mcmcfcast = mcmcfcast
        self.mcmcstorecoef = mcmcstorecoef
        self.verbose = verbose
        self.crit = crit

        self.TT = data.shape[0]  # Time-series sample size without lags
        self.n = data.shape[1]  # Number of variables in the VAR
        self.k = self.n * self.lags + 1  # Number of coefficients on each equation

        self._set_priors()
        self._regressor_matrix_ols()
        self._minimization()

        if self.fcast:
            self._forecasts()

        if self.mcmc:
            self._mcmc()

    def _set_priors(self):
        # Sets up the default choices for the priors of the BVAR of Giannone, Lenza and Primiceri (2012)
        if self.hyperpriors:
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

            priorcoef.loc['lambda', 'r_k'], priorcoef.loc['lambda', 'r_theta'] = \
                self._gamma_coef(mode_lambda, sd_lambda)
            priorcoef.loc['miu', 'r_k'], priorcoef.loc['miu', 'r_theta'] = self._gamma_coef(mode_miu, sd_miu)
            priorcoef.loc['theta', 'r_k'], priorcoef.loc['theta', 'r_theta'] = self._gamma_coef(mode_theta, sd_theta)
            priorcoef.loc['alpha', 'PSI'] = scalePSI
            priorcoef.loc['beta', 'PSI'] = scalePSI

            self.priorcoef = priorcoef

        else:
            self.priorcoef = None

    def _regressor_matrix_ols(self):
        # purpose is to construct the SS matrix

        # Constructs the matrix of regressors

        n = self.n
        lags = self.lags
        data = self.data

        x = np.zeros((self.TT, self.k))
        x[:, 0] = 1

        for i in range(1, self.lags + 1):
            x[:, 1 + (i - 1) * n: i * n + 1] = data.shift(i).values

        self.y0 = data.iloc[:lags, :].mean().values
        self.x = x[lags:, :]
        self.y = data.values[lags:, :]

        self.T = self.y.shape[0]  # Sample size after lags

        # OLS for AR(1) residual variance of each equation
        SS = np.zeros(self.n)

        for i in range(self.n):
            y_reg = self.y[1:, i]
            x_reg = np.hstack((np.ones((self.T - 1, 1)), self.y[:-1, i].reshape((-1, 1))))
            ar1 = OLS1(y_reg, x_reg)
            SS[i] = ar1.sig2hatols

        self.SS = SS

    def _minimization(self):
        # Starting values for the minimization
        self.lambda0 = 0.2  # std of MN prior
        self.theta0 = 1  # std of SUR prior
        self.miu0 = 1  # std NOC prior
        self.alpha0 = 2  # lag-decaying parameter of the MN prior
        self.psi0 = self.SS

        # Bounds for the minimization step
        self.lambda_min = 0.0001
        self.lambda_max = 5

        self.alpha_min = 0.1
        self.alpha_max = 5

        self.theta_min = 0.0001
        self.theta_max = 50

        self.miu_min = 0.0001
        self.miu_max = 50

        self.psi_min = self.SS / 100
        self.psi_max = self.SS * 100

        # Transforming inputs to unbounded and builds the initial guess
        x0 = np.array([-np.log((self.lambda_max - self.lambda0) / (self.lambda0 - self.lambda_min))])

        if self.mnpsi:
            inpsi = -np.log((self.psi_max - self.psi0) / (self.psi0 - self.psi_min))
            x0 = np.concatenate((x0, inpsi))

        if self.sur:
            intheta = np.array([-np.log((self.theta_max - self.theta0) / (self.theta0 - self.theta_min))])
            x0 = np.concatenate((x0, intheta))

        if self.noc:
            inmiu = np.array([-np.log((self.miu_max - self.miu0) / (self.miu0 - self.miu_min))])
            x0 = np.concatenate((x0, inmiu))

        if self.mnalpha:
            inalpha = np.array([-np.log((self.alpha_max - self.alpha0) / (self.alpha0 - self.alpha_min))])
            x0 = np.concatenate((x0, inalpha))

        # initial guess for the inverse Hessian
        H0 = 10 * np.eye(len(x0))

        # Minimization of the negative of the posterior of the hyperparameters
        def myfun(xxx):
            logML, _, _ = self._logmlvar_formin(xxx)
            return -logML

        # Optimization
        fh, xh, gh, h, itct, fcount, retcodeh = csminwel(fcn=myfun,
                                                         x0=x0,
                                                         h0=H0,
                                                         grad=None,
                                                         crit=self.crit,
                                                         nit=1000,
                                                         verbose=self.verbose)

        self.itct = itct
        self.xh = xh
        self.h = h
        self.log_post, self.betahat, self.sigmahat = self._logmlvar_formin(xh)
        self.lamb = self.lambda_min + (self.lambda_max - self.lambda_min) / (1 + np.exp(-xh[0]))
        self.theta = self.theta_max
        self.miu = self.miu_max

        if self.mnpsi:
            # diagonal elements of the scale matrix of the IW prior on the residual variance
            self.psi = self.psi_min + (self.psi_max - self.psi_min) / (1 + np.exp(-xh[1:self.n + 1]))

            if self.sur:
                # std of sur prior at the peak
                self.theta = self.theta_min + (self.theta_max - self.theta_min) / (1 + np.exp(-xh[self.n + 1]))

                if self.noc:
                    # std of noc prior at the peak
                    self.miu = self.miu_min + (self.miu_max - self.miu_min) / (1 + np.exp(-xh[self.n + 2]))

            else:  # self.sur == 0
                if self.noc:
                    # std of noc prior at the peak
                    self.miu = self.miu_min + (self.miu_max - self.miu_min) / (1 + np.exp(-xh[self.n + 1]))

        else:  # self.mnpsi == 0
            self.psi = self.SS

            if self.sur:
                # std of sur prior at the peak
                self.theta = self.theta_min + (self.theta_max - self.theta_min) / (1 + np.exp(-xh[1]))

                if self.noc:
                    # std of noc prior at the peak
                    self.miu = self.miu_min + (self.miu_max - self.miu_min) / (1 + np.exp(-xh[2]))

            else:
                if self.noc:
                    # std of noc prior at the peak
                    self.miu = self.miu_min + (self.miu_max - self.miu_min) / (1 + np.exp(-xh[1]))

        if not self.mnalpha:
            self.alpha = 2
        else:
            # Lag-decaying parameter of the MN prior
            self.alpha = self.alpha_min + (self.alpha_max - self.alpha_min) / (1 + np.exp(-xh[-1]))

    def _forecasts(self):
        # Forecasts ate the posterior mode
        Y = np.vstack([self.y, np.zeros((self.hz, self.n))])
        for tau in range(self.hz):
            indexes = list(range(self.T + tau - 1, self.T + tau - self.lags - 1, -1))
            xT = np.vstack([1, Y[indexes].T.reshape((self.k - 1, 1), order="F")]).T
            Y[self.T + tau, :] = xT @ self.betahat

        self.forecast = Y[-self.hz:, :]

    def _mcmc(self):
        # Jacobian of the transformation of the hyperparameters that has been
        # used for the constrained maximization
        JJ = np.exp(self.xh) / ((1 + np.exp(self.xh)) ** 2)
        JJ[0] = (self.lambda_max - self.lambda_min) * JJ[0]

        if self.mnpsi:
            JJ[1: self.n + 1] = (self.psi_max - self.psi_min) * JJ[1: self.n + 1]

            if self.sur:
                JJ[self.n + 1] = (self.theta_max - self.theta_min) * JJ[self.n + 1]

                if self.noc:
                    JJ[self.n + 2] = (self.miu_max - self.miu_min) * JJ[self.n + 2]

            else:
                if self.noc:
                    JJ[self.n + 1] = (self.miu_max - self.miu_min) * JJ[self.n + 1]

        else:

            if self.sur:
                JJ[1] = (self.theta_max - self.theta_min) * JJ[1]

                if self.noc:
                    JJ[2] = (self.miu_max - self.miu_min) * JJ[2]

            else:
                if self.noc:
                    JJ[1] = (self.miu_max - self.miu_min) * JJ[1]

        if self.mnalpha:
            JJ[-1] = (self.alpha_max - self.alpha_min) * JJ[-1]

        JJ = np.diag(JJ)
        HH = JJ @ self.h @ JJ

        # Regularization to assure that HH is positive-definite
        eigval, eigvec = np.linalg.eig(HH)
        HH = eigvec @ np.diag(np.abs(eigval)) @ eigvec.T

        # recovering the posterior mode
        postmode = np.array([self.lamb])

        if self.mnpsi:
            modepsi = np.array(self.psi)
            postmode = np.concatenate((postmode, modepsi))

        if self.sur:
            modetheta = np.array([self.theta])
            postmode = np.concatenate((postmode, modetheta))

        if self.noc:
            modemiu = np.array([self.miu])
            postmode = np.concatenate((postmode, modemiu))

        if self.mnalpha:
            modealpha = np.array([self.alpha])
            postmode = np.concatenate((postmode, modealpha))

        # starting value of the Metropolis algorithm
        P = np.zeros((self.ndraws, self.xh.shape[0]))
        logMLold = -10e15
        while logMLold == -10e15:
            P[0, :] = np.random.multivariate_normal(mean=postmode,
                                                    cov=(self.mcmccosnt ** 2) * HH)
            logMLold, betadrawold, sigmadrawold = self._logmlvar_formcmc(P[0])

        # matrix to store the draws of the VAR coefficients if MCMCstorecoeff is on
        if self.mcmcstorecoef:
            mcmc_beta = np.zeros((self.k, self.n, self.ndraws - self.ndrwasdiscard))
            mcmc_sigma = np.zeros((self.n, self.n, self.ndraws - self.ndrwasdiscard))
        else:
            mcmc_beta = None
            mcmc_sigma = None

        # matrix to store the forecasts if MCMCfcast is on
        if self.mcmcfcast:
            mcmc_Dforecast = np.zeros((self.hz, self.n, self.ndraws - self.ndrwasdiscard))
        else:
            mcmc_Dforecast = None

        # Metropolis iterations
        count = 0
        for i in tqdm(range(1, self.ndraws), 'MCMC Iterations', disable=not self.verbose):
            # draw candidate value
            P[i, :] = np.random.multivariate_normal(mean=P[i - 1, :],
                                                    cov=(self.mcmccosnt ** 2) * HH)
            logMLnew, betadrawnew, sigmadrawnew = self._logmlvar_formcmc(P[i, :])

            if logMLnew > logMLold:  # if there is an improvement, accept it
                logMLold = logMLnew
                count = count + 1
            else:  # If there is no improvement, there is a chance to accept the draw
                if np.random.rand() < np.exp(logMLnew - logMLold):  # If accetpted
                    logMLold = logMLnew
                    count = count + 1
                else:  # If not accepted, overwrite the draw with the last value
                    P[i, :] = P[i - 1, :]

                    # if MCMCfcast is on, take a new draw of the VAR coefficients with
                    # the old hyperparameters if have rejected the new ones
                    if self.mcmcfcast or self.mcmcstorecoef:
                        _, betadrawnew, sigmadrawnew = self._logmlvar_formcmc(P[i, :])

            # stores draws of VAR coefficients if MCMCstorecoeff is on
            if (i >= self.ndrwasdiscard) and self.mcmcstorecoef:
                mcmc_beta[:, :, i - self.ndrwasdiscard] = betadrawnew
                mcmc_sigma[:, :, i - self.ndrwasdiscard] = sigmadrawnew

            # produce and store the forecasts if MCMCfcast is on
            if (i >= self.ndrwasdiscard) and self.mcmcfcast:
                Y = np.vstack([self.y, np.zeros((self.hz, self.n))])
                for tau in range(self.hz):
                    indexes = list(range(self.T + tau - 1, self.T + tau - self.lags - 1, -1))
                    xT = np.vstack([1, Y[indexes].T.reshape((self.k - 1, 1), order="F")]).T
                    Y[self.T + tau, :] = xT @ betadrawnew + np.random.multivariate_normal(mean=np.zeros(self.n),
                                                                                          cov=sigmadrawnew)

                mcmc_Dforecast[:, :, i - self.ndrwasdiscard] = Y[-self.hz:, :]

        # store the draws of the hyperparameters
        mcmc_lambda = P[self.ndrwasdiscard:, 0]  # Standard Minesota Prior

        mcmc_psi = None
        mcmc_theta = None
        mcmc_miu = None

        if self.mnpsi:
            # diagonal elements of the scale matrix of the IW prior on the residual variance
            mcmc_psi = P[self.ndrwasdiscard:, 1:self.n+2]

            if self.sur:
                # std of sur prior
                mcmc_theta = P[self.ndrwasdiscard:, self.n + 1]

                if self.noc:
                    # std of noc prior
                    mcmc_miu = P[self.ndrwasdiscard:, self.n + 2]

            else:  # self.sur == 0
                if self.noc:
                    # std of noc prior
                    mcmc_miu = P[self.ndrwasdiscard:, self.n + 1]

        else:  # self.mnpsi == 0
            if self.sur:
                # std of sur prior
                mcmc_theta = P[self.ndrwasdiscard:, 1]

                if self.noc:
                    # std of noc prior
                    mcmc_miu = P[self.ndrwasdiscard:, 2]

            else:  # self.sur == 0
                if self.noc:
                    # std of noc prior
                    mcmc_miu = P[self.ndrwasdiscard:, 1]

        if self.mnalpha:
            # Lag-decaying parameter of the MN prior
            mcmc_alpha = P[self.ndrwasdiscard:, -1]
            self.mcmc_alpha = mcmc_alpha

        mcmc_accrate = np.mean((mcmc_lambda[1:] != mcmc_lambda[:-1]))

        # Save the chains as attributes
        self.mcmc_beta = mcmc_beta
        self.mcmc_sigma = mcmc_sigma
        self.mcmc_dforecast = mcmc_Dforecast
        self.mcmc_lambda = mcmc_lambda
        self.mcmc_psi = mcmc_psi
        self.mcmc_theta = mcmc_theta
        self.mcmc_miu = mcmc_miu

        self.mcmc_accrate = mcmc_accrate

    def _logmlvar_formin(self, par):
        """
        This function computes the log-posterior (or the logML if hyperpriors=0),
        the posterior mode of the coefficients and the covariance matrix of the
        residuals of the BVAR of Giannone, Lenza and Primiceri (2012)
        """

        # The following avoids the warning "referenced before assignment"
        theta = None
        miu = None

        # hyperparameters
        lambda_ = self.lambda_min + (self.lambda_max - self.lambda_min) / (1 + np.exp(-par[0]))
        d = self.n + 2

        if not self.mnpsi:
            psi = self.SS * (d - self.n - 1)

            if self.sur:
                theta = self.theta_min + (self.theta_max - self.theta_min) / (1 + np.exp(-par[1]))

                if self.noc:
                    miu = self.miu_min + (self.miu_max - self.miu_min) / (1 + np.exp(-par[2]))

            else:
                if self.noc:
                    miu = self.miu_min + (self.miu_max - self.miu_min) / (1 + np.exp(-par[1]))

        else:
            psi = self.psi_min + (self.psi_max - self.psi_min) / (1 + np.exp(-par[1:self.n + 1]))

            if self.sur:
                theta = self.theta_min + (self.theta_max - self.theta_min) / (1 + np.exp(-par[self.n + 1]))

                if self.noc:
                    miu = self.miu_min + (self.miu_max - self.miu_min) / (1 + np.exp(-par[self.n + 2]))

            else:
                if self.noc:
                    miu = self.miu_min + (self.miu_max - self.miu_min) / (1 + np.exp(-par[self.n + 1]))

        if not self.mnalpha:
            alpha = 2
        else:  # self.mnalpha == 1
            alpha = self.alpha_min + (self.alpha_max - self.alpha_min) / (1 + np.exp(-par[-1]))

        # Setting up the priors
        omega = np.zeros(self.k)
        omega[0] = self.vc

        for i in range(1, self.lags + 1):
            omega[1 + (i - 1) * self.n: 1 + i * self.n] = \
                (d - self.n - 1) * (lambda_ ** 2) * (1 / (i ** alpha)) / psi

        # Prior scale matrix for the covariance of the shocks
        PSI = np.diag(psi)

        # dummy observations if sur and / or noc = 1
        Td = 0
        xdsur = np.array([]).reshape((0, self.k))
        ydsur = np.array([]).reshape((0, self.n))

        xdnoc = np.array([]).reshape((0, self.k))
        ydnoc = np.array([]).reshape((0, self.n))

        y = self.y.copy()
        x = self.x.copy()
        T = self.T

        if self.sur:
            xdsur = (1 / theta) * np.tile(self.y0, (1, self.lags))
            xdsur = np.hstack((np.array([[1 / theta]]), xdsur))

            ydsur = (1 / theta) * self.y0

            y = np.vstack((y, ydsur))
            x = np.vstack((x, xdsur))

            Td = Td + 1

        if self.noc:

            ydnoc = (1 / miu) * np.diag(self.y0)

            # Set to zero the prior mean on the first own lag for variables selected in the vector pos
            if self.pos is not None:
                ydnoc[self.pos, self.pos] = 0

            xdnoc = (1 / miu) * np.tile(np.diag(self.y0), (1, self.lags))
            xdnoc = np.hstack((np.zeros((self.n, 1)), xdnoc))

            y = np.vstack((y, ydnoc))
            x = np.vstack((x, xdnoc))

            Td = Td + self.n

        T = T + Td

        # ===== OUTPUT ===== #
        # Minnesota prior mean
        b = np.zeros((self.k, self.n))
        diagb = np.ones(self.n)

        # Set to zero the prior mean on the first own lag for variables selected in the vector pos
        if self.pos is not None:
            diagb[self.pos] = 0

        b[1:self.n + 1, :] = np.diag(diagb)

        # posterior mode of the VAR coefficients
        matA = x.T @ x + np.diag(1 / omega)
        matB = x.T @ y + np.diag(1 / omega) @ b
        betahat = np.linalg.solve(matA, matB)  # np.solve runs more efficiently that inverting a gigantic matrix

        # VAR residuals
        epshat = y - x @ betahat

        # Posterior mode of the covariance matrix
        sigmahat = (epshat.T @ epshat + PSI + (betahat - b).T @ np.diag(1 / omega) @ (betahat - b))
        sigmahat = sigmahat / (T + d + self.n + 1)

        # logML
        aaa = np.diag(np.sqrt(omega)) @ x.T @ x @ np.diag(np.sqrt(omega))
        bbb = np.diag(1 / np.sqrt(psi)) @ (epshat.T @ epshat + (betahat - b).T @ np.diag(1/omega) @
                                           (betahat-b)) @ np.diag(1 / np.sqrt(psi))

        eigaaa = np.linalg.eig(aaa)[0].real
        eigaaa[eigaaa < 1e-12] = 0
        eigaaa = eigaaa + 1

        eigbbb = np.linalg.eig(bbb)[0].real
        eigbbb[eigbbb < 1e-12] = 0
        eigbbb = eigbbb + 1

        logML = - self.n * T * np.log(np.pi) / 2
        logML = logML + sum(gammaln((T + d - np.arange(self.n)) / 2) - gammaln((d - np.arange(self.n)) / 2))
        logML = logML - T * sum(np.log(psi)) / 2
        logML = logML - self.n * sum(np.log(eigaaa)) / 2
        logML = logML - (T + d) * sum(np.log(eigbbb)) / 2

        if self.sur or self.noc:
            yd = np.vstack((ydsur, ydnoc))
            xd = np.vstack((xdsur, xdnoc))

            # prior mode of the VAR coefficients
            betahatd = b

            # VAR residuals at the prior mode
            epshatd = yd - xd @ betahatd

            aaa = np.diag(np.sqrt(omega)) @ xd.T @ xd @ np.diag(np.sqrt(omega))
            bbb = np.diag(1 / np.sqrt(psi)) @ (epshatd.T @ epshatd + (betahatd - b).T @ np.diag(1 / omega) @
                                               (betahatd - b)) @ np.diag(1 / np.sqrt(psi))

            eigaaa = np.linalg.eig(aaa)[0].real
            eigaaa[eigaaa < 1e-12] = 0
            eigaaa = eigaaa + 1

            eigbbb = np.linalg.eig(bbb)[0].real
            eigbbb[eigbbb < 1e-12] = 0
            eigbbb = eigbbb + 1

            # normalizing constant
            norm = - self.n * Td * np.log(np.pi) / 2
            norm = norm + sum(gammaln((Td + d - np.arange(self.n)) / 2) - gammaln((d - np.arange(self.n)) / 2))
            norm = norm - Td * sum(np.log(psi)) / 2
            norm = norm - self.n * sum(np.log(eigaaa)) / 2
            norm = norm - (T + d) * sum(np.log(eigbbb)) / 2
            logML = logML - norm

        if self.hyperpriors:
            logML = logML + self._log_gammma_pdf(x=lambda_,
                                                 k=self.priorcoef.loc['lambda', 'r_k'],
                                                 theta=self.priorcoef.loc['lambda', 'r_theta'])

            if self.sur:
                logML = logML + self._log_gammma_pdf(x=theta,
                                                     k=self.priorcoef.loc['theta', 'r_k'],
                                                     theta=self.priorcoef.loc['theta', 'r_theta'])

            if self.noc:
                logML = logML + self._log_gammma_pdf(x=miu,
                                                     k=self.priorcoef.loc['miu', 'r_k'],
                                                     theta=self.priorcoef.loc['miu', 'r_theta'])

            if self.mnpsi:
                toadd = self._log_invgammma_to_pdf(x=psi / (d - self.n - 1),
                                                   alpha=self.priorcoef.loc['alpha', 'PSI'],
                                                   beta=self.priorcoef.loc['beta', 'PSI'])
                logML = logML + sum(toadd)

        return logML, betahat, sigmahat

    def _logmlvar_formcmc(self, par):
        """
        This function computes the log-posterior (or the logML if hyperpriors=0),
        and draws from the posterior distribution of the coefficients and of the
        covariance matrix of the residuals of the BVAR of Giannone, Lenza and
        Primiceri (2012)
        """

        # hyperparameters
        lambda_ = par[0]
        d = self.n + 2
        theta = self.theta_min
        miu = self.miu_min

        if not self.mnpsi:
            psi = self.SS * (d - self.n - 1)

            if self.sur:
                theta = par[1]

                if self.noc:
                    miu = par[2]

            else:  # if self.sur == 0
                if self.noc:
                    miu = par[1]

        else:
            psi = par[1:self.n + 1]

            if self.sur:
                theta = par[self.n + 1]

                if self.noc:
                    miu = par[self.n + 2]

            else:
                if self.noc:
                    miu = par[self.n + 1]

        if not self.mnalpha:
            alpha = 2
        else:
            alpha = par[-1]

        # Check if parameters are outside of parameter space and, if so, return a very low value of the posterior
        cond_lower_bound = np.any([lambda_ < self.lambda_min,
                                   np.any(psi < self.psi_min),
                                   theta < self.theta_min,
                                   miu < self.miu_min,
                                   alpha < self.alpha_min])

        cond_upper_bound = np.any([lambda_ > self.lambda_max,
                                   np.any(psi > self.psi_max),
                                   theta > self.theta_max,
                                   miu > self.miu_max])

        if cond_lower_bound or cond_upper_bound:
            logML = -10e15
            betadraw = None
            drawSIGMA = None
            return logML, betadraw, drawSIGMA

        else:
            # Priors
            omega = np.zeros(self.k)
            omega[0] = self.vc

            for i in range(1, self.lags + 1):
                omega[1 + (i - 1) * self.n: 1 + i * self.n] = \
                    ((d - self.n - 1) * (lambda_ ** 2) * (1 / (i ** alpha))) / psi

            # Prior scale matrix for the covariance of the shocks
            PSI = np.diag(psi)

            Td = 0
            xdsur = np.array([]).reshape((0, self.k))
            ydsur = np.array([]).reshape((0, self.n))

            xdnoc = np.array([]).reshape((0, self.k))
            ydnoc = np.array([]).reshape((0, self.n))

            # dummy observations if sur and / or noc = 1
            y = self.y.copy()
            x = self.x.copy()
            T = self.T

            if self.sur:
                xdsur = (1 / theta) * np.tile(self.y0, (1, self.lags))
                xdsur = np.hstack((np.array([[1 / theta]]), xdsur))

                ydsur = (1 / theta) * self.y0

                y = np.vstack((y, ydsur))
                x = np.vstack((x, xdsur))

                Td = Td + 1

            if self.noc:
                ydnoc = (1 / miu) * np.diag(self.y0)
                # Set to zero the prior mean on the first own lag for variables selected in the vector pos
                ydnoc[self.pos, self.pos] = 0

                xdnoc = (1 / miu) * np.tile(np.diag(self.y0), (1, self.lags))
                xdnoc = np.hstack((np.zeros((self.n, 1)), xdnoc))

                y = np.vstack((y, ydnoc))
                x = np.vstack((x, xdnoc))

                Td = Td + self.n

            # ===== Output =====
            # minesota prior mean
            b = np.zeros((self.k, self.n))
            diagb = np.ones(self.n)
            # Set to zero the prior mean on the first own lag for variables selected in the vector pos
            diagb[self.pos] = 0
            b[1:self.n + 1, :] = np.diag(diagb)
            # self.b = b

            # posterior mode of the VAR coefficients
            matA = x.T @ x + np.diag(1 / omega)
            matB = x.T @ y + np.diag(1 / omega) @ b
            betahat = np.linalg.solve(matA, matB)  # np.solve runs more efficiently that inverting a gigantic matrix

            # VAR residuals
            epshat = y - x @ betahat

            # logMl
            T = T + Td

            aaa = np.diag(np.sqrt(omega)) @ x.T @ x @ np.diag(np.sqrt(omega))
            bbb = np.diag(1 / np.sqrt(psi)) @ (epshat.T @ epshat + (betahat - b).T @ np.diag(1 / omega) @
                                               (betahat - b)) @ np.diag(1 / np.sqrt(psi))

            eigaaa = np.linalg.eig(aaa)[0].real
            eigaaa[eigaaa < 1e-12] = 0
            eigaaa = eigaaa + 1

            eigbbb = np.linalg.eig(bbb)[0].real
            eigbbb[eigbbb < 1e-12] = 0
            eigbbb = eigbbb + 1

            logML = - self.n * T * np.log(np.pi) / 2
            logML = logML + sum(gammaln((T + d - np.arange(self.n)) / 2) - gammaln((d - np.arange(self.n)) / 2))
            logML = logML - T * sum(np.log(psi)) / 2
            logML = logML - self.n * sum(np.log(eigaaa)) / 2
            logML = logML - (T + d) * sum(np.log(eigbbb)) / 2

            # More terms for logML in case of more priors
            if self.sur or self.noc:
                yd = np.vstack((ydsur, ydnoc))
                xd = np.vstack((xdsur, xdnoc))

                # prior mode of the VAR coefficients
                betahatd = b

                # VAR residuals at the prior mode
                epshatd = yd - xd @ betahatd

                aaa = np.diag(np.sqrt(omega)) @ xd.T @ xd @ np.diag(np.sqrt(omega))
                bbb = np.diag(1 / np.sqrt(psi)) @ (epshatd.T @ epshatd + (betahatd - b).T @ np.diag(1 / omega) @
                                                   (betahatd - b)) @ np.diag(1 / np.sqrt(psi))

                eigaaa = np.linalg.eig(aaa)[0].real
                eigaaa[eigaaa < 1e-12] = 0
                eigaaa = eigaaa + 1

                eigbbb = np.linalg.eig(bbb)[0].real
                eigbbb[eigbbb < 1e-12] = 0
                eigbbb = eigbbb + 1

                # normalizing constant
                norm = - self.n * Td * np.log(np.pi) / 2
                norm = norm + sum(gammaln((Td + d - np.arange(self.n)) / 2) - gammaln((d - np.arange(self.n)) / 2))
                norm = norm - Td * sum(np.log(psi)) / 2
                norm = norm - self.n * sum(np.log(eigaaa)) / 2
                norm = norm - (T + d) * sum(np.log(eigbbb)) / 2
                logML = logML - norm

                if self.hyperpriors:
                    logML = logML + self._log_gammma_pdf(x=lambda_,
                                                         k=self.priorcoef.loc['lambda', 'r_k'],
                                                         theta=self.priorcoef.loc['lambda', 'r_theta'])

                    if self.sur:
                        logML = logML + self._log_gammma_pdf(x=theta,
                                                             k=self.priorcoef.loc['theta', 'r_k'],
                                                             theta=self.priorcoef.loc['theta', 'r_theta'])

                    if self.noc:
                        logML = logML + self._log_gammma_pdf(x=miu,
                                                             k=self.priorcoef.loc['miu', 'r_k'],
                                                             theta=self.priorcoef.loc['miu', 'r_theta'])

                    if self.mnpsi:
                        toadd = self._log_invgammma_to_pdf(x=psi / (d - self.n - 1),
                                                           alpha=self.priorcoef.loc['alpha', 'PSI'],
                                                           beta=self.priorcoef.loc['beta', 'PSI'])
                        logML = logML + sum(toadd)

            # takes a draw from the posterior of SIGMA and beta, if draw is on
            draw = self.mcmcfcast or self.mcmcstorecoef

            if not draw:
                betadraw = None
                drawSIGMA = None
            else:
                S = PSI + epshat.T @ epshat + (betahat - b).T @ np.diag(1 / omega) @ (betahat - b)

                E, V = np.linalg.eig(S)
                Sinv = V @ np.diag(1 / np.abs(E)) @ V.T
                eta = np.random.multivariate_normal(mean=np.zeros(self.n),
                                                    cov=Sinv,
                                                    size=T+d)
                drawSIGMA = np.linalg.solve(eta.T @ eta, np.eye(self.n))
                cholSIGMA = self._cholred((drawSIGMA + drawSIGMA.T) / 2)
                cholZZinv = self._cholred(np.linalg.solve(x.T @ x + np.diag(1 / omega), np.eye(self.k)))
                betadraw = betahat + cholZZinv.T @ np.random.normal(size=betahat.shape) @ cholSIGMA

            return logML, betadraw, drawSIGMA

    @staticmethod
    def _gamma_coef(mode, sd):
        k = (2 + mode ** 2 / sd ** 2 + np.sqrt((4 + mode ** 2 / sd ** 2) * mode ** 2 / sd ** 2)) / 2
        theta = np.sqrt(sd ** 2 / k)
        return k, theta

    @staticmethod
    def _log_gammma_pdf(x, k, theta):
        r = (k - 1) * np.log(x) - x / theta - k * np.log(theta) - gammaln(k)
        return r

    @staticmethod
    def _log_invgammma_to_pdf(x, alpha, beta):
        r = alpha * np.log(beta) - (alpha + 1) * np.log(x) - beta * (1 / x) - gammaln(alpha)
        return r

    @staticmethod
    def _cholred(s):
        d, v = np.linalg.eig((s + s.T) / 2)
        d = d.real
        scale = np.diag(s).mean() * 1e-12
        J = d > scale
        C = np.zeros(s.shape)
        C[J, :] = (v[:, J] @ np.diag(d[J] ** 0.5)).T
        return C


class OLS1(object):
    """
    This is a simple OLS regression, but with a more leaner and simple layout
    """

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


class CRBVAR(object):

    def __init__(self, data, lags, hz=24, vc=10e6, stationary_prior=None, crit=1e-16,
                 hyperpriors=True, mnpsi=True, mnalpha=False, sur=True, noc=True,
                 fcast=False, mcmc=False, ndraws=20000, ndrawsdiscard=None, mcmcconst=1,
                 mcmcfcast=True, mcmcstorecoef=True, verbose=False, resample_method='full'):
        """
        This class implements the "Cube-Root" Bayesian VAR from Climadomo, Giannone, Lenza, Monti and Sokol (2020).
        The main idea of the models is to use the BVARGLP class to estimate a quarterly VAR and "monthlize" it for
        a state-space model capable of dealing with missing data and mixed frequancy data.

        This class only accepts data with at leaset one monthly time series. Quarterly variable are allowed but
        must be in the same pandas.DataFrame with a monthly index.

        @param hyperpriors: False = no priors on hyperparameters
                            True = reference priors on hyperparameters (default)
                            [NOTE: hyperpriors on psi calibrated for data expressed in
                            4 x logs, such as 4 x log(GDP). Thus if interest rate is in
                            percentage, divide by 100]

        @param vc: prior variance in the MN prior for the coefficients multiplying
                   the contant term (Default: vc=10e6)

        @param stationary_prior: names of the variables that enter the VAR in first
                                 differences and for which one might want to set the prior mean
                                 on the coefficient on the first own lag in the MN prior and the
                                 prior mean of the sum-of-coefficients prior to 0 (instead of
                                 the typical 1)

        @param mnpsi: False = diagonal elements of the scale matrix of the IW prior on
                      the covariance of the residuals NOT treated as
                      hyperparameters (set to the residual variance of an AR(1))
                      True = diagonal elements of the scale matrix of the IW prior on
                      the covariance of the residuals treated as hyperparameters (default)

        @param mnalpha:  False = Lag-decaying parameter of the MN prior set to 2 and
                         NOT treated as hyperparameter (default)
                         True = Lag-decaying parameter of the MN prior treated as
                         hyperparameter

        @param sur: False = single-unit-root prior is OFF
                    True = single-unit-root prior is ON and its std is treated as an
                    hyperparameter (default)

        @param noc: False = no-cointegration (sum-of coefficients) prior is OFF
                    True = no-cointegration (sum-of coefficients) is ON and its std is
                    treated as an hyperparameter (default)

        @param fcast: False = does not generate forecasts at the posterior mode
                      True = generates forecasts at the posterior mode (default)

        @param hz: number of quarters for which it generates forecasts (default: hz=8)

        @param mcmc: False = does not run the MCMC (default)
                     True = runs the MCMC after the maximization

        @param ndraws: number of draws in the MCMC (default: Ndraws=20000)

        @param ndrawsdiscard: number of draws initially discarded to allow convergence
                              in the in the MCMC (default=Ndraws/2)

        @param mcmcconst: scaling constant for the MCMC (should be calibrated to achieve
                          an acceptance rate of approx 25%) (default: MCMCconst=1)

        @param mcmcfcast: False = does not generate forecasts when running the MCMC
                          True = generates forecasts while running the MCMC
                          (for each draw of the hyperparameters the code takes a
                          draw of the VAR coefficients and shocks, and generates
                          forecasts at horizons hz) (default).

        @param mcmcstorecoef: False = does not store the MCMC draws of the VAR
                              coefficients and residual covariance matrix
                              True = stores the MCMC draws of the VAR coefficients and
                              residual covariance matrix (default)

        @param verbose: Prints relevant information during the estimation.

        @param crit: precision for convergence criteria.

        @param resample_method: 'full' only includes quarters that have all of its data available.
                                'last' uses the last observation available for each quarter.
        """

        assert data.index.inferred_freq == 'M', "input 'data' must be monthly and recognized by pandas."

        self.data = data

        if resample_method == 'full':
            self.data_quarterly = self._get_quarterly_df()
        elif resample_method == 'last':
            self.data_quarterly = data.resample('Q').last().dropna()
        else:
            raise NotImplementedError('resample method not implemented')

        self.lags = lags
        self.hyperpriors = hyperpriors
        self.vc = vc
        self.stationary_prior = stationary_prior
        self.mnalpha = mnalpha
        self.mnpsi = mnpsi
        self.sur = sur
        self.noc = noc
        self.fcast = fcast
        self.hz = hz
        self.mcmc = mcmc
        self.ndraws = ndraws
        self.ndrwasdiscard = int(ndraws/2) if ndrawsdiscard is None else ndrawsdiscard
        self.mcmccosnt = mcmcconst
        self.mcmcfcast = mcmcfcast
        self.mcmcstorecoef = mcmcstorecoef
        self.verbose = verbose
        self.crit = crit

        self.bvar_quarterly = BVARGLP(data=self.data_quarterly,
                                      lags=lags,
                                      hyperpriors=hyperpriors,
                                      vc=vc,
                                      stationary_prior=stationary_prior,
                                      mnpsi=mnpsi,
                                      mnalpha=mnalpha,
                                      sur=sur,
                                      noc=noc,
                                      fcast=fcast,
                                      hz=hz,
                                      mcmc=mcmc,
                                      ndraws=ndraws,
                                      ndrawsdiscard=ndrawsdiscard,
                                      mcmcconst=mcmcconst,
                                      mcmcfcast=mcmcfcast,
                                      mcmcstorecoef=mcmcstorecoef,
                                      verbose=verbose,
                                      crit=crit)

        betahat = self.bvar_quarterly.betahat
        sigmahat = self.bvar_quarterly.sigmahat
        k, n = betahat.shape

        _, _, _, aa, _, qq, c2, c1, CC, _, _, _ = self._build_monthly_ss(betahat, sigmahat)

        qqKF = np.zeros((n * lags, n * lags))
        qqKF[:n, :n] = qq.real

        # Next line is just a weird reshaping of the starting state
        initX = np.flip(self.data_quarterly.iloc[:lags].values, axis=0).T.reshape(-1, 1, order='F').reshape(-1)
        initV = np.eye(initX.shape[0]) * 1e-7

        kf = KalmanFilter(transition_matrices=aa,
                          transition_offsets=c2,
                          transition_covariance=qqKF,
                          observation_matrices=CC,
                          observation_offsets=c1,
                          observation_covariance=np.zeros((n, n)),
                          initial_state_mean=initX,
                          initial_state_covariance=initV)

        # Data format for Kalman Filter
        kf_data = ma.masked_invalid(self.data.values)
        self.logLik = kf.loglikelihood(kf_data)

        new_index = pd.date_range(start=self.data.index[0], periods=self.data.shape[0] + hz, freq='M')
        forecast_vector = ma.masked_invalid(self.data.reindex(new_index).values)
        smoothed_states = kf.smooth(forecast_vector)[0][:, :n]

        self.smoothed_states = pd.DataFrame(data=smoothed_states,
                                            index=new_index,
                                            columns=self.data.columns)

    def _get_quarterly_df(self):
        df = self.data

        df = df.dropna(how='all')

        is_quarterly = pd.Series(index=df.columns)
        for col in df.columns:
            freq = pd.infer_freq(df[col].dropna().index)
            is_quarterly.loc[col] = freq[0] == 'Q'

        quarterly_variables = list(is_quarterly[is_quarterly].index)
        obs2keep = df.drop(quarterly_variables, axis=1)
        obs2keep = obs2keep.resample('Q').count() == 3
        obs2keep = obs2keep.all(axis=1)

        df_quarterly = df.resample('Q').mean()
        df_quarterly = df_quarterly[obs2keep.values]

        return df_quarterly

    def _build_monthly_ss(self, beta, sigma):
        """
        Yt = c1 + CC @ Xt
        Xt+1 = C2 + AA @ Xt
        """

        qqflag = False
        k, n = beta.shape
        lags = int((k - 1) / n)

        # state equations
        AA = np.zeros((n * lags, n * lags))  # Autoregressive coefficients
        AA[0:n, :] = beta[1:, :].T
        AA[n:, 0:-n] = np.eye(n * (lags - 1))

        C2 = np.zeros(n * lags)  # constant
        C2[0:n] = beta[0, :].T

        # measurement equation
        CC = np.zeros((n, n * lags))  # maps first n states (current month) into observables
        CC[0:n, 0:n] = np.eye(n)

        c1 = np.zeros(n)  # constant

        # "shock" impact matrix
        BB = np.linalg.cholesky(sigma)

        aa = self._cuberoot(AA)
        aa2 = aa @ aa

        if lags > 1:
            kronmat = aa2[0:n, 0:n] - aa[0:n, 0:n] @ np.linalg.lstsq(np.tile(np.eye(n), (lags - 1, 1)),
                                                                     aa2[n:, 0:n],
                                                                     rcond=None)[0]
            Lambda, PP = np.linalg.eig(kronmat)
            PPinv = np.linalg.inv(PP)  # this could be optimized
            invmat = np.kron(PP, PP) @ (np.kron(PPinv, PPinv) / (1 + np.kron(Lambda, Lambda)))

        else:
            kronmat = aa
            Lambda, PP = np.linalg.eig(kronmat)
            PPinv = np.linalg.inv(PP)  # this could be optimized
            invmat = np.kron(PP, PP) @ (np.kron(PPinv, PPinv) / (1 + np.kron(Lambda, Lambda) +
                                                                 np.kron(Lambda ** 2, Lambda ** 2)))

        qq = (invmat @ sigma.reshape(-1, 1)).reshape(n, n)
        qq = 0.5 * (qq + qq.T)
        try:
            bb = np.linalg.cholesky(qq)

        except np.linalg.LinAlgError:
            d, v = np.linalg.eig(qq)
            d[d < 0] = 1e-10
            qq = v @ np.diag(np.abs(d)) @ np.matrix(v).H
            bb = np.linalg.cholesky(qq)
            qqflag = True

        c2 = np.linalg.inv(np.eye(aa.shape[0]) + aa + aa2) @ C2
        maxEig = np.abs(Lambda).max()
        minEig = np.abs(Lambda).min()

        return AA, BB, C2, aa, bb, qq, c2, c1, CC, qqflag, maxEig, minEig

    @staticmethod
    def _cuberoot(big_a):

        d, v = np.linalg.eig(big_a)
        s = np.sign(d.real)

        a = v @ np.diag(((d * s) ** (1 / 3)) * s) @ np.linalg.inv(v)

        if a.imag.sum() > 1e-4:
            msg = "This cube root is not real"
            raise ValueError(msg)
        else:
            a = a.real

        return a
