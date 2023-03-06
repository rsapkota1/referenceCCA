import warnings
from typing import Union, Iterable

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import SGDRegressor, Ridge, ElasticNet
from sklearn.utils._testing import ignore_warnings

from cca_zoo.utils import _process_parameter, _check_converged_weights
#from cca_zoo.models._iterative._base import _BaseIterative
from _baseiterative import _BaseIterative
from cca_zoo.models._iterative._pls_als import _PLSInnerLoop


class ElasticCCA(_BaseIterative):
    r"""
    Fits an elastic CCA by iterating elastic net regressions.

    By default, ElasticCCA uses CCA with an auxiliary variable target i.e. MAXVAR configuration

    .. math::

        w_{opt}, t_{opt}=\underset{w,t}{\mathrm{argmax}}\{\sum_i \|X_iw_i-t\|^2 + c\|w_i\|^2_2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        t^Tt=n

    But we can force it to attempt to use the SUMCOR form which will approximate a solution to the problem:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + c\|w_i\|^2_2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=n

    :Example:

    >>> from cca_zoo.models import ElasticCCA
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = ElasticCCA(c=[1e-1,1e-1],l1_ratio=[0.5,0.5], random_state=0)
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.9316638])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        deflation="cca",
        max_iter: int = 100,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-9,
        c: Union[Iterable[float], float] = None,
        l1_ratio: Union[Iterable[float], float] = None,
        maxvar: bool = True,
        stochastic=False,
        positive: Union[Iterable[bool], bool] = None,
        verbose=0,
        reference= np.ndarray,
        lr=float
    ):
        """
        Constructor for ElasticCCA

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, views will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param deflation: the type of deflation.
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param initialization: either string from "pls", "cca", "random", "uniform" or callable to initialize the score variables for _iterative methods
        :param tol: tolerance value used for early stopping
        :param c: lasso alpha
        :param l1_ratio: l1 ratio in lasso subproblems
        :param maxvar: use auxiliary variable "maxvar" formulation
        :param stochastic: use _stochastic regression optimisers for subproblems
        :param positive: constrain model weights to be positive
        """
        self.c = c
        self.l1_ratio = l1_ratio
        self.maxvar = maxvar
        self.stochastic = stochastic
        self.positive = positive
        self.reference = reference
        self.lr = lr
        if self.positive is not None and stochastic:
            self.stochastic = False
            warnings.warn(
                "Non negative constraints cannot be used with _stochastic regressors. Switching to _stochastic=False"
            )
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            deflation=deflation,
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
        )

    def _set_loop_params(self):
        self.loop = _ElasticInnerLoop(
            max_iter=self.max_iter,
            c=self.c,
            l1_ratio=self.l1_ratio,
            maxvar=self.maxvar,
            tol=self.tol,
            stochastic=self.stochastic,
            positive=self.positive,
            random_state=self.random_state,
            verbose=self.verbose,
            reference=self.reference,
            lr = self.lr
        )

    def _check_params(self):
        self.c = _process_parameter("c", self.c, 0, self.n_views)
        self.l1_ratio = _process_parameter("l1_ratio", self.l1_ratio, 0, self.n_views)
        self.positive = _process_parameter(
            "positive", self.positive, False, self.n_views
        )


class SCCA_IPLS(ElasticCCA):
    r"""
    Fits a sparse CCA model by _iterative rescaled lasso regression. Implemented by ElasticCCA with l1 ratio=1

    For default maxvar=False, the optimisation is given by:

    :Maths:

    .. math::

        w_{opt}=\underset{w}{\mathrm{argmax}}\{\sum_i\sum_{j\neq i} \|X_iw_i-X_jw_j\|^2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        w_i^TX_i^TX_iw_i=n

    :Citation:

    Mai, Qing, and Xin Zhang. "An _iterative penalized least squares approach to sparse canonical correlation analysis." Biometrics 75.3 (2019): 734-744.

    For maxvar=True, the optimisation is given by the ElasticCCA problem with no l2 regularisation:

    :Maths:

    .. math::

        w_{opt}, t_{opt}=\underset{w,t}{\mathrm{argmax}}\{\sum_i \|X_iw_i-t\|^2 + \text{l1_ratio}\|w_i\|_1\}\\

        \text{subject to:}

        t^Tt=n

    :Citation:

    Fu, Xiao, et al. "Scalable and flexible multiview MAX-VAR canonical correlation analysis." IEEE Transactions on Signal Processing 65.16 (2017): 4150-4165.


    :Example:

    >>> from cca_zoo.models import SCCA_IPLS
    >>> import numpy as np
    >>> rng=np.random.RandomState(0)
    >>> X1 = rng.random((10,5))
    >>> X2 = rng.random((10,5))
    >>> model = SCCA_IPLS(c=[0.001,0.001], random_state=0)
    >>> model.fit((X1,X2)).score((X1,X2))
    array([0.99998761])
    """

    def __init__(
        self,
        latent_dims: int = 1,
        scale: bool = True,
        centre=True,
        copy_data=True,
        random_state=None,
        deflation="cca",
        c: Union[Iterable[float], float] = None,
        max_iter: int = 100,
        maxvar: bool = False,
        initialization: Union[str, callable] = "pls",
        tol: float = 1e-9,
        stochastic=False,
        positive: Union[Iterable[bool], bool] = None,
        verbose=0,
        reference=np.ndarray,
        lr=float
    ):
        """
        Constructor for SCCA_IPLS

        :param latent_dims: number of latent dimensions to fit
        :param scale: normalize variance in each column before fitting
        :param centre: demean data by column before fitting (and before transforming out of sample
        :param copy_data: If True, views will be copied; else, it may be overwritten
        :param random_state: Pass for reproducible output across multiple function calls
        :param max_iter: the maximum number of iterations to perform in the inner optimization loop
        :param maxvar: use auxiliary variable "maxvar" form
        :param initialization: either string from "pls", "cca", "random", "uniform" or callable to initialize the score variables for _iterative methods
        :param tol: tolerance value used for early stopping
        :param c: lasso alpha
        :param stochastic: use _stochastic regression optimisers for subproblems
        :param positive: constrain model weights to be positive
        """
        super().__init__(
            latent_dims=latent_dims,
            scale=scale,
            centre=centre,
            copy_data=copy_data,
            max_iter=max_iter,
            initialization=initialization,
            tol=tol,
            c=c,
            l1_ratio=1,
            maxvar=maxvar,
            stochastic=stochastic,
            positive=positive,
            random_state=random_state,
            deflation=deflation,
            verbose=verbose,
            reference=reference,
            lr= lr
        )


class _ElasticInnerLoop(_PLSInnerLoop):
#class _ElasticInnerLoop():
    def __init__(
        self,
        max_iter: int = 100,
        tol=1e-9,
        c=None,
        l1_ratio=None,
        maxvar=True,
        stochastic=True,
        positive=None,
        random_state=None,
        verbose=0,
        reference=None,
        lr=None,
        **kwargs,
    ):
        super().__init__(
            max_iter=max_iter, tol=tol, random_state=random_state, verbose=verbose
        )
        self.stochastic = stochastic
        self.c = c
        self.l1_ratio = l1_ratio
        self.positive = positive
        self.maxvar = maxvar
        self.reference = reference
        self.lr= lr

    def _initialize(self, views):
        self.regressors = []
        for alpha, l1_ratio, positive in zip(self.c, self.l1_ratio, self.positive):
            if self.stochastic:
                self.regressors.append(
                    SGDRegressor(
                        penalty="elasticnet",
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        fit_intercept=False,
                        tol=self.tol,
                        warm_start=True,
                        random_state=self.random_state,
                    )
                )
            elif alpha == 0:
                self.regressors.append(
                    Ridge(
                        alpha=self.tol,
                        fit_intercept=False,
                        positive=positive,
                    )
                )
            else:
                self.regressors.append(
                    ElasticNet(
                        alpha=alpha,
                        l1_ratio=l1_ratio,
                        fit_intercept=False,
                        warm_start=True,
                        positive=positive,
                        random_state=self.random_state
                    )
                )

    def _update_view(self, views, view_index: int):
        """
        :param view_index: index of view being updated
        :return: updated weights
        """
        #self.reference = self.reference.fillna(0)
        if self.maxvar:
            # For MAXVAR we rescale the targets
            target = self.scores.mean(axis=0)
            target /= np.linalg.norm(target) / np.sqrt(self.n)
        else:
            target = self.scores[view_index - 1]

        # Solve the elastic regression
        
        #Implement Gradient Descent to update the weights
        all_c = self.c[:-1]

        final_c = all_c[view_index]
        if view_index==0:
            self.weights[view_index] = self.gradient_update_X_ref(views[view_index],target, self.weights[view_index], self.weights[view_index-1], final_c, lr=self.lr)
            #self.weights[view_index]=self.scca_gradient_descent_update_U(views[view_index],views[view_index-1], self.weights[view_index], self.weights[view_index-1], 0.001, 0.0001, 0.001, 0.0001 )
        else:
            self.weights[view_index]=self.gradient_update_Y_ref(target,views[view_index], self.weights[view_index], self.weights[view_index-1], final_c, lr=self.lr)

        # self.weights[view_index] = self._elastic_solver(
        #     views[view_index], target, view_index
        # )
        # For SUMCOR we rescale the projections
        if not self.maxvar:
            _check_converged_weights(self.weights[view_index], view_index)
            self.weights[view_index] = self.weights[view_index] / (
                np.linalg.norm(views[view_index] @ self.weights[view_index])
                / np.sqrt(self.n)
            )
        self.scores[view_index] = views[view_index] @ self.weights[view_index]

    @ignore_warnings(category=ConvergenceWarning)
    def _elastic_solver(self, X, y, view_index):
        return self.regressors[view_index].fit(X, y.ravel()).coef_
    

    def gradient_update_X_ref(self,X, T, U, V, l1_ratio, lr):
        grad_Wx = 2 * X.T @ (X @ U - T) + l1_ratio * 2 * X.T @( X @ U - self.reference) #Derivative of loss function wrt U
        new_weights = U - lr * grad_Wx #Final Weight = Original Weight - LR * gradient
        return new_weights
    
    def gradient_update_Y_ref(self,T, Y, U, V, l1_ratio, lr):
        grad_Wy = -1 * (2 * Y.T @ (T - Y @ V) + l1_ratio * 2 * Y.T @( Y @ V - self.reference)) #Derivative of loss function wrt V
        new_weights = V - lr * grad_Wy #Final Weight = Original Weight - LR * gradient
        return new_weights
    

    def _objective(self, views):
        c = np.array(self.c)
        ratio = np.array(self.l1_ratio)
        l1 = c * ratio
        l2 = c * (1 - ratio)
        total_objective = 0
        target = self.scores.mean(axis=0)
        reference = self.reference
        lr = self.lr
        for i, _ in enumerate(views):
            if self.maxvar:
                target /= np.linalg.norm(target) / np.sqrt(self.n)
            objective = np.linalg.norm(views[i] @ self.weights[i] - target) ** 2 / (
                2 * self.n
            )
            #self.reference = self.reference.fillna(0)
            l1_objective = l1[i] * (np.linalg.norm((views[i] @ self.weights[i] - self.reference))** 2 / (
                2 * self.n)
            )
            
            #l1_pen = l1[i] * np.linalg.norm(self.weights[i], ord=1)
            #l2_pen = l2[i] * np.linalg.norm(self.weights[i], ord=2)
            
            total_objective += objective + l1_objective
        #print(total_objective)
        return total_objective

    def _early_stop(self) -> bool:
        # Some kind of early stopping
        #return False
        if np.abs(self.track["objective"][-2] - self.track["objective"][-1]) < self.tol:
            return True
        else:
            return False