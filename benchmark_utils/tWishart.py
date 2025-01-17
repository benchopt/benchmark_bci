#################################
# Utils for SPD manifold        #
#################################
from scipy.stats import ortho_group, norm, uniform
import numpy as np
import numpy.linalg as la
from scipy.linalg import pinvh, sqrtm
from scipy.linalg.lapack import dtrtri
from scipy.special import gammaln, betaln

import pymanopt
from pymanopt import Problem
from pymanopt.optimizers import ConjugateGradient, SteepestDescent
from pymanopt.manifolds.manifold import RiemannianSubmanifold


def sym(x):
    """
    Computes the symmetric part of a given matrix

    Parameters
    ----------
    x : array
        Square Matrix.

    Returns
    -------
    array
        Symmetric part of x.

    """
    return 0.5 * (x + x.T)


def generate_random_SPD(p, cond, random_state=None):
    """
    Generate pxp SPD matrix of the form U^TDU and with conditionning number
    where U is pxp matrix drawn uniformly from the orthogonal group,
    and D is pxp diagonal matrix

    Parameters
    ----------
    p : int
        dimension of the random matrix.
    cond : float
        Conditionning number (in the sense of L2 norm)
    random_state : int, RandomState instance or None
        Controls the pseudo random number generation

    Returns
    -------
    array
        Random SPD matrix.

    """
    U = ortho_group.rvs(p, random_state=random_state)
    d = np.zeros(p)
    if p > 2:
        d[:p-2] = uniform.rvs(loc=1/np.sqrt(cond),
                              scale=np.sqrt(cond)-1/np.sqrt(cond),
                              size=p-2, random_state=random_state)
    d[p-2] = 1/np.sqrt(cond)
    d[p-1] = np.sqrt(cond)
    return U @ np.diag(d) @ U.T


class SPD(RiemannianSubmanifold):
    """
    Class of SPD manifold, endowed with the Fisher Information Metric of the
    form :
        g_S(A,B) = alpha*tr(AS^{-1}BS^{-1})+beta*tr((AS^{-1})tr(BS^{-1})
    where S is a SPD matrix and A,B are symmetric matrices (on the tangent
    space of S)

    Parameters
    ----------
    p : int
        number of time samples.
    alpha: float
        First parameter of the Fisher Information Metric on the SPD manifold
    beta :
        Second parameter of the Fisher Information Metric on the SPD manifold

    Attributes
    ----------
    p_ : int
        number of time samples.
    alpha_: float
        First parameter of the Fisher Information Metric on the SPD manifold
    beta_ : float
        Second parameter of the Fisher Information Metric on the SPD manifold

    """
    def __init__(self, p, alpha, beta):
        """Init.
        """
        if (alpha <= 0 or alpha*p+beta <= 0):
            raise NameError('value of alpha and/or beta invalid')

        self._p = p
        self._alpha = alpha
        self._beta = beta
        name = f"Manifold of positive definite {p}x{p} matrices"
        dimension = int(p*(p+1)/2)
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return np.sqrt(self.dim)

    def random_point(self, cond=100, random_state=None):
        return generate_random_SPD(self._p, cond, random_state=random_state)

    def random_tangent_vector(self, point):
        return self.projection(point, norm.rvs(size=(self._p, self._p)))

    def zero_vector(self, point):
        return np.zeros((self._p, self._p))

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        """
        Computes the Fisher Information Metric between two vectors of
        the tangent space (two symmetric matrices) at a given SPD matrix.

        Parameters
        ----------
        point : array, shape (p,p)
            SPD matrix.
        tangent_vector_a : array (p,p)
            Vector (symmetric matrix) on the tangent space at `point`.
        tangent_vector_b : array, shape (p,p)
            Vector (symmetric matrix) on the tangent space at `point`.

        Returns
        -------
        float
            Fisher Information Meric between `tangent_vector_a`
            and `tangent_vector_b` at `point`.

        """
        L = la.cholesky(point)
        iL, _ = dtrtri(L, lower=1)
        coor_a = iL @ tangent_vector_a @ iL.T
        if tangent_vector_a is tangent_vector_b:
            coor_b = coor_a
        else:
            coor_b = iL @ tangent_vector_b @ iL.T
        return self._alpha * np.tensordot(coor_a, coor_b, axes=point.ndim) +\
            self._beta * np.trace(coor_a) * np.trace(coor_b)

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner_product(point, tangent_vector,
                                          tangent_vector))

    def projection(self, point, vector):
        return sym(np.real(vector))

    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        """
        Computes the Riemannian gradient of a function at a given point of the
        manifold from its Euclidean gradient at this point.

        Parameters
        ----------
        point : array, shape (p,p)
            SPD matrix.
        euclidean_gradient : array, shape(p,p)
            Euclidean gradient of a function at `point`.

        Returns
        -------
        array, shape (p,p)
            Riemannian gradient at `point`.

        """
        return (point @ sym(euclidean_gradient) @ point) / self._alpha -\
            (self._beta/(self._alpha*(self._alpha + self._p * self._beta))) *\
            np.trace(euclidean_gradient @ point) * point

    def retraction(self, point, tangent_vector):
        return np.real(sym(point + tangent_vector
                           + 0.5 * tangent_vector
                           @ la.solve(point, tangent_vector)))

    def transport(self, point_a, point_b, tangent_vector_a):
        """
        Computes the parallel transport induced by the Fisher Informaion Metric

        Parameters
        ----------
        point_a : array, shape (p,p)
            "strating" SPD matrix.
        point_b : array, shape (p,p)
            "Ending" SPD matrix.
        tangent_vector_a : array, shape (p,p)
            Vector for the tangent space of `point_a`.

        Returns
        -------
        array, shape (p,p)
            Parallel transport of `tangent_vector_a` to the tangent space of
            `point_b`.

        """
        # (point_b point_a^{-1})^{1/2}
        tmp = sqrtm(la.solve(point_a, point_b).T)
        return tmp @ tangent_vector_a @ tmp.T

    def dist(self, point_a, point_b):
        """
        Induced distance by the Fisher Information Metric
        d(A,B)² = alpha* tr(log(AB^{-1})²) +beta*tr(log(AB^{-1}))²

        Parameters
        ----------
        point_a : array, shape (p,p)
            SPD matrix.
        point_b : array, shape(p,p)
            SPD matrix.

        Returns
        -------
        float
            Distance between `point_a` and `point_b`.

        """
        L = la.cholesky(point_a)
        iL, _ = dtrtri(L, lower=1)
        tmp = iL @ point_b @ iL.T
        log_eigs = np.log(la.eigh(tmp)[0])  # replace by some Cholesky ???
        return (self._alpha * np.sum(log_eigs**2)
                + self._beta * np.sum(log_eigs)**2)**0.5

##################################
# Utils for t- Wishart estimation#
##################################

# cost and grad for t- Wishart


def t_wish_cost(R, S, n, df):
    """
    computes the cost function (negative log-likelihood of t-Wishart up to
    a multiplicative positive constant)

    Parameters
    ----------
    R : array
        Symmetric positive definite matrix, plays the role of the distribution
        center.
    S : ndarray
        Samples, must be symmetric definite positive matrices of the same
        shape as `R`.
    n : int
        Degrees of freedom of the t-Wishart distribution.
    df : float
        Degrees of freedom of the t- modelling.

    Returns
    -------
    float
        The negative log-likelihood of the samples at `R` (divided by n*number
        of samples).

    """
    k, p, _ = S.shape
    a = np.einsum('kii->k', la.solve(R, S))  # tr(inv(R)@S[k])
    return 1/2 * np.log(la.det(R)) - np.sum(-(df+n*p)/2*np.log(1+a/df))/n/k


def t_wish_egrad(R, S, n, df):
    """
    Computes the Riemannian gradient of the cost (with respect to the Fisher
    Information Metric of t-Wishart)

    Parameters
    ----------
    R : array, shape (p,p)
        Symmetric positive definite matrix, plays the role of the
        distribution's center.
    S : ndarray, shape (K,p,p)
        Samples, must be symmetric definite positive matrices of the same
        shape as `R`.
    n : int
        Degrees of freedom of the t-Wishart distribution.
    df : float
        Degrees of freedom of the t- modelling.

    Returns
    -------
    array, shape(p,p)
        Riemannian gradient of the cost of samples at `R`.

    """
    k, p, _ = S.shape
    a = np.einsum('kii->k', la.solve(R, S))  # tr(inv(R)@S[k])
    psi = np.einsum('k,kij->ij', (df+n*p)/(df+a), S)
    return la.solve(R, la.solve(R.T, ((R - psi/n/k)/2).T).T)


def log_generator_density(n, p, df, neglect_df_terms=False):
    """
    Returns the log of the density generator function of the t-Wishart
    distribution

    Parameters
    ----------
    n : int
        Number of time samples, also called the degree of freedom.
    p : int
        Number of channels, which represents the dimension of SPD matrices.
    df : float
        Degree of freedom (shape parameter) of the t- Wishart distribution.
    neglect_df_terms : bool, optional
        If true, the output does not take into account constant terms
        depending on `df`. The default is False.

    Returns
    -------
    log_h : function
        Log of the generator density function of t- Wishart of parameters n
        and df and whose matrix dimension is p.

    """
    def log_h(t):
        if df == np.inf:
            return -0.5*t
        else:
            if neglect_df_terms:
                return -0.5*(df+n*p)*np.log(1+t/df)
            else:
                cst = -betaln(df/2, n*p/2)+gammaln(n*p/2)
                return -0.5*(df+n*p)*np.log(1+t/df)-0.5*n*p*np.log(0.5*df)+cst

    return log_h


# estimation algorithms to estimate MLE for the center of t- Wishart

def t_wish_est(S, n, df, algo="RCG"):
    """
    computes iteratively the MLE for the center of samples drawn from
    t-Wishart with parameters n and df using the Riemannian Gradient Descent
    or Riemannian Conjugate Gradient algorithm.

    Parameters
    ----------
    S : ndarry, shape (K,p,p)
        samples, symmetric definite matrices.
    n : int
        Degrees of freedom.
    df : float
        Degrees of freedom of the t- modelling.
    algo : str, default="RCG"
        Type of first-order Riemannian optimization algorithm,
        either "RCG" for conjugate gradient or "RGD" for gradient descent.

    Returns
    -------
    array, shape (p,p)
        MLE of the center parameter.
    """
    p = S.shape[1]
    if df == np.inf:
        return np.mean(S, axis=0)/n

    alpha = n/2*(df+n*p)/(df+n*p+2)
    beta = n/2*(alpha-n/2)
    manifold = SPD(p, alpha, beta)

    @pymanopt.function.numpy(manifold)
    def cost(R):
        return t_wish_cost(R, S, n, df)

    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(R):
        return t_wish_egrad(R, S, n, df)

    problem = Problem(manifold=manifold, cost=cost,
                      euclidean_gradient=euclidean_gradient)
    assert algo in ["RCG", "RGD"], "Wrong Algorithm Name"
    if algo == "RCG":  # conjugate gradient
        optimizer = ConjugateGradient(verbosity=0)
    else:
        optimizer = SteepestDescent(verbosity=0)
    optim = optimizer.run(problem, initial_point=np.eye(S.shape[-1]))
    return optim.point


# estimate dof for t-Wishart

def kurtosis_estimation(samples, n, center, rmt=False):
    """
    Estimates the degree of freedom (shape parameter) of t- wishart
    distribution given its parameter n and its center matrix, based
    on the provided samples

    Parameters
    ----------
    samples : ndarray, shape (n_trials,n_channels, n_channels)
        ndarray of SPD matrices.
    n : int
        Parameter of the t- Wishart distribution.
    center : array, shape (p,p)
        SPD matrix, plays the role of the center matrix.
    rmt : bool, optional.
        If true, the RMT correction is applied in the estimation process.
        The default is False.

    Returns
    -------
    float
        Degree of freedom (shape parameter) of the t- Wishart distribution.

    """
    # traces of whitened samples
    K, p, _ = samples.shape
    traces = np.einsum("kij,ji->k", samples, pinvh(center))
    # kappa = (E(Q²)/E(Q)²)*(np/(np+2))-1
    kappa = ((n*p)/(n*p+2))*np.mean(traces**2)/(np.mean(traces)**2)-1
    if rmt:
        kappa = (kappa+1)/(1-p/(n*K))-1  # kappa+1=1/theta
    if kappa > 0:
        return 4+2/kappa
    else:
        if kappa == 0:
            return np.inf
    # if kappa<0, df<4 for eg we can choose df=2
    return 2
