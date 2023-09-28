from benchopt import safe_import_context
from benchmark_utils.augmented_dataset import AugmentedBCISolver


with safe_import_context() as import_ctx:
    from sklearn.pipeline import make_pipeline
    from sklearn.pipeline import FunctionTransformer
    from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
    from sklearn.utils.extmath import softmax

    from pyriemann.estimation import Covariances
    from pyriemann.utils.base import logm,_matrix_operator,sqrtm

    from joblib import Parallel, delayed

    from skorch.helper import to_numpy
    import numpy as np    
    import numpy.linalg as la
    from scipy.linalg.lapack import dtrtri
    from scipy.stats import ortho_group, norm, uniform
    
    import pymanopt
    from pymanopt import Problem
    from pymanopt.optimizers import ConjugateGradient
    from pymanopt.manifolds.manifold import RiemannianSubmanifold
    
    from functools import partial


class Solver(AugmentedBCISolver):
    name = "tWDA"
    parameters = {
        "df": [10],
        "covariances_estimator" : ["scm"],
        "augmentation" : ["IdentityTransform"],
        **AugmentedBCISolver.parameters
    }

    install_cmd = "conda"
    requirements = ["pyriemann"]

    def set_objective(self, X, y, sfreq):
        """Set the objective information from Objective.get_objective.

        Objective
        ---------
        X: training data for the model
        y: training labels to train the model.
        sfreq: sampling frequency to allow filtering the data.
        """

        self.sfreq = sfreq
        self.X = X
        self.y = y
        self.n = to_numpy(X).shape[2]-1
        self.clf = make_pipeline(
            FunctionTransformer(to_numpy),
            Covariances(estimator=self.covariances_estimator),
            tWDA(n=self.n,df=self.df)
        )
        
        
        
class tWDA(BaseEstimator, ClassifierMixin, TransformerMixin):
    """Classification by t-Wishart.
    """
    
    def __init__(self,n,df,n_jobs=1):
        """Init."""
        self.n = n #nb of time samles
        self.df = df
        self.n_jobs = n_jobs
        
        
    def compute_class_center(self,S,df):
        _,p,_ = S.shape
        if df==np.inf:
            return np.mean(S,axis=0)/self.n
        return RCG(S,p,self.n,df=df)


    def fit(self, S, y):
        """Fit (estimates) the centroids.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        y : ndarray shape (n_trials, 1)
            labels corresponding to each trial.
        Returns
        -------
        self : wishart classifier instance
        """
        self.classes_ = np.unique(y)
        Nc = len(self.classes_)
        
        y = np.asarray(y)
        p,_ = S[0].shape
        if self.n_jobs==1:
            self.centers = [self.compute_class_center(S[y==self.classes_[i]],self.df) for i in range(Nc)]
        else:
            self.centers = Parallel(n_jobs=self.n_jobs)(delayed(self.compute_class_center)(S[y==self.classes_[i]],self.df) for i in range(Nc))
        
        self.pi = np.ones(Nc)
        
        for k in range(Nc):
            self.pi[k]= len(y[y==self.classes_[k]])/len(y)
        
        return self
 
    
    def _predict_distances(self, covtest):
        """Helper to predict the distance. equivalent to transform."""
        Nc = len(self.centers)
        K,p,_ =covtest.shape
        dist = np.zeros((K,Nc)) #shape= (n_trials,n_classes)
        
        
        for i in range(Nc):
            if self.df==np.inf:
                log_h = lambda t:-0.5*t
            else:
                log_h = lambda t:-0.5*(self.df+self.n*p)*np.log(1+t/self.df)
            center = self.centers[i].copy()
            inv_center = _matrix_operator(center,lambda x : 1/x)
            logdet_center = np.trace(logm(center))
            for j in range(K):
                #distance between the center of the class i and the cov_j
                dist[j,i] = np.log(self.pi[i])-0.5*self.n*logdet_center+log_h(np.matrix.trace(inv_center@covtest[j]))
                
        return dist

    def predict(self, covtest):
        """get the predictions.
        Parameters
        ----------
        covtest : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        pred : ndarray of int, shape (n_trials, 1)
            the prediction for each trials according to the closest centroid.
        """
        dist = self._predict_distances(covtest)
        preds = []
        n_trials,n_classes = dist.shape
        for i in range(n_trials):
            preds.append(self.classes_[dist[i,:].argmax()])
        preds = np.asarray(preds)
        return preds

    def transform(self, S):
        """get the distance to each centroid.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        dist : ndarray, shape (n_trials, n_classes)
            the distance to each centroid according to the metric.
        """
        return self._predict_distances(S)

    def fit_predict(self, S, y):
        """Fit and predict in one function."""
        self.fit(S, y)
        return self.predict(S)

    def predict_proba(self, S):
        """Predict proba using softmax.
        Parameters
        ----------
        S : ndarray, shape (n_trials, n_channels, n_channels)
            ndarray of SPD matrices.
        Returns
        -------
        prob : ndarray, shape (n_trials, n_classes)
            the softmax probabilities for each class.
        """
        return softmax(-self._predict_distances(S)**2)


def RCG(samples,p,n,df=5):
    alpha = n/2*(df+n*p)/(df+n*p+2)
    beta = n/2*(alpha-n/2)
    manifold = SPD(p,alpha,beta)
    return wishart_t_est(samples,n,df,manifold)
    

def sym(x):
	return 0.5 * (x + x.T)

	
class SPD(RiemannianSubmanifold):
    def __init__(self, p, alpha, beta):
        if (alpha <= 0 or alpha*p+beta <=0):
	        raise NameError('value of alpha and/or beta invalid, must have alpha>0 and alpha*p+beta>0')

        self._p = p
        self._alpha = alpha
        self._beta = beta
        name = f"Manifold of positive definite {p}x{p} matrices"
        dimension = int(p*(p+1)/2)
        super().__init__(name, dimension)

    @property
    def typical_dist(self):
        return np.sqrt(self.dim)

    def random_point(self, cond=100):
        U = ortho_group.rvs(self._p)
        #
        d = np.zeros(self._p)
        if self._p>2:
	        d[:self._p-2] = uniform.rvs(loc=1/np.sqrt(cond),scale=np.sqrt(cond)-1/np.sqrt(cond),size=self._p-2)
        d[self._p-2] = 1/np.sqrt(cond)
        d[self._p-1] = np.sqrt(cond)
        #
        return U @ np.diag(d) @ U.T

    def random_tangent_vector(self, point):
        return self.projection(point, norm.rvs(size=(self._p,self._p)))

    def zero_vector(self, point):
        return np.zeros((self._p,self._p))

    def inner_product(self, point, tangent_vector_a, tangent_vector_b):
        L = la.cholesky(point)
        iL, _ = dtrtri(L, lower=1)
        coor_a = iL @ tangent_vector_a @ iL.T
        if tangent_vector_a is tangent_vector_b:
            coor_b = coor_a
        else:
            coor_b = iL @ tangent_vector_b @ iL.T
        return self._alpha * np.tensordot(coor_a, coor_b, axes=point.ndim) + self._beta * np.trace(coor_a) * np.trace(coor_b)

    def norm(self, point, tangent_vector):
        return np.sqrt(self.inner_product(point, tangent_vector, tangent_vector))

    def projection(self, point, vector):
        return sym(np.real(vector))
	
    def euclidean_to_riemannian_gradient(self, point, euclidean_gradient):
        return (point @ sym(euclidean_gradient) @ point) / self._alpha - (self._beta / (self._alpha*(self._alpha + self._p * self._beta))) * np.trace(euclidean_gradient @ point) * point

    def retraction(self, point, tangent_vector):
        return np.real(sym( point + tangent_vector + 0.5 * tangent_vector @ la.solve(point, tangent_vector)))

    def transport(self, point_a, point_b, tangent_vector_a):
        tmp = sqrtm(la.solve(point_a, point_b).T) # (point_b point_a^{-1})^{1/2}
        return tmp @ tangent_vector_a @ tmp.T
	
    def dist(self, point_a, point_b):
        L = la.cholesky(point_a)
        iL,_ = dtrtri(L, lower=1)
        tmp = iL @ point_b @ iL.T
        log_eigs = np.log(la.eigh(tmp)[0]) # replace by some Cholesky ???
        return (self._alpha * np.sum(log_eigs**2) + self._beta * np.sum(log_eigs)**2)**0.5
    
def wishart_t_est(S,n,df,manifold):
    @pymanopt.function.numpy(manifold)
    def cost(R):
        return t_wish_cost(R,S,n,df)
    @pymanopt.function.numpy(manifold)
    def euclidean_gradient(R):
        return t_wish_egrad(R,S,n,df)
    #
    problem = Problem(manifold=manifold, cost=cost, euclidean_gradient=euclidean_gradient)
    init = np.eye(S.shape[-1])
    optimizer = ConjugateGradient(verbosity=0)
    return optimizer.run(problem, initial_point=init).point

def t_wish_cost(R,S,n,df):
    p, _ = R.shape
    return ellipt_wish_cost(R,S,n,partial(t_logh,df=df,dim=n*p))

def t_wish_egrad(R,S,n,df):
    p, _ = R.shape
    return ellipt_wish_egrad(R,S,n,partial(t_u,df=df,dim=n*p))


def ellipt_wish_cost(R,S,n,logh):
    k, p, _ = S.shape
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    return 1/2 * np.log(la.det(R)) - np.sum(logh(a))/n/k


def ellipt_wish_egrad(R,S,n,u):
    k, p, _ = S.shape
    # psi
    a = np.einsum('kii->k',la.solve(R,S)) # tr(inv(R)@S[k])
    psi = np.einsum('k,kij->ij',u(a),S)
    #
    return la.solve(R,la.solve(R.T,((R  - psi/n/k) /2).T).T)

def t_logh(t,df,dim):
    return -(df+dim)/2*np.log(1+t/df) 

def t_u(t,df,dim):
    return (df+dim)/(df+t)