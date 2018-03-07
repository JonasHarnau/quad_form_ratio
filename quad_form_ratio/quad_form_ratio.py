import numpy as np
from scipy.optimize import fsolve
from scipy.optimize import brentq
from scipy import stats
import pandas as pd

def simulate_R(A, B, seed=None, reps=10**5):
    """
    Generates draws for ratios of quadratic forms.
    
    This function draws realizations from the distribution of 
    R = (e'Ae)/(e'Be) where e is N(0,I).
        
    Parameters
    ----------
    
    A : numpy.array
        n by n matrix
    
    B : numpy.array
        n by n matrix
        
    seed : int, optional
           Random seed. (Default is 'None')
    
    reps : int, optional
           Number of draws from R. (Default is 10**5)
          
    Returns
    -------
    
    R : np.array
        Vector of length 'reps' containing the draws.
        
    Examples
    --------
    
    This example corresponds to draws from an F(1,1) distribution.
    
    >>> import numpy as np
    >>> A = np.array([[1, 0], [0, 0]])
    >>> B = np.array([[0, 0], [0, 1]])
    >>> R = simulate_R(A, B)
    
    """
    
    if A.shape[0] != A.shape[1] != B.shape[0] != B.shape[1]:
        raise ValueError('A and B have to be square and of the ' +
                         'same dimension.')
    
    np.random.seed(seed)
    U = np.random.normal(size=(reps, A.shape[0]))
    
    numerator = np.einsum('ij,ij->i', U.dot(A), U)
    denominator = np.einsum('ij,ij->i', U.dot(B), U)
    R = numerator/denominator
    
    return R

def saddlepoint_cdf_R(A, B, quantiles):
    """
    Saddlepoint approximation to the cdf of ratios of quadratic forms.
    
    This implements a saddlepoint approximation to P(R <= q) for 
    R = (e'Ae)/(e'Be) where e is N(0,I). The implementation follows 
    Liebermann (1994), Butler and Paolella (2008).
    
    Parameters
    ----------
    
    A : numpy.array
        n by n matrix of rank at least one.
    
    B : numpy.array
        n by n positive semidefinite matrix
    
    quantiles : numpy.array
                Vector of quantiles 'q'. Must be iterable.
        
    Returns
    -------
        
    approx_prob : pandas.Series
                  Series with index corresponding to quantiles and
                  values corresponding to the saddlepoint approximated
                  probabilities.
                  
    Examples
    --------
    
    This example corresponds to an F(2,2) distribution. The approximated 
    probabilities are within 10e-3 of the exact ones.
    
    >>> import numpy as np
    >>> from scipy import stats
    >>> A = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
    >>> B = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    >>> p_saddlepoint = saddlepoint_cdf_R(A, B, [0.01, 1, 10, 100])
    >>> p = stats.f.cdf([0.01, 1, 10, 100], dfn=2, dfd=2)
    >>> print(np.allclose(p_saddlepoint, p, atol=10e-3))
    
    References
    ----------
    
    Lieberman, O. (1994). Saddlepoint approximation for the distribution of
    a ratio of quadratic forms in normal variables. Journal of the American
    Statistical Association, 89(427), 924–928. 
    
    Butler, R. W., & Paolella, M. S. (2008). Uniform saddlepoint 
    approximations for ratios of quadratic forms. Bernoulli, 14(1), 140–154.
    
    """
    
    np.seterr('raise')
    
    def K(s, l):
        return -np.sum(np.log(1- 2 * s * l))/2
    def K_prime(s, l):
        return np.sum(l/(1- 2 * s * l))
    def K_double_prime(s, l):
        # return as array for fsvolve
        return np.array([np.sum(2 * (l/(1- 2 * s * l))**2)]) 
    def K_triple_prime(s, l):
        return np.sum(8 * (l/(1- 2 * s * l))**3)
    
    def find_saddlepoint(l):
        try:
            s_high = (1/(2*l[l > 0])).min() 
        except ValueError: 
            # have to consider case where l[l > 0] yields an empty set
            # if l >= 0 with some l > 0 then K' is 0 for s_hat = inf
            # Then K is infinite. So w_hat is inf. Also K'' is 0. p = 1
            s_hat = np.inf
            w_hat = np.inf
            return s_hat, w_hat
        try:
            s_low = (1/(2*l[l < 0])).max()
        except ValueError:
            # if l <= 0 with some l < 0 then s_hat = -inf. 
            # Oppositive from above. p = 0.
            s_hat = -np.inf
            w_hat = -np.inf
            return s_hat, w_hat
        try: # start searching with the mean
            s_hat = fsolve(K_prime, (s_low + s_high)/2, args=l, 
                           fprime=K_double_prime, maxfev=800)
            w_hat = np.sign(s_hat) * np.sqrt(-2*K(s_hat, l))
            return s_hat, w_hat
        except FloatingPointError: 
            # if the mean does not work as initial value use a grid search
            step = (s_high - s_low)/100
            for s in np.arange(s_low + step, s_high, step):
                try:
                    s_hat = fsolve(K_prime, s, args=l, fprime=K_double_prime, 
                                   maxfev=800)
                    w_hat = np.sign(s_hat) * np.sqrt(-2*K(s_hat, l))
                    return s_hat, w_hat
                except FloatingPointError:
                    pass
            # only ever get here if no solution for any of these
            raise ValueError('Could not find a solution for s_hat')
            
    probs = []
    
    for r in quantiles:
        l = np.linalg.eigvalsh(A - r*B)
        l = l[~np.isclose(l,0)] # helps numerical stability
        if ~np.isclose(np.sum(l), 0, atol=10e-5): 
            # scenario where E(X_r) != 0
            s_hat, w_hat = find_saddlepoint(l)
            try: 
                # catch cases with s_hat infinite. Yields error because 0 * inf
                u_hat = s_hat * np.sqrt(K_double_prime(s_hat, l))
                p = (stats.norm.cdf(w_hat) + stats.norm.pdf(w_hat) 
                     * (1/w_hat - 1/u_hat))[0]
            except FloatingPointError:
                if s_hat == np.inf:
                    p = 1
                elif s_hat == -np.inf:
                    p = 0
                else: 
                    raise ValueError('Numerical issue despite finite s_hat')
        else: # case when E(X_r) = 0
            p = (1/2 + K_triple_prime(0, l)/(6 * np.sqrt(2 * np.pi) 
                                             * K_double_prime(0, l)**(3/2)))[0]
        probs.append(p)
        
    approx_prob = pd.Series(probs, index=quantiles)
    
    return approx_prob
    
def find_support_Butler_Paollela(A, B):
    """
    Determines (parts of) the support of the ratio of quadratic forms.
    
    This function finds the support of R = (e'Ae)/(e'Be) where e is N(0,I)
    following Lemma 3 of Butler and Paolella (2008). Except in a special 
    case (Case 1. in the Lemma), only the upper bound of the support is
    returned. As Butler and Paolella (2008) point out, in such cases the
    lower bound may be found by considering the upper bound of -R, thus 
    rerunning the function with -A, B.
    
    Parameters
    ----------
    
    A : numpy.array
        n by n matrix of rank at least one.
    
    B : numpy.array
        n by n positive semidefinite matrix
        
    Returns
    -------
        
    case : str
           The case from Lemma 3 of Butler and Paolella (2008) given by
           'A' and 'B'
    
    lower_bound : float 
                  Lower bound of the support. For cases of Lemma 3 of 
                  Butler and Paolella (2008) where this is not specified
                  this is 'numpy.nan'.
                  
    upper_bound : float
                  Upper bound of the support.
                  
    Examples
    --------
    
    The examples correspond to those given in Butler and Paolella (2008).
    
    This example corresponds to an F(1,1) distribution. 
    
    >>> import numpy as np
    >>> from scipy import stats
    >>> A = np.array([[1, 0], [0, 0]])
    >>> B = np.array([[0, 0], [0, 1]])
    >>> case, l, r = find_support_Butler_Paollela(A, B)
    >>> print('{}, lower bound: {}, upper bound: {}'.format(case, l, r))
    
    This example corresponds to an Cauchy distribution. 
    
    >>> import numpy as np
    >>> from scipy import stats
    >>> A = np.array([[0, 1/2], [1/2, 0]])
    >>> B = np.array([[1, 0], [0, 0]])
    >>> case, l, r = find_support_Butler_Paollela(A, B)
    >>> print('{}, lower bound: {}, upper bound: {}'.format(case, l, r))
    
    References
    ----------
    
    Butler, R. W., & Paolella, M. S. (2008). Uniform saddlepoint 
    approximations for ratios of quadratic forms. Bernoulli, 14(1), 140–154.
    
    """
    
    if A.shape[0] != A.shape[1] != B.shape[0] != B.shape[1]:
        raise ValueError('A and B have to be square and of same dimension.')
        
    l_B, O_B_T = np.linalg.eigh(B)
    l_B_0 = np.isclose(l_B,0)
    l_B = l_B[~l_B_0]
    
    l_A = np.linalg.eigvalsh(A)
    l_A_0 = np.isclose(l_A,0)
    l_A = l_A[~l_A_0]
    
    if (l_B < 0).any():
        raise ValueError('B is not positive semi-definite')
    
    ## Case 1
    # B > 0 (positive definite)
    if not l_B_0.any():
        # A has rank at least one.
        if not l_A_0.all():
            B_inv_A = np.linalg.inv(B).dot(A)
            l_B_inv_A = np.linalg.eigvalsh(B_inv_A)
            lower_bound = np.min(l_B_inv_A)
            upper_bound = np.max(l_B_inv_A)
            case = 'Case1'
    ## Case 2
    # B has at least one zero eigenvalue
    else:
        # Define C matrix
        C = O_B_T.T.dot(A).dot(O_B_T)
        C11 = C[~l_B_0,:][:,~l_B_0]
        C12 = C[~l_B_0,:][:,l_B_0]
        C21 = C[l_B_0,:][:,~l_B_0]
        C22 = C[l_B_0,:][:,l_B_0]
        l_C22 = np.linalg.eigvalsh(C22)
        l_C22_0 = np.isclose(l_C22, 0)
        ## Case 2(a)
        # C22 has a positive Eigenvalue
        if (l_C22[~l_C22_0] > 0).any():
            lower_bound = np.nan
            upper_bound = np.inf
            case = 'Case2(a)'
        ## Case 2(b)
        # C22 is negative definite
        elif not l_C22_0.any():
            Schur_complement = C11 - C12.dot(np.linalg.inv(C22)).dot(C21)
            L_B_inv_Schur = np.diag(1/l_B).dot(Schur_complement)
            l_L_B_inv_Schur = np.linalg.eigvalsh(L_B_inv_Schur)
            lower_bound = np.nan
            upper_bound = np.max(l_L_B_inv_Schur)
            case = 'Case2(b)'
        ## Case 2(c)
        # C22 is negative semidefinite
        else:
            # find nullspace of C22.
            if C22.shape == (1, 1):
                null_C22 = 0 if (C22 == 0) else 1
            else:
                Q_C22_T, _ = np.linalg.qr(C22.T, mode='complete')
                null_C22 = Q_C22_T[:,C22.shape[0]:]
            # find nullspace of C12
            if C12.shape == (1, 1):
                null_C12 = 0 if (C12 == 0) else 1
            else:
                Q_C12_T, _ = np.linalg.qr(C12.T, mode='complete')
                null_C12 = Q_C12_T[:,C12.shape[0]:]
            if not np.allclose(null_C22, null_C12):
                lower_bound = np.nan
                upper_bound = np.inf
                case = 'Case2(c1)'
            else:
                l_C22, O_C22_T = np.linalg.eigh(C22)
                l_C22_0 = np.isclose(l_C22,0)
                O_C22_C1 = O_C22_T.T[:,~l_C22_0]
                Schur_like = C11 - C12.dot(O_C22_C1).dot(
                    np.diag(1/l_C22[~l_C22_0])).dot(O_C22_C1.T.dot(C21))
                L_B_inv_Schur_like = np.diag(1/l_B).dot(Schur_like)
                l_L_B_inv_Schur_like = np.linalg.eigvalsh(L_B_inv_Schur_like)
                lower_bound = np.nan
                upper_bound = np.max(l_L_B_inv_Schur_like)
                case = 'Case2(c2)'
    return case, lower_bound, upper_bound

def find_support(A, B):
    """
    Finds the support of R.
    
    This function determines the support of R both both lower and upper tail.
    It does so by first calling 'find_support_Butler_Paollela(A, B)' and, if 
    the lower tail is not given, rerunning it with parameters '(-A, B)'. Then,
    the upper bound of the support of -R multiplied by negative one is the lower
    bound of R.
    
    Parameters
    ----------
    
    A : numpy.array
        n by n matrix of rank at least one.
    
    B : numpy.array
        n by n positive semidefinite matrix
        
    Returns
    -------
        
    lower_bound : float 
                  Lower bound of the support.
                  
    upper_bound : float
                  Upper bound of the support.
                  
    Examples
    --------
    
    The examples correspond to those given in Butler and Paolella (2008).
    
    This example corresponds to an F(1,1) distribution. 
    
    >>> import numpy as np
    >>> A = np.array([[1, 0], [0, 0]])
    >>> B = np.array([[0, 0], [0, 1]])
    >>> l, r = find_support(A, B)
    >>> print('lower bound: {}, upper bound: {}'.format(l, r))
    
    This example corresponds to an Cauchy distribution. 
    
    >>> import numpy as np
    >>> A = np.array([[0, 1/2], [1/2, 0]])
    >>> B = np.array([[1, 0], [0, 0]])
    >>> l, r = find_support(A, B)
    >>> print('lower bound: {}, upper bound: {}'.format(l, r))
    
    References
    ----------
    
    Butler, R. W., & Paolella, M. S. (2008). Uniform saddlepoint 
    approximations for ratios of quadratic forms. Bernoulli, 14(1), 140–154.

    """
    
    _, lower_bound, upper_bound = find_support_Butler_Paollela(A, B)
    if np.isnan(lower_bound):
        _, _, lower_bound = find_support_Butler_Paollela(-A, B)
        lower_bound = -lower_bound if (lower_bound != 0) else 0

    return (lower_bound, upper_bound)

def saddlepoint_inv_cdf_R(A, B, probabilities, precision=10e-5):
    """
    Saddlepoint approximation to inverse cdf of ratios of quadratic forms.
    
    This implements a saddlepoint approximation to p = P^(-1)(R <= q) for 
    R = (e'Ae)/(e'Be) where e is N(0,I). That is, the function provides 
    quantiles for given probabilities. This is solved iteratively through
    repeated calls of the saddlepoint approximation to the cdf as 
    implemented in 'saddlepoint_cdf_R' in this package.
    
    Parameters
    ----------
    
    A : numpy.array
        n by n matrix of rank at least one.
    
    B : numpy.array
        n by n positive semidefinite matrix
    
    probabilities : numpy.array
                    Vector of probabilities 'p'. Must be iterable.
    
    precision : float, optional
                The precision with which we want to solve the equation
                p = P^(-1)(R <= q) for q. (Default is 10e-5)
        
    Returns
    -------
        
    approx_quants : pandas.Series
                    Series with index corresponding to probabilites and
                    values corresponding to the saddlepoint approximated
                    quantiles.
    
    Notes
    -----
    
    Infinite lower or upper bounds of the support, which is determined
    through a call to 'find_support', are initially capped at 10e20.
    If there are still run-time errors, this range is iteratively 
    decreased towards zero in steps of 10e5.
    
    Examples
    --------
    
    This example corresponds to a Cauchy distribution. It confirms that
    we reach the desired approximation quality relative to 'saddlepoint_cdf_R',
    
    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy import stats
    >>> A = np.array([[0, 1/2], [1/2, 0]])
    >>> B = np.array([[1, 0], [0, 0]])
    >>> probabilities = [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.8]
    >>> q_saddlepoint = saddlepoint_inv_cdf_R(A, B, probabilities)
    >>> q = stats.cauchy.ppf(probabilities)
    >>> print(pd.concat(
    >>>     [q_saddlepoint.rename('SP'),
    >>>      pd.Series(q, index=q_saddlepoint.index, name='True')], axis=1))
    >>> p_saddlepoint = saddlepoint_cdf_R(A, B, q_saddlepoint)
    >>> max_diff = np.max(p_saddlepoint - probabilities)
    >>> print('Maximum difference to quantiles: {}'.format(max_diff))

    """
    
    lower_bound, upper_bound = find_support(A, B)
    
    scale_lower = False
    if np.isinf(lower_bound):
        lower_bound = -10e20
        scale_lower = True
    scale_upper = False
    if np.isinf(upper_bound):
        upper_bound = 10e20
        scale_upper = True
    
    def get_quantiles(probabilities, lower_bound, upper_bound):
        def f(q, p):
            return saddlepoint_cdf_R(A, B, [q]) - p
        
        quants = []
        try: 
            for p in probabilities:
                quants.append(brentq(f, lower_bound, upper_bound, 
                                     args=p, xtol=precision))
            return quants
        except RuntimeError: # if convergence fails we shrink search range
            if scale_lower:
                lower_bound -= 10e5
            if scale_upper:
                upper_bound -= 10e5
            return get_quantiles(probabilities, lower_bound, upper_bound)
    
    approx_quants = get_quantiles(probabilities, lower_bound, upper_bound)
    
    approx_quants = pd.Series(approx_quants, index=probabilities)
    
    return approx_quants