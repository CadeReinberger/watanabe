import numpy as np
from scipy.special import binom as choose

def series_recip(a):
    '''
    Returns the series expansion of the reciprocal of the series a. Will return
    None to indicate error in the case that a[0] == 0, in which case the series
    expansion can no longer be done as a simple taylor series. 
    
    Derived by applying Cauchy's product rule to S*S^-1 = 1, noting a_tilde[0]
    is trivial, and then using the higher order expansion of S*S^-1 in order
    to derive a recursion for all larger a_tilde[n]. 
    '''
    N=len(a)-1
    a_tilde=np.zeros(N+1)
    if a[0]==0:
        #reciprocal has singularity at 0
        return None
    a_tilde[0]=1/a[0]
    for n in range(1,N+1):
        a_tilde[n]=-sum(a[k]*a_tilde[n-k] for k in range(1,n+1))/a[0]
    return a_tilde

def series_log(a):
    '''
    Returns the series expansion of the logarithm of the series a. Will return 
    None to indicate error in the case that a[0] == 0, in which case the series
    expansion can no longer be done as a simple taylor series. 
    
    Derived by differentiating S, using Cauchy's Product rule to produce an 
    expansion for S' with S^-1, derived using prior methods, and integrating 
    term-by-term to produce an explicit expression.     
    '''
    N=len(a)-1
    a_hat=np.zeros(N+1)
    if a[0]==0:
        #logarithm has singularity at 0
        return None
    a_hat[0]=np.log(a[0])
    a_tilde=series_recip(a)
    for n in range(0,N):
        a_hat[n+1]=sum((k+1)*a[k+1]*a_tilde[n-k] for k in range(0,n+1))/(n+1)
    return a_hat
    

def series_multiply(a,b):
    '''
    Produce the product of the taylor series a and b. Ignores terms of a or b
    that are of higher order than the other has. Uses Cauchy's Product Rule. 
    '''
    N=min(len(a),len(b))-1
    c=np.zeros(N+1)
    c[0]=a[0]*b[0]
    for n in range(1,N+1):
        c[n]=sum(a[k]*b[n-k] for k in range(0,n+1))
    return c

def integrate(a, constant=0):
    '''
    Integrate the series a term by term. Can specify the constant term, by 
    default it's 0
    '''
    #we won't add in  the extra term, but technically you could. 
    N=len(a)-1
    res=np.zeros(N+1)
    for k in range(1,N+1):
        res[k]=a[k-1]/k
    res[0]=constant
    return res

def series_add(a,b):
    '''
    Just adds the series a and b term-by-term. 
    '''
    #unlike some of the previous ones, this won't work for lists as input.
    if len(a)<len(b):
        return a+b[:len(a)]
    elif len(a)>len(b):
        return a[:len(b)]+b
    else:
        return a+b
    
def fill(*args):
    '''
    Takes some by hand terms (all of the arguments except the last), and fills
    them out with zeros to get a series that only has the terms given, and has
    leading order given by the last argument
    '''
    res=np.zeros(args[-1]+1)
    for (i,a) in enumerate(args[:min(len(args)-1,args[-1]+1)]):
        res[i]=a
    return res

def scale(series, k):
    '''
    Returns the taylor expansion for f(kx) where f(x) expands like k
    '''
    return np.multiply(series, np.power(k, range(len(series))))

def series_power(series, alpha):
    '''
    computes the taylor expansion of f(x)^alpha with JCP Miller's formula
    '''
    assert(series[0] != 0)
    N = len(series) - 1
    a = series/series[0]
    b = np.zeros(N+1)
    b[0] = 1
    for k in range(1, N+1):
        b[k] = sum((alpha*(k-p)-p)*b[p]*a[k-p] for p in range(k))/k
    result = b * (series[0] ** alpha)
    return result

def series_power_hardcode(series, alpha):
    '''
    computes the taylor expansion of f(x)^alpha with JCP Miller's formula
    '''
    assert(series[0] != 0)
    N = len(series) - 1
    a = series/series[0]
    
    #initialize output
    b = np.zeros(N+1)
    b[0] = 1
    
    #make aprime to binomial
    ap = a[:]
    ap[0] = 0
    
    # binomially expand in series land
    app = ap[:] # a prime power
    for p in range(1, N+1):
        b += choose(alpha, p) * app
        app = series_multiply(app, ap)
        
    # rescale the result and return it!
    result = b * (series[0] ** alpha)
    return result
    
def cos(N):
    '''
    Returns the Taylor series of cosine to order N with DP
    '''
    series = np.zeros(N+1)
    cur = 1
    for n in range(0, N//2 + 1):
        series[2*n] = cur
        cur /= -(2*n+1)*(2*n+2)
    return series

def sin(N):
    '''
    Returns the Taylor series of sine to order N with DP
    '''
    series = np.zeros(N+1)
    cur = 1
    for n in range(0, (N+1)//2):
        series[2*n+1] = cur
        cur /= -(2*n+2)*(2*n+3)
    return series

def exp(N):
    '''
    Returns the taylor series of exp to order N with DP
    '''
    series = np.zeros(N+1)
    cur = 1
    for n in range(N+1):
        series[n] = cur
        cur /= (n+1)
    return series

def arctanh(N):
    series = np.zeros(N+1)
    for n in range(N+1):
        if n % 2 == 1:
            series[n] = 1/n
    return series

def deriv(series):
    '''
    Returns the deriative of a series (as the same size)
    '''
    N = len(series)-1
    b = np.zeros(N+1)
    for n in range(N):
        b[n] = (n+1)*series[n+1]
    return b

def VDSolve(alpha, b):
    '''
    uses the bjork-pereyra algorithm for solving vandermonde systems. All the
    previous papers have used an explicit inversion formula, but I'm worried 
    about the ill-conditioning of the numerical methods, so it's this for now. 
    source:
        https://github.com/nschloe/vandermonde/blob/master/vandermonde/main.py
    '''
    if isinstance(alpha, int):
        alpha = np.array(range(1, alpha + 1))
    n = len(b)
    x = b.copy()
    for k in range(n):
        x[k + 1 : n] -= alpha[k] * x[k : n - 1]
    for k in range(n - 1, 0, -1):
        x[k:n] /= alpha[k:n] - alpha[: n - k]
        x[k - 1 : n - 1] -= x[k:n]
    return x

def binom_transform(series, shift):
    '''
    Returns the same polynomial series but as the new coefficients bn so that
        sum_{0 <= n <= N} a[n]*x^n == sum_{0 <= n <= N} b[n] * (x + shift)^n
    '''
    N = len(series)-1
    b = np.zeros(N+1)
    for j in range(N):
        b[j] = sum(choose(k, j) * series[k] * shift**(k-j) for k in range(j, N+1))
    return b
        