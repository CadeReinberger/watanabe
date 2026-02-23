import numpy as np
from matplotlib import pyplot as plt
from scipy.special import loggamma, eval_legendre
from tqdm import tqdm

def compute_g(N):
    # First, let's make the h in closed form
    h = np.zeros(N+1)
    h[0] = 1
    for n in range(1, N+1):
        h[n] = (16/np.pi**2) * sum(1/((2*k+1)*(2*n-2*k-1)) for k in range(n))
    
    # Now, we make the g
    g = np.zeros(N+1)
    g[0] = 1/h[0]
    for n in range(1, N+1):
        g[n] = -(1/h[0]) * sum(h[k]*g[n-k] for k in range(1, n+1))
   
    # Return the result
    return g

def compute_alpha(N):
    # Compute the I first
    I = np.zeros(N+1)
    I[0] = .5*np.arctan(2)
    for k in range(1, N+1):
        first_term = .5*np.arctan(2) * (-.25)**k
        second_term = .25 * sum((-.25)**(k-j-1) / (2*j+1) for j in range(k))
        I[k] = first_term + second_term
    
    # Next, compute the a
    a = np.zeros((N+1, N+1))
    for n in range(N+1):
        for l in range(n+1):
            lg = loggamma(2*(n+l)+1) - n*np.log(4) - loggamma(2*l+1) - loggamma(n-l+1) - loggamma(n+l+1)
            a[n, l] = (-1)**(n+l) * np.exp(lg)
            
    # Finally, compute the alpha
    alpha = np.zeros(N+1)
    for n in range(N+1):
        alpha[n] = (4*n+1) * sum(a[n, l] * I[l] for l in range(n+1))
        
    # Now, we return the alpha
    return alpha

def compute_lambda(N):
    _lambda = np.zeros(N+1)
    _lambda[0] = 1
    for n in range(1, N+1):
        _lambda[n] = -_lambda[n-1]/n
    return _lambda 

def make_errors_one(N):
    # Make the evaluation points and exact solution
    z_ev = np.linspace(0, 20, num=1000)
    f_ex = 1/(16*np.exp(-2*z_ev) - 16*np.exp(-z_ev) + 5)
    
    # Make the g we're gonna use
    u = np.tanh(.5*np.pi * (2*np.exp(-z_ev) - 1))
    g = compute_g(N)
    
    # Now we prepare to iterate
    f_cur = np.zeros(len(z_ev))
    ns = list(range(N+1))
    errs = np.zeros(N+1)
    for (n, gn) in enumerate(g):
        f_cur += gn * u**(2*n) 
        errs[n] = np.nanmax(np.abs(f_ex - f_cur))
    
    return ns, errs

def make_errors_two(N):
    # Make the evaluation points and exact solution
    z_ev = np.linspace(0, 20, num=1000)
    f_ex = 1/(16*np.exp(-2*z_ev) - 16*np.exp(-z_ev) + 5)
    
    # Make the g we're gonna use
    alpha = compute_alpha(N)
    u = 2*np.exp(-z_ev) - 1
    
    # Now we prepare to iterate
    f_cur = np.zeros(len(z_ev))
    ns = list(range(N+1))
    errs = np.zeros(N+1)
    for (n, alphan) in enumerate(alpha):
        f_cur += alphan * eval_legendre(2*n, u) 
        errs[n] = np.nanmax(np.abs(f_ex - f_cur))
    
    return ns, errs

def make_errors_three(N):
    # Make the evaluation points and exact solution
    z_ev = np.linspace(0, 20, num=1000)
    f_ex = 1/(16*np.exp(-2*z_ev) - 16*np.exp(-z_ev) + 5)
    
    # Make the g we're gonna use
    u = np.log(16*np.exp(-2*z_ev) - 16*np.exp(-z_ev) + 5)
    _lambda = compute_lambda(N)
    
    # Now we prepare to iterate
    f_cur = np.zeros(len(z_ev))
    ns = list(range(N+1))
    errs = np.zeros(N+1)
    for (n, lambdan) in enumerate(_lambda):
        f_cur += lambdan * u**(n) 
        errs[n] = np.nanmax(np.abs(f_ex - f_cur))
    
    return ns, errs

def make_combined_error_plot():
    # Make the errors from all three expansions
    one_ns, one_errs = make_errors_one(15)
    two_ns, two_errs = make_errors_two(15)
    three_ns, three_errs = make_errors_three(15)
    
    # Plot them!
    plt.figure(dpi=300)
    plt.semilogy(one_ns, one_errs, 'b-', label='Strip Transform')
    plt.semilogy(two_ns, two_errs, 'g-', label='Legendre Expansion')
    plt.semilogy(three_ns, three_errs, 'r-', label='Log Expansion')
    
    # Label the plot
    plt.xlabel('Expansion N')
    plt.ylabel('L_infty Error')
    plt.title('Max Error for All Expansions')
    plt.legend()
    
    # Show the plot
    plt.show()
    
make_combined_error_plot()