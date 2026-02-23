import numpy as np
from scipy.special import loggamma, eval_legendre
from matplotlib import pyplot as plt

def compute_I(N):
    I = np.zeros(N+1)
    I[0] = .5*np.arctan(2)
    for k in range(1, N+1):
        first_term = .5*np.arctan(2) * (-.25)**k
        second_term = .25 * sum((-.25)**(k-j-1) / (2*j+1) for j in range(k))
        I[k] = first_term + second_term
    return I

def compute_a_mat(N):
    a = np.zeros((N+1, N+1))
    for n in range(N+1):
        for l in range(n+1):
            lg = loggamma(2*(n+l)+1) - n*np.log(4) - loggamma(2*l+1) - loggamma(n-l+1) - loggamma(n+l+1)
            a[n, l] = (-1)**(n+l) * np.exp(lg)
    return a

def compute_alpha(N):
    # We make the alpha here
    alpha = np.zeros(N+1)
    # Make the I and a so we can use them
    I = compute_I(N)
    a = compute_a_mat(N)
    # Now assemble it appropriately
    for n in range(N+1):
        alpha[n] = (4*n+1) * sum(a[n, l] * I[l] for l in range(n+1))
    return alpha

def plot_hardcode(n_list):
    
    plt.figure(dpi=300)
    
    # First, let's plot the function
    z_eval = np.linspace(-1, 1, num=1000)
    f_ex = 1/(1 + 4*z_eval**2)
    plt.plot(z_eval, f_ex, 'k-', label='Exact')
    
    # Now iterate over our other guys
    for N in n_list:
        f_ev = np.zeros(z_eval.shape)
        alpha = compute_alpha(N)
        for (n, alphan) in enumerate(alpha):
            f_ev += alphan * eval_legendre(2*n, z_eval) 
        plt.plot(z_eval, f_ev, '--', label=f'N={N}')
    
    # Now we add in the limit plots
    plt.plot((-1, -1), (-.5, 1.5), 'k-')
    plt.plot((1, 1), (-.5, 1.5), 'k-')
    
    # Now we make the plot
    plt.xlim((-1.2, 1.2))
    plt.ylim(-.5, 1.5)
    plt.legend()
    plt.title('Convergence For Legendre Expansion')
    plt.show()
    
ns_list = [1, 3, 5, 7]
plot_hardcode(ns_list)