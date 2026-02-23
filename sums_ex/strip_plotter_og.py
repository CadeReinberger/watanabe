import taylor_lib
import numpy as np
from matplotlib import pyplot as plt

def compute_g_coefficients(N):
    at = taylor_lib.arctanh(N)
    denom = taylor_lib.fill(1, N) + (16/np.pi**2) * taylor_lib.series_multiply(at, at)
    g_series = taylor_lib.series_recip(denom)
    return g_series

def plot_hardcode(n_list):
    
    plt.figure(dpi=300)
    
    # First, let's plot the function
    z_eval = np.linspace(0, 10, num=1000)
    f_ex = 1/(1 + 4*(2*np.exp(-z_eval)-1)**2)
    plt.plot(z_eval, f_ex, 'k-', label='Exact')
    
    # Make the u to use it a little bit
    u_ev = np.tanh(.5*np.pi*(2*np.exp(-z_eval)-1))
    
    # Now iterate over our other guys
    for N in n_list:
        g = compute_g_coefficients(N)
        f_ev = np.ones(z_eval.shape)
        for (n, gn) in enumerate(g):
            if n == 0:
                continue
            f_ev += gn * u_ev**n 
        plt.plot(z_eval, f_ev, '--', label=f'N={N}')
    
    # Now we make the plot
    plt.xlim((0, 10))
    plt.ylim(.2, 1)
    plt.legend()
    plt.title('Convergence For Strip Resummation')
    plt.show()
    
ns = (1, 5, 20, 50, 100, 200)
plot_hardcode(ns)