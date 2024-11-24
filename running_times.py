""" Compare computation times and erros of direct and spinal methods for 
Monte-Carlo estimation of the variance of the total resources in the colonies.

Usage:
======
    Simply run the script and wait for the plot. You can change the parameter
    values in the first section but please refrain from changing the number of
    Monte Carlo iterations M.
"""
import functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import timeit

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Parameter values
division_rate = 1
death_rate = 0.3
init_pop_size = 10
t_max = 2
t_pts = 20
growth_rate = 1
init_pop_mass = 10
sigma = 1
delta = 0.5     

# Initialization of arrays
M_cvg = 500000  # Enough for convergence 
times = np.linspace(0, t_max, t_pts)
mean_r_t = init_pop_size * init_pop_mass \
            * np.exp((growth_rate - death_rate) * times)
true_value = np.zeros(t_pts)
estimator_r = np.zeros(t_pts)
n_pts = 10
M = np.logspace(1.5, 4, n_pts) 
times_direct = np.zeros(n_pts)
times_spinal = np.zeros(n_pts)
sse_direct = np.zeros(n_pts)
sse_spinal = np.zeros(n_pts)
    
# Asymptotic value of Var(R_t) when M is large
for i in tqdm(range(0, M_cvg), mininterval=1, ncols=100,
              desc="x_bar mean estimation. Progress: "):
    t, n_t, r_t, y_t = functions.trajectory_spinal_opti(division_rate, 
                                     death_rate, growth_rate, sigma, delta, 
                                     init_pop_size, init_pop_mass, t_max,t_pts)
    for j in range(t_pts):
        estimator_r[j] += r_t[j] / M_cvg
true_value = (estimator_r - mean_r_t)* mean_r_t

# Computation times and errors
for k in range(n_pts):
    if k < n_pts - 2:
        N = 40
        err_spine = np.zeros(N)
        err_direct = np.zeros(N)
    elif k < n_pts - 1:
        N = 30
        err_spine = np.zeros(N)
        err_direct = np.zeros(N)
    else: 
        N = 20
        err_spine = np.zeros(N)
        err_direct = np.zeros(N)
    m = int(M[k])
    # Spinal method
    err = 0
    estimator_r = np.zeros(t_pts)
    estimator_var_r = np.zeros(t_pts)
    start = timeit.default_timer()
    for i in tqdm(range(0, m), mininterval=1, ncols=100,
         desc='compute %1.d'%(2 * k + 1)+'/%1.d'%(2 * n_pts) + '. Progress: '):
        t, n_t, r_t, y_t = functions.trajectory_spinal_opti(division_rate, 
                                    death_rate, growth_rate, sigma, delta, 
                                    init_pop_size, init_pop_mass, t_max, t_pts)
        for j in range(t_pts):
            estimator_r[j] += r_t[j] / m   
    estimator_var_r = (estimator_r - mean_r_t)* mean_r_t
    times_spinal[k] = timeit.default_timer() - start    
    err_spine[0] = np.mean(abs(estimator_var_r - true_value)[1:]\
                            / true_value[1:])
    
    for l in tqdm(range(0,N-1), mininterval=1, ncols=100,
                  desc='Spine error estimation. Progress: '):
        estimator_r = np.zeros(t_pts)
        estimator_var_r = np.zeros(t_pts)
        for i in range(m):
            t, n_t, r_t, y_t = functions.trajectory_spinal_opti(division_rate, 
                                    death_rate, growth_rate, sigma, delta, 
                                    init_pop_size, init_pop_mass, t_max, t_pts)
            for j in range(t_pts):
                estimator_r[j] += r_t[j] / m   
        estimator_var_r = (estimator_r - mean_r_t) * mean_r_t
        err_spine[l+1] =np.mean(abs(estimator_var_r - true_value)[1:]\
                                / true_value[1:])
        #err += np.mean((estimator_var_r - true_value)[-1] ** 2)
    sse_spinal[k] = np.mean(err_spine)
    
    # direct method
    err = 0
    estimator_r = np.zeros(t_pts)
    estimator_var_r = np.zeros(t_pts)
    start = timeit.default_timer()
    for i in tqdm(range(0, m), mininterval=1, ncols=100,
         desc='compute %1.d'%(2 * k + 2)+'/%1.d'%(2 * n_pts) + '. Progress: '):
        t, n_t, r_t = functions.trajectory_opti(division_rate, death_rate, 
                                     growth_rate, sigma, delta, 
                                     init_pop_size, init_pop_mass, t_max,t_pts)
        for j in range(t_pts):
            estimator_r[j] += r_t[j] ** 2 / m
    estimator_var_r =  estimator_r - mean_r_t ** 2 
    times_direct[k] = timeit.default_timer() - start     
    err_direct[0] = np.mean(abs(estimator_var_r - true_value)[1:]\
                            / true_value[1:])
    for l in tqdm(range(0,N-1), mininterval=1, ncols=100,
                  desc='Direct error estimation. Progress: '):
        estimator_r = np.zeros(t_pts)
        start = timeit.default_timer()
        for i in range(m):
            t, n_t, r_t = functions.trajectory_opti(division_rate, death_rate, 
                                     growth_rate, sigma, delta, 
                                     init_pop_size, init_pop_mass, t_max,t_pts)
            for j in range(t_pts):
                estimator_r[j] += r_t[j] ** 2 / m
        estimator_var_r =  estimator_r - mean_r_t ** 2 
        err_direct[l+1] = np.mean(abs(estimator_var_r - true_value)[1:]\
                                / true_value[1:])
    sse_direct[k] = np.mean(err_direct) 

# Plot 
size = 30
fig, ax1 = plt.subplots(figsize=(8, 8))
ax2 = ax1.twinx()    
ax1.plot(M, 100 * sse_direct, label = r'Error direct method', color = 'red', 
         linestyle='--', marker='o')
ax1.plot(M, 100 * sse_spinal, label = r'Error spinal method', color = 'orange', 
         linestyle='--', marker='o')
ax1.plot(M, 100 / np.sqrt(M), label = r'Slope $\frac{1}{2}$', color = 'black')
ax2.plot(M, times_direct, label = r'Simulation time direct method', 
         linestyle='--', marker='o', color = 'blue')
ax2.plot(M, times_spinal, label = r'Simulation time spinal method', 
         linestyle='--', marker='o', color = 'cyan')
ax1.legend(loc = 'upper left', fontsize = size)  
ax2.legend(loc = 'upper right', fontsize = size) 
ax1.set_xlabel(r'Monte Carlo size M', fontsize = size)
ax2.set_ylabel(r'Simulation time for M trajectories (s)' , 
               fontsize = size, labelpad=20)
ax1.set_yscale('log')
ax2.set_yscale('log')
ax1.tick_params(axis='both', which='major', labelsize=size)
ax1.tick_params(axis='both', which='minor', labelsize=size)
ax2.tick_params(axis='both', which='major', labelsize=size)
ax2.tick_params(axis='both', which='minor', labelsize=size)
ax1.set_ylabel(r'Relative error ($\%$)', fontsize = size)
ax1.set_xscale('log')
ax1.grid(True, which="both", ls="-", alpha=0.5)