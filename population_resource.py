""" Plot the mean population resources of the colonial process, estimated from 
Monte-Carlo method with the spinal algorithm.

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

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'
  
# Parameters
list_division_rate = [1, 10]
list_death_rate = [0.1, 2, 4]
init_pop_size = 3
t_max = 5
growth_rate = 5
init_pop_mass = 10
list_sigma = [0, 1, 2]
delta = 1

# Initialization   
M = 200000  # Large enough (<5% relative error)
t_pts = 20
times = np.linspace(0, t_max, t_pts)
fig,axes = plt.subplots(3, 3)
c = 0
K_theta = 0
for i in range(10000):
    thet = functions.theta_func()
    K_theta += thet * (1 - thet) / 10000
    
# Main computation loop
for div in range(2):
    division_rate = list_division_rate[div]
    for k in range(3):
        death_rate = list_death_rate[k]
        for l in range(3):
            c += 1
            sigma = list_sigma[l]
            estimator_r = np.zeros(t_pts)
            estimator_var_r = np.zeros(t_pts)
            # Boundaries computation 
            mean_r_t = init_pop_size * init_pop_mass \
                * np.exp((growth_rate - death_rate) * times)
            rho = sigma / np.sqrt( 1 + delta ** 2)
            A = growth_rate - death_rate + (rho * delta) ** 2
            B = init_pop_mass * (rho ** 2 + death_rate) 
            C_sup = growth_rate + sigma ** 2
            C_inf = growth_rate + sigma ** 2 - 2 * division_rate * K_theta
            X_0_inf = init_pop_size * init_pop_mass - B / (C_inf - A) 
            X_0_sup = init_pop_size * init_pop_mass - B / (C_sup - A)
    
            if C_inf != A:
                lower_bound_1 = B * np.exp(C_inf * times) / (C_inf - A) \
                        + X_0_inf * np.exp(A * times)
            else:
                lower_bound_1 = B * times * np.exp(C_inf * times) / C_inf \
                    + B / (C_inf * A) * (1 - np.exp(A * times)) \
                    + init_pop_size * init_pop_mass * np.exp(A * times)
            if C_sup != A:
                upper_bound_1 = B * np.exp(C_sup * times) / (C_sup - A) \
                    + X_0_sup * np.exp(A * times)
            else:
                upper_bound_1 = B * times * np.exp(C_sup * times) / C_sup \
                    + B / (C_sup * A) * (1 - np.exp(A * times)) \
                    + init_pop_size * init_pop_mass * np.exp(A * times)
            up_bnd = mean_r_t * (upper_bound_1 - mean_r_t)
            low_bnd = mean_r_t * (lower_bound_1 - mean_r_t)
            
            # Variance Monte-Carlo estimation 
            traj_r = []
            for i in tqdm(range(0, M), mininterval=1, ncols=100,
               desc="Trajectories computations. Step %1.d/18. Progress: "%(c)):
                t, n_t, r_t, y_t = functions.trajectory_spinal_opti(
                          division_rate, death_rate, growth_rate, sigma, delta, 
                                    init_pop_size, init_pop_mass, t_max, t_pts)
                for j in range(t_pts):
                    estimator_r[j] += r_t[j] / M
            estimator_var_r = (estimator_r - mean_r_t)* mean_r_t
            if div == 0:
                axes[k][l].plot(times[1:], up_bnd[1:], label = 'upper bound', 
                                                               color = 'black') 
            axes[k][l].plot(times[1:], estimator_var_r[1:], 
                   label = r'$ \widehat{Var}$('  + 
                                       r'$\lambda$=' + '%1.d)'%(division_rate),
                                                   linestyle='--', marker='o',)
            axes[k][l].plot(times[1:], low_bnd[1:] , label = 'low bnd(' + 
                                       r'$\lambda$=' + '%1.d)'%(division_rate))
            axes[k][l].set_title(r'$\mu$ = ' + '%.1f '%(death_rate) \
                                           + r' $\sigma$ = ' + '%.1d '%(sigma), 
                                                                 fontsize = 30)


# Plot        
for i in range(3):
    for j in range(3):
        axes[i][j].tick_params(axis='both', labelsize = 15)  
        axes[i][j].set_yscale('log')
        axes[i][j].tick_params(axis='both', labelsize = 30)
        axes[2][j].set_xlabel(r'Time (s)', fontsize = 30)
        if i < 2:
            axes[i][j].set_xticklabels([])
        axes[i][j].grid(True, which="both", ls="-", alpha=0.5)
handles, labels = axes[0][0].get_legend_handles_labels()
axes[0][1].legend(handles, labels, loc='upper center', ncol = 5,
             bbox_to_anchor=(0.5, 1.5),fancybox=False, shadow=False,
             fontsize = 30, handlelength = 1)         
plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.08, top = 0.89, 
                    wspace = 0.2, hspace = 0.25 ) 
# axes[0][1].legend(handles, labels, loc='upper center', ncol = 1,
#              bbox_to_anchor=(2.8, 0.5),fancybox=False, shadow=False,
#              fontsize = 35, handlelength = 1)         
# plt.subplots_adjust(left = 0.05, right = 0.75, bottom = 0.1, top = 0.95, 
#                     wspace = 0.2, hspace = 0.2 )  
# plt.suptitle(r'$a$ = '+'%.1f, '%(growth_rate) +
#              r'$\lambda$ = '+ '%.d, ' %(division_rate) \
#             + r'M = ' + '%.d'%(M), x=0.35, y=.97, fontsize = 35)   
