""" Plot the distribution of resources in the original process, estimated from 
Monte-Carlo method with the spinal algorithm.

Usage:
======
    Simply run the script and wait for the plot. You can change the parameter
    values in the first section but please refrain from drastically incrising
    the number of Monte Carlo iterations M.
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
matplotlib.rcParams['savefig.dpi'] = 300
matplotlib.rcParams["figure.dpi"] = 200
  
# Parameters
division_rate = 3
death_rate = 1
list_t_max = [0.05,1,2,10]
growth_rate = 1
init_pop_mass = 1
init_pop_size = 10
sigma = 10
list_delta = [0,3,30]
M = 1000  # Large enough (<5% relative error)
t_pts = 30

# Initialization   
fig,axes = plt.subplots(1, 3,figsize=(12,3),constrained_layout=True)
c = 0
K_theta = 0
for i in range(10000):
    thet = functions.theta_func()
    K_theta += thet * (1 - thet) / 10000
    
# Main computation loop
x_plot = np.linspace(1,init_pop_size,init_pop_size)
for l in range(3):
    axes[l].plot(x_plot, np.ones(init_pop_size) / init_pop_size, 
                 label = r'$ T = 0$',linestyle='--', color = 'black')
for k in range(np.size(list_t_max)):
    t_max = list_t_max[k]
    for l in range(3):
        c += 1
        delta = list_delta[l]
        rho = sigma ** 2 / (1 + delta ** 2)
        
        estimator_traits = np.zeros(init_pop_size * 2)
        
        # Variance Monte-Carlo estimation 
        traj_r = []
        for i in tqdm(range(0, M), mininterval=1, ncols=100,
           desc="Trajectories computations. Step %1.d/12. Progress: "%(c)):
            traits = np.zeros(init_pop_size * 2)
            ts = functions.endpoint_direct(division_rate, 
                                    death_rate, growth_rate, sigma, delta,
                                    init_pop_size, init_pop_mass, t_max)
            ts = sorted(ts[ts > 0], reverse=True)
            n_indiv = len(ts)
            traits[0:n_indiv] = np.asarray(ts)
            if np.sum(traits) != 0:
                traits = traits / np.sum(traits)
            estimator_traits += traits / M
        estimator_traits = estimator_traits / np.sum(estimator_traits)
        estim_to_plot = np.extract(estimator_traits != 0,estimator_traits)
        if k == 0:
            axes[l].plot(x_plot[0:np.minimum(init_pop_size,np.size(estim_to_plot))],
                         estim_to_plot[0:init_pop_size],'o-',fillstyle='none',
                         markersize=8, label = r'$T = $' + '%.1f'%(t_max))
        else:
            axes[l].plot(x_plot[0:np.minimum(init_pop_size,np.size(estim_to_plot))],
                         estim_to_plot[0:init_pop_size],'o-',fillstyle='none',
                         markersize=8, label = r'$T = $' + '%.d'%(t_max))
        if l == 2:
            axes[l].set_title(r'$\frac{\sigma^2}{1 + \delta^2}$ = ' 
                              + '%.1f'%(rho) + '    ', fontsize = 18, y = 1.0, 
                              pad = -25)
        else:
            axes[l].set_title(r'$\frac{\sigma^2}{1 + \delta^2}$ = ' 
                              + '%.d'%(rho) + '    ', fontsize = 18, y = 1.0,
                              pad = -25)
        


# Plot   
axes[0].set_ylabel('Relative resources', fontsize = 15)
axes[1].set_yticklabels([])
axes[2].set_yticklabels([])        
for i in range(3):
    axes[i].set_xlabel('Sorted individuals label', fontsize = 15)
    axes[i].tick_params(axis='both', labelsize = 12)
    axes[i].set_ylim([-0.05,1.01])
    axes[i].grid(True, which="both", color = 'lightgray', ls = "-", alpha = 0.2)
    handles, labels = axes[i].get_legend_handles_labels()   
    axes[i].legend(handles, labels, loc='upper center', ncol = 1,
                 bbox_to_anchor=(0.855, 1.02),fancybox=False, shadow=False,
                 fontsize = 10, handlelength = 1)  
plt.subplots_adjust(left = 0.05, right = 0.95, bottom = 0.08, top = 0.89, 
                    wspace = 0.2, hspace = 0.25 ) 

