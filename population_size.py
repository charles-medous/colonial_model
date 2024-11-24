""" Plot the mean population size and the survival probability of the colonial 
process, estimated from Monte-Carlo method with the direct algorithm.

Usage:
======
    Simply run the script and wait for the plot. You can change the parameter
    values in the first section but please refrain from increasing the number 
    of Monte Carlo iterations M.
"""
import functions
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm
import scipy.special as sc

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

########## Plot E[N_t] with 3*3 subplot
# Parameters
list_division_rate = [0.5, 0.7, 1, 1.2, 1.5, 3, 4, 8, 10]
death_rate = 1
init_pop_size = 1
t_max = 30

# Initialization
M = 50000  # Large enough
t_pts = int(2 * t_max)
times = np.linspace(0, t_max, t_pts)
fig,axes = plt.subplots(3, 3)

# Main computation loop
for k in range(3):
    if k == 0:
        t_max = 12
    elif k == 1:
        t_max = 25
    else:
        t_max = 100
    t_pts = int(100 * t_max)
    times = np.linspace(0, t_max, t_pts)
    for l in range(3):
        traj_n = []
        estimator_n = np.zeros(t_pts)
        division_rate = list_division_rate[k * 3 + l]
        upper_bound = np.zeros(t_pts)
        lower_bound = (division_rate * times + init_pop_size)\
                             * np.exp(- death_rate * times)                     
        t_1 = init_pop_size * (np.exp(division_rate / death_rate) - 1)
        n_t_1 = division_rate * (1 - np.exp(- death_rate * t_1)) / death_rate \
            + init_pop_size * np.exp(- death_rate * t_1)
        for m in range(0, t_pts):
            if times[m] <= t_1:
                upper_bound[m] = division_rate / death_rate \
                        * (1 - np.exp(- death_rate * times[m]))\
                        + init_pop_size * np.exp(- death_rate * times[m])
            else:
                upper_bound[m] = t_1 * division_rate / death_rate \
                * (sc.expi(death_rate * times[m]) - sc.expi(death_rate * t_1))\
                * np.exp(- death_rate * times[m])\
                + n_t_1 * np.exp(- death_rate * (times[m] - t_1))                                   
        for i in tqdm(range(0, M), mininterval=1, ncols=100,
                      desc="x_bar mean estimation. Progress: "):
            t, data1 = functions.trajectory_size(division_rate, death_rate, 
                                              init_pop_size, t_max, t_pts)
            for j in range(0, t_pts):
                estimator_n[j] += data1[j]/M                
        axes[k][l].plot(times,estimator_n, 
               label = 'estimator ' +r'$ E\left[ N_t\right]$', color = 'black')
        axes[k][l].plot(times, lower_bound , label = 'lower bound')
        axes[k][l].plot(times, upper_bound, label = 'upper bound') 
        axes[k][l].tick_params(axis='both', labelsize = 15)  
        axes[k][l].set_title(r'$\lambda$ = ' + '%.2f'%(division_rate), 
                             fontsize = 30)
        
# Plot        
for i in range(3):
    for j in range(3):
        axes[i][j].tick_params(axis='both', labelsize = 30)
        axes[2][j].set_xlabel(r'Time (s)', fontsize = 40)
        if i == 0:
            axes[i][j].set_ylim([0.0000001,1.01])
        elif i == 1:
            axes[i][j].set_ylim([0.00001,3.1])
        else:
            axes[i][j].set_ylim([0.001,10.1])
handles, labels = axes[0][0].get_legend_handles_labels()
axes[0][1].legend(handles, labels,loc='upper center', ncol = 3,
             bbox_to_anchor=(1.05, 1.65), fancybox=False, shadow=False,
             fontsize = 40)      
plt.subplots_adjust(left = 0.03, right = 0.98, bottom = 0.1, top = 0.86, 
                    wspace = 0.2, hspace = 0.28 )

########## Plot Proba survie 3*3
# Parameters
list_division_rate = [0.5, 0.7, 1, 1.2, 1.5, 3, 4, 5, 6]
death_rate = 1
init_pop_size = 1
t_max = 30

# Initialization
M = 200
t_pts = int(2 * t_max)
times = np.linspace(0.01, t_max, t_pts)
fig,axes = plt.subplots(3, 3)

# Main computation loop
for k in range(3):
    if k == 0:
        M = 5000
        t_max = 10
        t_pts = int(100 * t_max)
        times = np.linspace(0, t_max, t_pts)
    elif k == 1:
        M = 5000
        t_max = 100
        t_pts = int(100 * t_max)
        times = np.linspace(0, t_max, t_pts)
    else:
        M = 5000
        t_max = 400
        t_pts = int(100 * t_max)
        times = np.linspace(0, t_max, t_pts)
    for l in range(3):
        traj_n = []
        estimator_n = np.zeros(t_pts)
        division_rate = list_division_rate[k * 3 + l]
        upper_bound = np.zeros(t_pts, dtype=np.float64)                           
        upper_bound = np.minimum(init_pop_size / (times  * division_rate)\
                            * (np.exp(division_rate / death_rate) - 1), 1)
        t_1 = init_pop_size * (np.exp(division_rate / death_rate) - 1)
        n_t_1 = division_rate * (1 - np.exp(- death_rate * t_1)) / death_rate \
            + init_pop_size * np.exp(- death_rate * t_1)                    
        for i in tqdm(range(0, M), mininterval=1, ncols=100,
                      desc="x_bar mean estimation. Progress: "):
            t, data1 = functions.trajectory_size(division_rate, death_rate, 
                                              init_pop_size, t_max, t_pts)
            data1 = data1 > 0
            for j in range(0, t_pts):
                estimator_n[j] += data1[j] /M                
        axes[k][l].plot(times,estimator_n, 
               label = r'estimator ' +r'probability survival', color = 'black')
        axes[k][l].plot(times, upper_bound, label = 'upper bound') 
        axes[k][l].tick_params(axis='both', labelsize = 15)  
        axes[k][l].set_title(r'$\lambda$ = ' + '%.2f'%(division_rate), 
                             fontsize = 35)
        #axes[k][l].set_yscale('log') # Uncomment for log scale plot

# Plot
for i in range(3):
    for j in range(3):
        axes[i][j].tick_params(axis='both', labelsize = 30)
        axes[2][j].set_xlabel(r'Time (s)', fontsize = 40)
        if i == 0:
            axes[i][j].set_ylim([0.0000001,1.01])
        elif i == 1:
            axes[i][j].set_ylim([0.00001,1.01])
        else:
            axes[i][j].set_ylim([0.001,1.01])
axes[0][1].legend(handles, labels,loc='upper center', ncol = 3,
             bbox_to_anchor=(1.21, 1.65), fancybox=False, shadow=False,
             fontsize = 40)      
plt.subplots_adjust(left = 0.03, right = 0.98, bottom = 0.1, top = 0.86, 
                    wspace = 0.2, hspace = 0.35 )       
        