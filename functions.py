"""Define the function that generates a trajectory of both processes.

Usage:
======
    Use 'trajectory' functions to compare running times and estimates values.
    The other functions are the succesives steps that build up the trajectory.
"""
import numpy as np

def pop_size(X):
    """Compute the size of the population from the vector of traits X.

    Parameters
    ----------
    list
        The list that contains all the traits of the individuals, dead or alive

    Returns
    -------
    int
        The number n of individuals alive in the population 
    """
    return(np.size(np.where(X > 0)))

def pop_resources(X):
    """Compute the total resource of the population from the traits vector X.

    Parameters
    ----------
    list
        The list that contains all the traits of the individuals, dead or alive

    Returns
    -------
    float
        The sum of the resources of the individuals alive in the population 
    """
    return(np.sum(X, where = X > 0))

def sde_spinal(X, t_0, t_1, e, growth_rate, sigma, delta):
    """Update the spinal vector of traits X between times t_0 and t_1

    Parameters
    ----------
    list
        The list that contains all the traits of the individuals at time t_0
    float
        The starting time
    float
        The ending time
    int
        The label of the spinal individual
    float
        The parameter giving the growth rate of every individuals
    float
        The parameter giving the intrinsic noise of every individuals
    float
        The parameter giving the extrinsic noise 

    Returns
    -------
    list
        The updated traits vector of individuals in the population at time t_1 
    """
    mask = np.full(len(X), True)
    mask[e] = False  
    dt = t_1 - t_0
    W_t = np.sqrt(dt) * np.random.normal(0, 1)
    B_t = np.sqrt(dt) * np.random.normal(0, 1, pop_size(X))
    X_temp = np.copy(X)
    rho = sigma / np.sqrt( 1 + delta ** 2)
    a = growth_rate + (rho * delta) ** 2 - sigma ** 2 / 2
    X_temp[X_temp > 0] = X_temp[X_temp > 0] * np.exp(a * dt) \
       * np.exp(rho * (B_t + delta * W_t)) 
    X_temp[e] = X_temp[e] * np.exp(rho ** 2 * dt)
    X_temp[((X_temp < 0) & (X_temp > -1)) & (mask)] = 0.000000001    
    return(X_temp)

def sde(X, t_0, t_1, growth_rate, sigma, delta):
    """Update the vector of traits X between times t_0 and t_1

    Parameters
    ----------
    list
        The list that contains all the traits of the individuals at time t_0
    float
        The starting time
    float
        The ending time
    float
        The parameter giving the growth rate of every individuals
    float
        The parameter giving the intrinsic noise of every individuals
    float
        The parameter giving the extrinsic noise 

    Returns
    -------
    list
        The updated traits vector of individuals in the population at time t_1 
    """
    dt = t_1 - t_0
    W_t = np.sqrt(dt) * np.random.normal(0, 1)
    B_t = np.sqrt(dt) * np.random.normal(0, 1, pop_size(X))
    X_temp = np.copy(X)
    rho = sigma / np.sqrt( 1 + delta ** 2)
    a = growth_rate - sigma ** 2 / 2
    X_temp[X_temp > 0] = X_temp[X_temp > 0] * np.exp(a * dt) \
       * np.exp(rho * (B_t + delta * W_t))
    X_temp[((X_temp < 0) & (X_temp > -1))] = 0.00000001    
    return(X_temp)

def theta_func(idx = 1):
    """Generate a realization of the sharing of resources at spliting events

    Parameters
    ----------
    int {0, 1, 2}
        The index for choosing the law at spliting events

    Returns
    -------
    float
        A realization of the sharing of resources at spliting events
    """
    if idx == 0:
        return(np.random.uniform())
    elif idx == 1:
        return(np.random.beta(2,2))
    elif idx == 2:
        return(np.random.beta(10,2))

def choose_spine(X):     
    """Choose a spinal individual in a population or a subpopulation

    Parameters
    ----------
    list
        A list with the traits of the individuals in the subpopulation

    Returns
    -------
    int
        The label of the spinal individual
    """ 
    v_0 = np.random.uniform() * pop_resources(X)
    c_0 = X[0]
    new_spine = 0
    while c_0 < v_0:
          new_spine += 1
          c_0 += X[new_spine]
    return(new_spine)
    
def trajectory_size(division_rate, death_rate, init_pop_size, t_max, t_pts):
    """Compute a skeleton of one trajectory of the colonial process size.

    Parameters
    ----------
    float
        The parameter giving the division rate of every individuals
    float
        The parameter giving the death rate of every individuals
    float
        The initial number of individuals 
    float
        The ending simulation time
    float
        The number of nodes of the skeleton 

    Returns
    -------
    list
        The times of the nodes of the skeleton
    list 
        The number of individuals at each skeleton nodes 
    """
    size_max = 2 + int(3 * t_max * (division_rate + death_rate *
                                           init_pop_size))
    t_jump = np.zeros(size_max)
    n_t = np.zeros(t_pts)
    n_t[0] = init_pop_size   
    t_simu = np.linspace(0, t_max, t_pts)
    c = 1
   # t_current = 0
    for i in range(1,size_max):
        rate = division_rate + death_rate * n_t[c-1]
        t_jump[i] = t_jump[i - 1] + np.random.exponential(1 / rate)
        if t_jump[i] > t_max:
            while c < t_pts:
                n_t[c] = n_t[c-1]   
                c += 1
            break
        #Boucle sur les pas de discretisation
        while c < t_pts and t_simu[c] < t_jump[i]:    
            n_t[c] = n_t[c-1]   
            #t_current = t_simu[c]
            c += 1
        #t_current = t_jump[i]
        if c == t_pts:
            break
        # Gestion des sauts à t_jump[i]    
        u = np.random.uniform() * rate
        if u <= division_rate:
            n_t[c] = n_t[c-1] + 1
        else:
            n_t[c] = n_t[c-1] - 1
        if n_t[c] == 0:
            break
        c += 1
    return(t_simu, n_t)

def trajectory_spinal_size(division_rate, death_rate, init_pop_size, t_max):
    """Compute a skeleton of one trajectory of the spinal process size.

    Parameters
    ----------
    float
        The parameter giving the division rate of every individuals
    float
        The parameter giving the death rate of every individuals
    float
        The initial number of individuals 
    float
        The ending simulation time

    Returns
    -------
    list
        The times of the nodes of the skeleton
    list 
        The number of individuals at each skeleton nodes 
    """
    size_max = 2 + int(3 * t_max * (division_rate + death_rate *
                                           init_pop_size))
    t_jump = np.zeros(size_max)
    n_t = init_pop_size  
    for i in range(1,size_max):
        rate = division_rate + death_rate * (n_t - 1)
        t_jump[i] = t_jump[i - 1] + np.random.exponential(1 / rate)
        if t_jump[i] > t_max:
            break
        # Gestion des sauts à t_jump[i]    
        if n_t == 1:
            n_t += 1
        else: 
            u = np.random.uniform() * rate
            if u <= division_rate:
                n_t += 1
            else:
                n_t -= 1
    return(n_t)

def trajectory_opti(division_rate, death_rate, growth_rate, sigma, delta,
               init_pop_size, init_pop_mass, t_max, t_pts):
    """Compute an exact skeleton of one trajectory of the colonial process.

    Parameters
    ----------
    float
        The parameter giving the division rate of every individuals
    float
        The parameter giving the death rate of every individuals
    float
        The parameter giving the growth rate of every individuals
    float
        The parameter giving the intrinsic noise of every individuals
    float
        The parameter giving the extrinsic noise 
    float
        The initial number of individuals
    float
        The initial mass of each individual 
    float
        The ending simulation time
    float
        The number of nodes of the skeleton 

    Returns
    -------
    list
        The times of the nodes of the skeleton
    list 
        The number of individuals at each skeleton nodes
    list
        The total resources at each skeleton nodes   
    """
    # Variables initialization
    traits = np.concatenate([init_pop_mass * np.ones(int(init_pop_size)),
                            (-1) * np.ones(3 * int(division_rate * t_max))])
    size_max = 2 + int(3 * t_max * (division_rate + death_rate *
                                       init_pop_size))
    t_jump = np.zeros(size_max)
    n_t = (-1) * np.ones(t_pts)
    r_t = (-1) * np.ones(t_pts)
    n_t[0] = pop_size(traits)   
    r_t[0] = pop_resources(traits)
    t_simu = np.linspace(0, t_max, t_pts)
    c = 1
    t_current = 0
    
    #Jumping times loop
    for i in range(1,size_max):        
        rate_sup = division_rate + death_rate * pop_size(traits)
        t_jump[i] = t_jump[i - 1] + np.random.exponential(1 / rate_sup)
        if t_jump[i] > t_max:
            while t_current < t_max:
                traits = sde(traits, t_current,t_simu[c], 
                              growth_rate, sigma, delta)
                n_t[c] = pop_size(traits)   
                r_t[c] = pop_resources(traits)
                t_current = t_simu[c]
                c += 1
            break
        
        #Discrete time steps loop
        while t_simu[c] < t_jump[i]:    
            traits = sde(traits, t_current, t_simu[c], 
                         growth_rate, sigma, delta)
            n_t[c] = pop_size(traits)   
            r_t[c] = pop_resources(traits)
            t_current = t_simu[c]
            c += 1
        if t_simu[c] >= t_jump[i]:
            traits = sde(traits, t_current, t_jump[i], 
                         growth_rate, sigma, delta)
            t_current = t_jump[i]
            
        # Computing jump events at t_jump[i]
        u = np.random.uniform() * rate_sup
        if u < division_rate:    # Division event
            v = np.random.uniform() * pop_resources(traits)
            index = 0
            count_mass = 0
            for x in traits[traits >= 0]:
                count_mass += x
                if count_mass >= v:
                    break
                else:
                    index += 1
            theta = theta_func()
            if np.size(np.where(traits == -1)[0]) > 0:
                traits[np.where(traits == -1)[0][0]] = (1 - theta)\
                                                            * traits[index]
            else:
                np.append(traits,(1 - theta) * traits[index])
            traits[index] *= theta
        else:   # Death event
            w = np.random.randint(0,np.size(traits[traits >= 0]))
            while traits[w] == 0:
                w = np.random.randint(0,np.size(traits[traits >= 0]))
            traits[w] = 0
        if pop_size(traits) == 0:
            break
    return(t_simu, n_t, r_t)

def trajectory_spinal_opti(division_rate, death_rate, growth_rate, sigma, 
                           delta, init_pop_size, init_pop_mass, t_max, t_pts):
    """Compute an exact skeleton of one trajectory of the spine process.

    Parameters
    ----------
    float
        The parameter giving the division rate of every individuals
    float
        The parameter giving the death rate of every individuals
    float
        The parameter giving the growth rate of every individuals
    float
        The parameter giving the intrinsic noise of every individuals
    float
        The parameter giving the extrinsic noise 
    float
        The initial number of individuals
    float
        The initial mass of each individual 
    float
        The ending simulation time
    float
        The number of nodes of the skeleton 

    Returns
    -------
    list
        The times of the nodes of the skeleton
    list 
        The number of individuals at each skeleton nodes
    list
        The total resources at each skeleton nodes
    list
        The spinal resources at each skeleton nodes    
    """
    # Variables initialization
    size_max = 2 + int(3 * t_max \
                       * (division_rate + death_rate * init_pop_size))
    t_jump = np.zeros(size_max)
    t_simu = np.linspace(0, t_max, t_pts)
    c = 1
    t_current = 0
    traits = np.concatenate([init_pop_mass * np.ones(int(init_pop_size)),
                            (-1) * np.ones(3 * int(division_rate * t_max))])
    n_t = (-1) * np.ones(t_pts)
    r_t = (-1) * np.ones(t_pts)
    y_t = (-1) * np.ones(t_pts)
    label_spine = choose_spine(traits)
    n_t[0] = pop_size(traits)   
    r_t[0] = pop_resources(traits)
    y_t[0] = traits[label_spine]
    
    #Jumping times loop
    for i in range(1,size_max):        
        rate_sup = division_rate + death_rate * (pop_size(traits) - 1)
        t_jump[i] = t_jump[i - 1] + np.random.exponential(1 / rate_sup)
        # Ending point managing
        if t_jump[i] > t_max:
            while t_current < t_max:
                traits = sde_spinal(traits, t_current,t_simu[c], label_spine,
                              growth_rate, sigma, delta)
                n_t[c] = pop_size(traits)   
                r_t[c] = pop_resources(traits)
                t_current = t_simu[c]
                c += 1
            break
        #Discrete time steps loop
        while t_simu[c] < t_jump[i]:    
            traits = sde_spinal(traits, t_current, t_simu[c], label_spine,
                         growth_rate, sigma, delta)
            n_t[c] = pop_size(traits)   
            r_t[c] = pop_resources(traits)
            t_current = t_simu[c]
            c += 1
        if t_simu[c] >= t_jump[i]:
            traits = sde_spinal(traits, t_current, t_jump[i], label_spine,
                         growth_rate, sigma, delta)
            t_current = t_jump[i]

        rate = division_rate + death_rate * (pop_size(traits) - 1)
        # Computing jump events at t_jump[i]
        u = np.random.uniform() * rate
        if u < division_rate: # Spliting event
            v = np.random.uniform() * pop_resources(traits)
            index = 0
            count_mass = 0
            for x in traits[traits >= 0]:
                count_mass += x
                if count_mass >= v:
                    break
                else:
                    index += 1
            theta = theta_func()
            if index == label_spine: # The spinal individual splits
                change_label = choose_spine(np.array((theta, 1- theta)))
                if change_label == False:
                    if np.size(np.where(traits == -1)[0]) > 0:
                        traits[np.where(traits == -1)[0][0]] = (1 - theta
                                                        ) * traits[index]
                    else:
                        np.append(traits,(1 - theta) * traits[index])
                    traits[index] *= theta
                else:
                    if np.size(np.where(traits == -1)[0]) > 0:
                        traits[np.where(traits == -1)[0][0]] = theta \
                            * traits[index]
                    else:
                        np.append(traits,(1 - theta) * traits[index])    
                    traits[index] *= 1 - theta
            else:   # A non spinal individual splits
                if np.size(np.where(traits == -1)[0]) > 0:
                    traits[np.where(traits == -1)[0][0]] = (1 - theta
                                                        ) * traits[index]
                else:
                    np.append(traits,(1 - theta) * traits[index])
                traits[index] *= theta
        elif u >= division_rate and u <= rate: # Death event
            w = np.random.randint(0,np.size(traits[traits >= 0]))
            while traits[w] == 0 or w == label_spine:
                w = np.random.randint(0,np.size(traits[traits >= 0]))
            traits[w] = 0
    return(t_simu, n_t, r_t, y_t)

def endpoint_direct(division_rate, death_rate, growth_rate, sigma, delta,
               init_pop_size, init_pop_mass, t_max):
    """Compute the last point of an exact trajectory of the colonial process.

    Parameters
    ----------
    float
        The parameter giving the division rate of every individuals
    float
        The parameter giving the death rate of every individuals
    float
        The parameter giving the growth rate of every individuals
    float
        The parameter giving the intrinsic noise of every individuals
    float
        The parameter giving the extrinsic noise 
    float
        The initial number of individuals
    float
        The initial mass of each individual 
    float
        The ending simulation time

    Returns
    -------
    list
        The traits of every individuals at ending time  
    """
    # Variables initialization
    traits = np.concatenate([init_pop_mass * np.ones(int(init_pop_size)),
                            (-1) * np.ones(3 * int(division_rate * t_max))])
    size_max = 2 + int(3 * t_max * (division_rate + death_rate *
                                       init_pop_size))
    t_jump = np.zeros(size_max)
    #Jumping times loop
    for i in range(1,size_max):        
        rate_sup = division_rate + death_rate * pop_size(traits)
        t_jump[i] = t_jump[i - 1] + np.random.exponential(1 / rate_sup)
        if t_jump[i] > t_max:
            traits = sde(traits, t_jump[i - 1], t_max, growth_rate, sigma, delta)
            break
        traits = sde(traits, t_jump[i - 1], t_jump[i], growth_rate, sigma, delta)
       
        # Computing jump events at t_jump[i]
        u = np.random.uniform() * rate_sup
        if u < division_rate:    # Division event
            v = np.random.uniform() * pop_resources(traits)
            index = 0
            count_mass = 0
            for x in traits[traits >= 0]:
                count_mass += x
                if count_mass >= v:
                    break
                else:
                    index += 1
            theta = theta_func()
            if np.size(np.where(traits == -1)[0]) > 0:
                traits[np.where(traits == -1)[0][0]] = (1 - theta)\
                                                            * traits[index]
            else:
                np.append(traits,(1 - theta) * traits[index])
            traits[index] *= theta
        else:   # Death event
            w = np.random.randint(0,np.size(traits[traits >= 0]))
            while traits[w] == 0:
                w = np.random.randint(0,np.size(traits[traits >= 0]))
            traits[w] = 0
        if pop_size(traits) == 0:
            break
    return(traits[traits >= 0])
