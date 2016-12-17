from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pickle
import scipy.optimize as opt


class Ising:
    grid = None
    Nside = -1
    Ndim = -1
    T_arr = None

#    Mag_markov_chain = []
#    Energy_markov_chain = []
    
    def __init__(self, Nside, Ndim, T_arr):
        self.shape = np.array([Nside for i in range(Ndim)])
        self.Ndim = Ndim
        self.Nside = Nside
        self.T_arr = T_arr
        self.init_grid()

    def init_grid(self):
        self.grid = np.random.choice([1,-1], self.shape)

    def monte_carlo_sim_T_arr(self, steps, alg, sample_freq):
        self.mean_mag = []
        self.mean_energy = []
        for temp in self.T_arr:
            self.init_grid()
            self.monte_carlo_evolve(steps, alg, temp, sample_freq)
            self.mcmc_macros()


    def mcmc_macros(self):
 #       expected_mag = np.mean(np.fabs(self.Mag_markov_chain))
#        expected_energy = np.mean(self.Energy_markov_chain)
        #energy

        self.mean_mag.append(expected_mag)
        self.mean_energy.append(expected_energy)

    def monte_carlo_evolve(self, steps, alg, temp, sample_freq=1):
        self.Mag_markov_chain = []
        self.Energy_markov_chain = []
        for k in range(steps):
            self.update_grid(alg, temp) #alg = {'metro', 'wolff', 'sw'}
            if (k/sample_freq)==int(k/sample_freq):
                self.calculate_macros()

    def update_grid(self, alg, temp):
        if alg=='metro':
            self.metropolis(temp)
        elif alg=='wolff':
            self.wolff(temp)
        elif alg=='sw':
            self.sw(temp)
        else:
            assert False, "alg={'metro', 'wolff', 'sw'} must be chosen"
    

    def compute_spin_flip_deltaE(self, center):
        deltaE = 0
        center_spin = self.grid[tuple(center)]
        neighbor_coords = self.get_neighbor_coords(center)
        for coord in neighbor_coords:
            neighbor_spin = self.grid[tuple(coord)]
            deltaE = deltaE + center_spin*neighbor_spin
        return 2*deltaE


    def metropolis(self, temp):
        rand_spin = np.array([np.random.randint(0,self.Nside) for i in range(self.Ndim)])
        delta_E = self.compute_spin_flip_deltaE(rand_spin)
        if delta_E < 0:
            self.grid[tuple(rand_spin)] = -1*self.grid[tuple(rand_spin)]
        else:
            prob = np.exp(-1.0/temp*delta_E)
            r = np.random.ranf()
            if r <= prob:
                self.grid[tuple(rand_spin)] = -1*self.grid[tuple(rand_spin)]

        return


    def sw(self, temp):
        flat_grid = np.arange(0,len(self.grid.flatten()))
        clustered_spins = []
        for flat_ind in flat_grid:
            tup_ind = np.unravel_index(flat_ind, self.shape)
            tup_spin = self.grid[tup_ind]
            new_spin = np.random.choice([1,-1])
            self.sw_cluster(temp, np.array(tup_ind), flat_ind, tup_spin, new_spin, clustered_spins)

    def sw_cluster(self, temp, center_coord, flat_ind, cluster_spin, new_spin, clustered_spins):
        if not np.in1d(flat_ind, clustered_spins):
            self.grid[tuple(center_coord)] = new_spin
            clustered_spins.append(flat_ind)
            neighbor_coords = self.get_neighbor_coords(center_coord)
            parallel_neighbor_coords = [coord for coord in neighbor_coords \
                                       if self.grid[tuple(coord)]==cluster_spin]
            for coord in parallel_neighbor_coords:
                w = np.exp(-2.0/temp)
                p = np.random.ranf()
                if p >= w:
                    coord_flat_ind = np.ravel_multi_index(coord, self.shape)
                    self.sw_cluster(temp, coord, coord_flat_ind, cluster_spin, new_spin, clustered_spins)



    def wolff(self, temp):
        rand_spin = np.array([np.random.randint(0,self.Nside) for i in range(self.Ndim)])
        cluster_spin_value = self.grid[tuple(rand_spin)]
        self.wolff_cluster(temp, rand_spin, cluster_spin_value)
            
    def wolff_cluster(self, temp, center_coord, cluster_spin_value):
        self.grid[tuple(center_coord)] = -1*self.grid[tuple(center_coord)]
        neighbor_coords = self.get_neighbor_coords(center_coord)
        parallel_neighbor_coords = [coord for coord in neighbor_coords \
                                       if self.grid[tuple(coord)]==cluster_spin_value]
        for coord in parallel_neighbor_coords:
            w = np.exp(-2.0/temp)
            p = np.random.ranf()
            if p >= w:
                self.wolff_cluster(temp, coord, cluster_spin_value)



    def calculate_macros(self):
        M = self.get_magnetization()
        self.Mag_markov_chain.append(M)
        En = self.get_energy()
        self.Energy_markov_chain.append(En)

        return


    def get_neighbor_coords(self, center):
        neighbor_coords = []
        for i in range(self.Ndim):
            for j in [1,-1]:
                coord = np.copy(center)
                coord[i] = (coord[i]+j+self.Nside)%self.Nside
                neighbor_coords.append(coord)
        return neighbor_coords

    def set_temp(self, Temp):
        self.T = Temp
    def get_temp(self):
        return self.T

    def get_magnetization(self):
        flat_grid = self.grid.flatten()
        Nspins = len(flat_grid)
        return 1.0/Nspins*np.sum(flat_grid)

    def get_energy(self):
        energy = 0
        flat_inds = np.arange(0,len(self.grid.flatten()))
        for flat_ind in flat_inds:
            tup_ind = np.unravel_index(flat_ind, self.shape)
            en_spin = self.compute_spin_flip_deltaE(list(tup_ind))/(-2.0)
            energy = energy + en_spin
        return energy/2


    def find_critical_temp(self,ydata):
        def logistic(t,A,K,Q,B,M):
            return A + (K-A)/(1+Q*np.exp(-B*(t-M)))
        A,K,Q,B,M = opt.curve_fit(logistic,self.T_arr,ydata)[0]
        return (B*M-np.log(1.0/Q))/B



    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    def load(self,filename):
        with open(filename, 'rb') as f:
            ising = pickle.load(f)
        return ising
