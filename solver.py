__docformat__ = "restructuredtext en"

import sys
import os
import logging

from math import sqrt
import numpy as np

import scipy.constants as co
from scipy import integrate
from scipy import sparse
from scipy.sparse.linalg import spsolve

# directory name, using a under to specify
file_dir = os.path.dirname(__file__)    
# finish appending current folder to system path                                        
sys.path.append(file_dir)                                                                                                                                                                                                         

from process import Process
from target import Target


GAMMA = sqrt(2*co.elementary_charge/co.electron_mass)                                                    
TOWNSEND = 1e-21                                                                                         
KB = co.k                                                                                                
ELECTRONVOLT = co.eV


class ConvergenceError(Exception):
    pass


"""
The Class BoltzmannSolver() solves the boltzmann equation so a EEDF can be
obtained and derived properties, such as reaction rate and electron mobility
within the system 

grid -> mesh to be used 

-------------------------------------------------------------------------------
variables needed to initialize the system are

EN -> reduced electric field in the system. Defined by the user 
kT -> background gas temperature. Defined by the user 
grid -> mesh to be used to solve the system 
target -> target background gas in the system 
"""
class BoltzmannSolver(object):

    def __init__(self,grid):
        self.density = dict()                                                                                          
        self.EN = None
        self.kT = None
        self.grid = grid                                                                               
        self.target = {}
        self.ii = 0
    
    """
    the function _get_grid() is a getter for the input variable grid
    """
    def _get_grid(self): 
        return self._grid
    
    """
    the function _set_gri()d is a setter for the grid in the system 
    """
    # it is desired when setting the class to pass throught the setter 
    # first and foremost. Having self._grid allows us to do that
    def _set_grid(self, grid):
        self._grid = grid                                                                                
        
        self.benergy = self.grid.b                                                                       
        self.cenergy = self.grid.c                                                                       
        self.denergy = self.grid.d                                                                        
        
        self.denergy32 = self.benergy[1:]**1.5 - self.benergy[:-1]**1.5                                 
        
        self.n = grid.n                                                                                  
        
    # automatically gets and set grid in the system if needed accordingly
    grid = property(_get_grid, _set_grid)                                                                
    
    """
    the function set_density sets the density of the background gas in the 
    system
    
    species -> string depicting the background gas
    density -> fractional composition of the background gas(es) involved
    """    
    def set_density(self, species, density):                                                              
        self.target[species].density = density
   
    """
    The method load_collisions() loads the collisions to Process and Target 
    modules so the input information can be used by the solver

    dict_processes -> argument consisting of the dictionaries created by 
    the parser module
    """
    def load_collisions(self, dict_processes):
        # executes add_process from the processes in provided dictionary                                                                                                                      
        plist = [self.add_process(**p) for p in dict_processes] 
        
        # as target is a dictionary, target.items() return the key and the item
        # associated with it                                          
        for key, item in self.target.items(): 
            # checks if a elastic reaction exist or otherwise creates it from
            # effective 
            item.ensure_elastic()                                                                        
            
        return plist
    
    """
    The method add_process() adds the dictionary of reactions to Process() 
    module
    
    **kwargs -> keyworded argument passed to Process module 
    """
    def add_process(self, **kwargs):
        proc = Process(**kwargs)        
        
        try:
            target = self.target[proc.target_name]
        except KeyError:
            # initialize target module if it has not been done before
            target = Target(proc.target_name)
            # initialization of dictionary target
            self.target[proc.target_name] = target    
        
        # when it checks that the target key exist, it add the process to 
        # the target in question
        target.add_process(proc)
        
        return proc                                                                                      
    
    
    def search(self, signature, product=None, first=True):
        if product is not None:
            l = self.target[signature].by_product[product]
            
            if not l:
                raise KeyError("Process %s not found" % signature)
                
            return l[0] if first else l
        
        t, p = [x.strip() for x in signature.split('->')]
        return self.search(t, p, first=first)
    
    """
    The method iter_elastic(), from the Target() class defined by previous
    functions, and stored in a dictionary, obtains the target and the process 
    that is elastic 
    """
    def iter_elastic(self):
        for target in self.target.values():                                                             
            if target.density > 0:
                for process in target.elastic:  
                    yield target, process                                                               

    """
    The method iter_inelastic(), from the Target() class defined by previous
    functions, and stored in a dictionary, obtains the target and the process 
    that is inelastic  
    """
    def iter_inelastic(self):
        for target in self.target.values():
            if target.density > 0:
                for process in target.inelastic:
                    yield target, process
                    
    # def iter_ionization(self):
    #     for target in self.target.values():
    #         if target.density > 0:
    #             for process in target.ioniztion:
    #                 yield target, process
                    
    def iter_growth(self):
        for target in self.target.values():
            for process in target.ionization:
                yield target, process
                
            for process in target.attachment:
                yield target, process
                
    def iter_everything(self):
        for target in self.target.values():
            for process in target.everything:
                yield target, process
                
    # change to code LINE
    # def iter_attachment(self):
    #     for target in self.target.values():
    #         if target.density > 0:
    #             for process in target.attachmnt:
    #                 yield target, process
            
    # changes to code LINE
    def iter_all(self):
        # for t, k in self.iter_attachment():
        #     yield t,k
        for t, k in self.iter_elastic():
            yield t, k
        for t, k in self.iter_inelastic():
            yield t, k
        # for t, k in self.iter_ionization():
        #     yield t, k
        
    def iteration_all(self):
        for t,k in self.iter_everything():
            yield t, k
            
    def iter_momentum(self):
        return self.iter_all()
    
    """
    The method init() initializes the solver variables before the solver starts
    
    """
    def init(self):
        
        # initializes the elastic momentum-transfer cross section summation
        self.sigma_eps = np.zeros_like(self.benergy)     
        # initializes for each cell the nodes in each cell for the total
        # momentum-transfer cross section 
        self.sigma_m = np.zeros_like(self.benergy)                                                       

        # elastic processes 
        for target, process in self.iter_elastic():
            # interpolates cell centre's specified electron energy from the 
            # mesh in comparison to the provided cross sectional data.
            s = target.density*process.interp(self.benergy)
            self.sigma_eps += 2*target.mass_ratio*s
            self.sigma_m += s
            
            # temporal mesh for overlapping process 
            process.set_grid_cache(self.grid)
        
        # inelastic processes 
        for target, process in self.iter_inelastic():
            self.sigma_m += target.density*process.interp(self.benergy)                                  
            process.set_grid_cache(self.grid)   
            
        # for target, process in self.iter_ionization():
        #     self.sigma_m += target.density*process.interp(self.benergy)                                  
        #     process.set_grid_cache(self.grid)   

                                                      
            
        # negative flow velocity ~incomplete representation~
        self.W = -GAMMA*self.benergy**2*self.sigma_eps
        
        # (1) diffusion coefficient in the system ~incomplete representation~
        self.DA = (GAMMA/3. * self.EN**2 * self.benergy)
        
        # (2) diffusion coefficient in the system ~incomplete representation~
        self.DB = (GAMMA * self.kT * self.benergy**2 * self.sigma_eps)
        
        logging.info("Solver succesfully initialized/updated")
        
    
    """
    The method maxwell() initializes a maxwellian distribution with a 
    arbitrary electron energy. This is done to initialized the calculation of
    the EEDF
    
    kT -> equilibrium temperature of the system, in eV 
    """
    def maxwell(self, kT):
        # 2\sqrt{\frac{\epsilon}{\pi}}(kT_e)^{-3/2}\exp(-\epsilon/kT_e)
        return (2*np.sqrt(self.cenergy/np.pi)*kT**(-3./2.) * np.exp(-self.cenergy/kT))
    
    
    """
    method iterate() solves the EEDF and is called every time a iteration is 
    required. 
    
    f0 -> reference EEDF from the initialization or the previous iteration 
        step
    delta -> 
    """
    def iterate(self, f0, delta=1e14):
        A, Q = self._linsystem(f0)
        
        f1 = spsolve(sparse.eye(self.n)+delta*A-delta*Q, f0)
        
        return self._normalized(f1)
    
    """
    The method converge() initializes the the solution of the EEDF to a 
    solution where the difference of n and n-1 are less than a user defined 
    threshold 
    
    f0 ->
    maxn ->
    rtol ->
    delta0 ->
    m ->
    full -> 
    """
    def converge(self, f0, maxn=100, rtol=1e-5, delta0=1e14, m=4.0, full=False, **kwargs):
        err0 = err1 = 0
        delta = delta0
        
        # iterates solution until rtol is less than the user defined threshold
        for i in range(maxn):
            if 0 < err1 < err0:
                delta = delta*np.log(m)/(np.log(err0)-np.log(err1))
                
            f1 = self.iterate(f0, delta=delta, **kwargs)
            err0 = err1
            err1 = self._norm(abs(f0-f1))
            
            logging.debug("After iteration %3d, err = %g (target: %g)" % (i+1,err1,rtol))
            
            if err1 < rtol:
                logging.info("Convergence achieved after %d iterations. " 
                             "err = %g" % (i+1,err1))
                if full:
                    return f1, i+1, err1
                
                return f1
            f0 = f1
        
        logging.error("Convergence failed")
        
        raise ConvergenceError()
        
    def _linsystem(self, F):
        Q = self._PQ(F)
        
        # contributions calculated from Q(integral) and considering the 
        # distribution function, the following is obtained 
        # \nu = \gamma\int^{\infty}_{0}\Bigg(\sum_{k = ionization}x_k\sigma_k-\sum_{k=atttachment}x_k\sigma_k\Bigg)\epsilon F_0d\epsilon
        nu = np.sum(Q.dot(F))
        
        # \sigma_m = \sigma_m + \frac{\nu}{\gamma\epsilon^{1/2}}
        sigma_tilde = self.sigma_m + nu/np.sqrt(self.benergy) / GAMMA
        
        
        G = 2*self.denergy32*nu/3
        
        A = self._scharf_gummel(sigma_tilde, G)
        
        return A, Q
    
    def _norm(self, f):
        return integrate.simps(f*np.sqrt(self.cenergy), x=self.cenergy)
    
    def _normalized(self, f):
        N = self._norm(f)
        return f/N
    
    
    def _scharf_gummel(self, sigma_tilde, G=0):
        
        # drift diffusion equation put together
        D = self.DA / (sigma_tilde) + self.DB
        
        # setup of the peclet number used in the exponential part of the 
        # numerical scheme 
        z = self.W*np.r_[np.nan, np.diff(self.cenergy),np.nan] / D
        
        a0 = self.W/(1-np.exp(-z))
        a1 = self.W/(1-np.exp(z))
        
        
        diags = np.zeros((3,self.n))
        
        
        diags[0,0] = a0[1]
        
        #
        diags[0,1:] = a0[2:] - a1[1:-1]
        diags[1,:] = a1[:-1]
        diags[2,:] = -a0[1:]

        diags[2, -2] = -a0[-2]
        diags[0, -1] = -a1[-2]

        diags[0, :] += G

        A = sparse.dia_matrix((diags, [0,1,-1]), shape=(self.n, self.n))

        return A

    """
    the method g() is a logarithmic slope, which is needed in the scatttering
    in and scattering out processes 
    
    F0 -> distribution function previously calculated. 
    """
    def _g(self, F0):
        Fp = np.r_[F0[0], F0, F0[-1]]
        cenergyp = np.r_[self.cenergy[0], self.cenergy, self.cenergy[-1]]
        
        self.ii +=1
        print(self.ii)        
        # g_i = \frac{1}{\epsilon_{i+1}-\epsilon_{i-1}}\ln\Big(\frac{F_{0, i+1}}{F_{0,i-1}}\Big)
        g = np.log(Fp[2:]/Fp[:-2])/(cenergyp[2:]-cenergyp[:-2])

        
        return g
    
    def _PQ(self, F0, reactions=None):
        
        # generates a sparse matrix of dimensions (n,n)
        PQ = sparse.csr_matrix((self.n,self.n))
        
        # calculates the logarithmic slope required for calculation of the
        # scattering terms
        g = self._g(F0)
        
        
        if reactions is None:
            reactions = list(self.iter_inelastic())
            
        data = []
        rows = []
        cols = []
        
        # simultaneously iterates throught two elements at the same time, as 
        # that was the iter.inelastic() returned in the first place 
        for t,k in reactions:
            r = t.density*GAMMA*k.scatterings(g, self.cenergy)
            in_factor = k.in_factor
            
            # scattering in, scattering out integrals 
            data.extend([in_factor*r, -r])
            #(i, j)
            rows.extend([k.i, k.j])
            #(j, j)
            cols.extend([k.j, k.j])

        # data, rows and cols generated by before, and extended, are rather 
        # saved as elements of arrays. by np.hstack, all the data is stored
        # in a single array 
        data, rows, cols = (np.hstack(x) for x in (data, rows, cols))

        # sparse matrix in COOrdinate formate 
        PQ = sparse.coo_matrix((data, (rows, cols)),shape=(self.n, self.n))
        
        return PQ
    
    def rate(self, F0, k, weighted=False):
        
        g = self._g(F0)
        
        if isinstance(k, (bytes, str)):
            k = self.search(k)
            
        k.set_grid_cache(self.grid)
        
        r = k.scatterings(g, self.cenergy)
        
        P = sparse.coo_matrix((GAMMA*r, (k.j, np.zeros(r.shape))),shape=(self.n,1)).todense()

        P = np.squeeze(np.array(P))
        
        rate = F0.dot(P)
        
        if weighted:
            rate *= k.target.density
        
        return rate
    
    def mobility(self, F0):
        
        DF0 = np.r_[0.0, np.diff(F0)/np.diff(self.cenergy),0.0]
        Q = self._PQ(F0, reactions=self.iter_growth())
        
        nu = np.sum(Q.dot(F0))/GAMMA
        
        sigma_tilde = self.sigma_m + nu / np.sqrt(self.benergy)
        
        y = DF0*self.benergy/sigma_tilde
        y[0] = 0
        
        return -(GAMMA/3)*integrate.simps(y, x=self.benergy)
    
    def diffusion(self, F0):
        
        Q = self._PQ(F0, reactions=self.iter_growth())
        
        nu = np.sum(Q.dot(F0))/GAMMA
        
        sigma_m = np.zeros_like(self.cenergy)
        for target, process in self.iter_momentum():
            s = target.density*process.interp(self.cenergy)
            sigma_m += s
        
        sigma_tilde = sigma_m + nu/np.sqrt(self.cenergy)
        
        y = F0*self.cenergy/sigma_tilde
        
        return (GAMMA/3)*integrate.simps(y, x=self.cenergy)
    
    def mean_energy(self, F0):
        de52 = np.diff(self.benergy**2.5)
        return np.sum(0.4*F0*de52)    