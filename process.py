import logging

import numpy as np
from scipy.interpolate import interp1d

"""
The Class Process() with inputs target, kind, data, comment, mass_ratio,
product, threshold, weight_ratio stores the properties of the process in 
question such as ionization, excitation, attachment, elastic, momentum and
effective 

target -> what is the species that will get transformed from the impact with
    an electron, as example
kind -> what type of process is involved? ionization, attachment, elastic...?
data -> cross sectional data as a function of electron energy 
comments -> comments to the process 
mass_ratio -> for elastic and effective collisions, the ratio of electron 
    mass to the target particle masss. If other mass_ratio defaults to None
product -> The end result of the collision process between the target and 
    collisional specie (usually electron)
threshold -> for ionization and excitation processes, electron energy loss is 
    quantified in eV. If other threshold defaults to None 
weight_ratio -> seldom used, required for superelastic processes 

-------------------------------------------------------------------------------
variables needed to initialize the system are

target_name -> initialized the target in the class
kind -> type of process
data -> arranges energy to cross sectional data in a array
x -> electron energy (eV)
y -> cross-sectional data (m2)
comment -> comments to the process 
mass_ratio -> ratio of electron mass to target particle mass
product -> end results of reactants interacting
threshold -> electron energy loss 
weight ratio -> superelastic reaction property
interp -> taking the data provided with the respective cross sections and 
whatnot, it creates an interpolation map
isnull -> placeholder 
in_factor -> from the IN_FACTOR dictionary, it gets the desired key depending 
    on the involved process 
shift_factor -> from the SHIFT_FACTOR dictionary, it gets the desired key
    depending on the involved process 

"""
class Process(object):
    IN_FACTOR = {'EXCITATION': 1,
                 'IONIZATION': 2,
                 'ATTACHMENT': 0,
                 'ELASTIC': 1,
                 'MOMENTUM': 1,
                 'EFFECTIVE': 1}
    
    SHIFT_FACTOR = {'EXCITATION': 1,
                    'IONIZATION': 2,
                    'ATTACHMENT': 1,
                    'ELASTIC': 1,
                    'MOMENTUM': 1,
                    'EFFECTIVE': 1}                                                                                                                                          
    
    def __init__(self, target=None, kind=None, data=None, comment='', mass_ratio=None, product=None, threshold=0, weight_ratio=None):
        self.target_name = target        
        self.target = None 
        self.kind = kind
        self.data = np.array(data)
        self.x = self.data[:, 0]
        self.y = self.data[:, 1]
        self.comment = comment
        self.mass_ratio = mass_ratio
        self.product = product
        self.threshold = threshold
        self.weight_ratio = weight_ratio

        self.interp = padinterp(self.data)                                                                                                          
        self.isnull = False
        
        self.in_factor = self.IN_FACTOR.get(self.kind, None)                                                                                        
        self.shift_factor = self.SHIFT_FACTOR.get(self.kind, None)                                                                                  
        
        #safety check
        if np.amin(self.data[:,0]) < 0:                                                                                                             
            raise ValueError("Negative energy in the cross section %s" % str(self))             
            
        #safety check
        if np.amin(self.data[:, 1]) < 0:                                                                                                            
            raise ValueError("Negative cross section for %s" % str(self))
        
        self.cached_grid = None            
    
    """
    The method scatterings() solve the integral for the scattering in and 
    scattering out, as shown in eqn (47) and 48 of Hagelaar, 2005
    
    g -> local logarithmic slope calculated in solver module
    eps -> grid cell centers of energy in the system
    """
    def scatterings(self, g, eps):
        # verification process 
        # print(self.j)
        if len(self.j) == 0:
            return np.array([], dtype='f')
        
        gj = g[self.j]
        epsj = eps[self.j]
        # solve integral
        r = int_linexp0(self.eps[:, 0], self.eps[:,1], self.sigma[:, 0], self.sigma[:, 1], gj, epsj)
        
        return r
        
    
    """
    The method set_grid_cache() sets a temporary grid for solving the cell 
    overlapping as described by Haalegar, 2005
    
    grid -> input of original grid, needed for setting up temporal grid  
    """
    def set_grid_cache(self, grid):
        # check if already defined. If not lets start settiong everything up
        if self.cached_grid is grid:
            return
        
        self.cached_grid = grid                                                                                                                       
        
        # setup of the cell shifted by a threshold energy u_k
        # there is a factor of 2 applied to the energy as the scattering-in 
        # term represents the secondary electrons being inserted at the same 
        # energy as primary electrons. only for inelastic collisions
        eps1 = self.shift_factor * grid.b + self.threshold  
        eps1[:] = np.maximum(eps1, grid.b[0] + 1e-9)
        eps1[:] = np.minimum(eps1, grid.b[-1] - 1e-9)
        
        # checks the mesh centers and the values provided by the cross 
        # sectional area and determines a unique set of nodes accordingly
        fltb = np.logical_and(grid.b >= eps1[0], grid.b <= eps1[-1])
        fltx = np.logical_and(self.x >= eps1[0], self.x <= eps1[-1])
        nodes = np.unique(np.r_[eps1, grid.b[fltb], self.x[fltx]])                                                                                   
        
        # interpolates cross sectional area to nodes provided 
        sigma0 = self.interp(nodes)                                                                                                                  
        
        # j nodes are based on the points within the mesh
        self.j = np.searchsorted(grid.b, nodes[1:]) - 1  
        # i nodes are based on the shift due to the threshold of energy
        self.i = np.searchsorted(eps1, nodes[1:]) - 1
        self.sigma = np.c_[sigma0[:-1], sigma0[1:]]
        self.eps = np.c_[nodes[:-1], nodes[1:]]
        
    def __str__(self):
        return "{%s: %s %s}" % (self.kind, self.target_name, "->" + self.product if self.product else "")
    
class NullProcess(Process):
    def __init__(self, target, kind):
        self.data = np.empty((0,2))
        self.interp = lambda x: np.zeros_like(x)
        self.target_name = target
        self.kind = kind
        self.isnull = True
        
        self.comment = None
        self.mass_ratio = None
        self.product = None
        self.threshold = None
        self.weight_ratio = None
        
        self.x = np.array([])
        self.y = np.array([])
        
    def __str__(self):
        return "{NULL}"
        
"""
the method padinterp() sets the interpolation for the cross sectional area required. 
"""
def padinterp(data):
    if data[0,0] > 0:
        x = np.r_[0.0, data[:,0], 1e8]
        y = np.r_[data[0,1], data[:,1], data[-1,1]]
    else:
        x = np.r_[data[:,0], 1e8]
        y = np.r_[data[:,1], data[-1, 1]]
        
    return interp1d(x, y, kind='linear')
    
"""
the method int_linexp0() integrates the integral present in the scattering 
equations. The integral in [a, b] is set up such as u(x)exp(g*(x0-x))*x 
assuming that u is linear with u({a,b}) = {u0, u1}. In other words, u is a 
function of electron energy and as such, it also needs to be considered in the 
integration process

NOTE: 
The expressions involve the following exponentials that are problematic:
expa = np.exp(g * (-a + x0))
expb = np.exp(g * (-b + x0))
The problems come with small g: in that case, the exp() rounds to 1 and 
neglects the order 1 and 2 terms that are required to cancel the 1/g**2 and 
1/g**3 below.  The solution is to rewrite the expressions as functions of e
xpm1(x) = exp(x) - 1, which is guaranteed to be accurate even for small x.

a -> electron energy array, i,j-1/2
b -> electron energy array, i,j+1/2
u0 -> cross sectional area (function of electron energy), i
u1 -> cross sectional area (function of electron energy), i+1
g -> logarithmic slope
x0 -> shifted electron energy array, ej
"""
def int_linexp0(a,b,u0,u1,g,x0):

    # exponent parts of the integration process 
    expm1a = np.expm1(g * (-a + x0))
    expm1b = np.expm1(g*(-b + x0))
    
    ag = a*g
    bg = b*g

    ag1 = ag + 1
    bg1 = bg + 1

    g2 = g*g
    g3 = g2*g
    
    # numerical representation of 
    # C_0\int_{\epsilon_1}^{\epsilon_2}\epsilon\exp((\epsilon_j-\epsilon)g_j)d\epsilon
    A1 = (expm1a * ag1 + ag - expm1b * bg1 - bg) / g2
    
    # numerical representation (without adding C1) of 
    # C_1\int_{\epsilon_1}^{\epsilon_2}\epsilon^2\exp((\epsilon_j-\epsilon)g_j)d\epsilon
    A2 = (expm1a * (2 * ag1 + ag * ag) + ag * (ag + 2) - expm1b * (2 * bg1 + bg * bg) - bg * (bg + 2)) / g3

    # calculation of coefficient of the linear function as u (cross sectional area) was assumed to 
    # be linear
    c0 = (a*u1-b*u0)/(a-b)
    c1 = (u0-u1)/(a-b)

    # coupling of integral calculated 
    r = c0*A1+c1*A2

    return np.where(np.isnan(r), 0.0, r)    
        
        
        
        