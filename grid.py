import numpy as np
from scipy.interpolate import interp1d

"""
The class Grid() with inputs x0, x1 and n initialize the mesh generation 
for solving the EEDF in the system. The arguments are as follow

x0 -> lowest energy boundary (lowest value of grid (eV))
x1 -> highest energy boundary (highest value of grid (eV))
n -> number of cells in the mesh

-------------------------------------------------------------------------------
variables needed to initialize the system are

delta -> difference between lowest and highest energy boundary
fx0, fx1 -> obtained from the child function f on the methods within class 
    Grid. sets the highest and lowest limits depending on the type of mesh 
    selected
fx -> sets all the mesh points in a linear manner. 
    To be further converted within class Grid
b -> conversion of fx to represent the real mesh points
c -> center of the mesh
d -> determines the deltas between cells. 
df -> difference between boundaries of first cell. Only usable for linear 
    cases in cell method as there is no geometric distortion
df32 -> needed in other code dependencies that integrates eps**(1/2)*f.
    determines the deltas between cells

-------------------------------------------------------------------------------
functions within the class

interpolate(f, other)
    using interp1d scipy module, it linearly interpolates the x values 
    of a newly defined grid to a old grid and distribution function 
    
cell(x)
    outputs the cell number. only works for linear meshes 
"""
class Grid(object):
    def __init__(self, x0, x1, n):

        self.x0 = x0                                                            
        self.x1 = x1                                                            
        
        self.delta = x1-x0                                                       
        self.fx0 = self.f(x0)                                                   
        self.fx1 = self.f(x1)                                                   
        
        self.n = n                                                              
        
        fx = np.linspace(self.fx0, self.fx1, self.n + 1)                        
        self.b = self.finv(fx)                                                  
        
        self.c = 0.5*(self.b[1:] + self.b[:-1])                                  
        self.d = np.diff(self.b)                                                
        self.df = fx[1] - fx[0]                                                
        self.d32 = self.b[1:]**1.5 - self.b[:-1]**1.5                           
        
        self._interp = None                 
        
    def interpolate(self, f, other):                                            
        if self._interp is None:                                               
            self._interp = interp1d(np.r_[other.x0, other.c, other.x1],        
                                    np.r_[f[0], f, f[-1]],
                                    bounds_error=False, fill_value=0)           
       
        return self._interp(self.c)                                         
        
    def cell(self, x):
        return int((self.f(x)-self.fx0)/self.df)                                

"""
The class LinearGrid() defines a linear mesh for the EEDF to be solved

-------------------------------------------------------------------------------
functions within the class

f(x)
    sets the values of the lower and uppermost values of the grid for further
    processing by finv.
finv(w)
    takes the linear grid defined by the class as baseline using uppermost
    and lowermost limits of f(x) function and converts it to to the mesh.
"""    
class LinearGrid(Grid):
    def f(self, x):
        return x                                                               
    def finv(self, w):
        return w                                                                

"""
The class QuadraticGrid() defines a squared spaced mesh for the 
EEDF to be solved. 

-------------------------------------------------------------------------------
functions within the class

f(x)
    sets the values of the lower and uppermost values of the grid for further
    processing by finv.
finv(w)
    takes the linear grid defined by the class as baseline using uppermost
    and lowermost limits of f(x) function and converts it to to the quadratic 
    mesh. x0 is added at the end of the return in order to define the 
    starting point
"""   
class QuadraticGrid(Grid):
    def f(self, x):
        return np.sqrt(x - self.x0)                                             
    
    def finv(self, w):
        return w**2 + self.x0

"""
The class GeometricGrid() defines a geometric spaced mesh for the 
EEDF to be solved. The arguments are as follow

x0 -> lowest energy boundary (lowest value of grid (eV))
x1 -> highest energy boundary (highest value of grid (eV))
n -> number of cells in the mesh
r -> common ratio

-------------------------------------------------------------------------------
functions within the class

f(x)
    in form of
    \frac{\log \frac{1+(x_1-x_0)(r^n-1)}{x_1-x_0}}{\log r}
    for x0, it defaults to a zero value
    for x1, it defaults to n, the number of cells in the system 

finv(w)
    in form of 
    x_0+(x_{th}-x_0)\Big[\frac{r^i-1}{r^n-1}\Big]\\i=0,1,2,3,...,n
    it increases the mesh in a geometric manner until the mesh ends at x_th
"""      
class GeometricGrid(Grid):
    def __init__(self, x0, x1, n, r=1.1):                                      
        self.r = r                                                              
        self.logr = np.log(r)
        self.rn_minus_1 = np.exp(n*self.logr) - 1                              
        
        #avoid using the base class name explicitly
        super(GeometricGrid, self).__init__(x0,x1,n)
    
    def f(self,x):
        return (np.log(1 + (x - self.x0) * self.rn_minus_1/self.delta)/self.logr)
    
    def finv(self, w):
        # print(w)
        return (self.x0 + self.delta*(np.exp(w*self.logr) - 1) / (self.rn_minus_1))
    
"""
The class LogGrid() defines a logarithmic grid

x0 -> lowest energy boundary (lowest value of grid (eV))
x1 -> highest energy boundary (highest value of grid (eV))
n -> number of cells in the mesh
s -> failsafe variable to avoid log(0)

-------------------------------------------------------------------------------
functions within the class

f(x)
    sets the values of the lower and uppermost values of the grid for further
    processing by finv   

finv(w)
    takes the linear grid defined by the class as baseline using uppermost
    and lowermost limits of f(x) function and converts it to to the logarithmic 
    mesh. x0 is added at the end of the return in order to define the starting 
    point
"""  
class LogGrid(Grid):
    def __init__(self, x0, x1, n, s=10):                                       
        self.s = s
        
        #avoid using the base class name explicitly
        super(LogGrid, self).__init__(x0, x1, n)                               
        
    def f(self, x):                                                            
        return np.log(self.s + (x-self.x0))                                   
    
    def finv(self, w):                                                         
        return np.exp(w) - self.s + self.x0
    
"""
The class AutomaticGrid() automatically sets a grid using a previous
estimation of the EEDF

grid -> grid used to calculated EEDF 
f0 -> EEDF
"""  
class AutomaticGrid(Grid):
    def __init__(self, grid, f0, delta=1e-4):
        
        # np.cumsum -> return the cumulative sum of the elements along a axis
        # np.cumsum(grid.d32*f0) resembles the energy distribution 
        # normalization
        # \int _0 ^{\infty}\epsilon^{1/2}F_0\ d\epsilon = 1
        cum = np.r_[0.0, np.cumsum(grid.d32*f0)]
        
        # however, as it was not quite a proper integration, some normalization
        # needs to be done
        cum[:] = cum/cum[-1]
        
        interp = interp1d(cum, grid.b)
        nnew = np.linspace(0.0, 1.0, grid.n + 1)
        
        self.n, self.x0, self.x1 = grid.n, grid.x0, grid.x1
        
        self.b = interp(nnew)
        self.b[-1] = self.b[-2]+(self.b[-2]-self.b[-3])
        
        self.c = 0.5 * (self.b[1:] + self.b[:-1])
        self.d = np.diff(self.b)
        
        self._interp = None

"""
The function mkgrid(), by usage of a dictionary of classes in the module,
generates the desired mesh

**kwargs accepts named arguments
*args accepts non-key worded arguments
kind -> string specifying the type of mesh wanted
"""          
def mkgrid(kind, *args, **kwargs):
    
    GRID_CLASSES = {'linear': LinearGrid,
                    'lin': LinearGrid,
                    'quadratic': QuadraticGrid,
                    'quad': QuadraticGrid,
                    'geometric': GeometricGrid,
                    'geo': GeometricGrid,
                    'logarithmic': LogGrid,
                    'log': LogGrid}
    
    klass = GRID_CLASSES[kind]
    
    return klass(*args, **kwargs)
        

        
    