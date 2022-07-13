from bolos import grid, parser, solver 
import numpy as np
import pandas as pd
from math import floor, log10
import matplotlib.pyplot as plt

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array-value)).argmin()
    return idx

def exponent_min(number):
    base10 = log10(number)
    return floor(base10)

gr = grid.LinearGrid(0,5000,100)
boltzmann = solver.BoltzmannSolver(gr)

with open('helium_.dat') as fp:
    processes = parser.parse(fp)
    
boltzmann.load_collisions(processes)
boltzmann.set_density('He', 1)

boltzmann.kT = 300*solver.KB/solver.ELECTRONVOLT

EN_grid = np.linspace(1,1000,100)

n = 0
lproc = []

for target, proc in boltzmann.iteration_all():
    n+=1
    lproc.append(str(proc))
    
out = np.zeros(shape=(len(EN_grid),n+4))
EEDF = np.zeros(shape=(len(gr.b)-1,20))
hEEDF = []

header = []
header.append('R#')
header.append('E/N(Td)')
header.append('Energy(eV)')

for reaction in lproc:
    header.append(reaction.replace(" ",""))
    
header.append('mu')

ie = 0
fi = 1

for ii, EN_ in enumerate(EN_grid):
    print(ii)
    gr = grid.LinearGrid(0,5000,100)
    boltzmann = solver.BoltzmannSolver(gr)
    
    with open('helium_.dat') as fp:
        processes = parser.parse(fp)
        
    boltzmann.load_collisions(processes)
    boltzmann.set_density('He', 1)

    boltzmann.kT = 300*solver.KB/solver.ELECTRONVOLT
    
    boltzmann.EN = EN_*solver.TOWNSEND
    boltzmann.init()
    fMaxwell = boltzmann.maxwell(2)
    
    f = boltzmann.converge(fMaxwell, maxn = 2000, rtol = 1*10**-4)
    
    x = 0
    
    while(True):
        lgrid = max(boltzmann.benergy)*0.80
        idp = find_nearest(boltzmann.benergy, lgrid)
        
        if f[idp-1] < 1*10**-13:
            if x == 0:
                oldgrid = gr
                x = 1
                
            newgrid = grid.QuadraticGrid(0, lgrid-lgrid*0.25, 100)
             
            boltzmann.grid = newgrid
            boltzmann.init()
            
            finterp = boltzmann.grid.interpolate(f, oldgrid)
            
            f = boltzmann.converge(finterp, maxn = 2000, rtol = 1e-04)
            oldgrid = newgrid
             
        else:
            if x == 0:
                oldgrid = gr
                x = 1
                
            newgridl = grid.QuadraticGrid(0, max(boltzmann.benergy)+0.25, 100)
            
            boltzmann.grid = newgridl
            boltzmann.init()
            
            finterp = boltzmann.grid.interpolate(f, oldgrid)
            f = boltzmann.converge(finterp, maxn = 2000, rtol = 1e-04)
            oldgrid = newgridl
            
        if (min(f)/max(f) >= 0.1*10**-10 and min(f)/max(f) <= 0.1*10**-9):
            break
        
    if (ii+1)%10 == 0:
        for ih, (ee,ff) in enumerate(zip(boltzmann.cenergy,f)):
            EEDF[ih,ie] = ee
            EEDF[ih,fi] = ff
            
        ie += 2
        hEEDF.append('energy_'+str(floor(EN_)))
        fi += 2
        hEEDF.append('EEDF_'+str(floor(EN_)))
        
    for idx, (target, proc) in enumerate(boltzmann.iter_everything()):
        out[ii,0] = ii+1
        out[ii,1] = EN_
        out[ii,2] = boltzmann.mean_energy(f)
        out[ii,idx+3] = boltzmann.rate(f, proc)
        out[ii,n+3] = boltzmann.mobility(f)
        
    
        
df = pd.DataFrame(out)
df.columns = header
df.to_csv(r'C:\Users\MillaJo\OneDrive - Lam Research\Documents\BOLSIG+-data-comparison\pyBolsig\data.dat', sep='\t',index=False)

EDF = pd.DataFrame(EEDF)
EDF.columns = hEEDF
EDF.to_csv(r'C:\Users\MillaJo\OneDrive - Lam Research\Documents\BOLSIG+-data-comparison\pyBolsig\dataEEDF.dat', sep='\t',index=False)



