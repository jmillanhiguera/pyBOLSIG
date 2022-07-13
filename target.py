from collections import defaultdict
import logging
import sys
import os

import numpy as np

file_dir = os.path.dirname(__file__)                                                                   
sys.path.append(file_dir)   

from process import Process, NullProcess

"""
The Class Target() defines the target specie of the system  

name -> from process, it gets the name of the target species

-------------------------------------------------------------------------------
variables needed to initialize the system are

name -> name of the target variable
density -> density in comparison to other background gases. Is a ratio between 
    0 and 1

"""
class Target(object):
    def __init__(self, name):
        self.name = name
        self.mass_ratio = None
        self.density = 0.0
        
        self.elastic            =   []
        self.effective          =   []
        self.attachment         =   []
        self.ionization         =   []
        self.excitation         =   []
        self.weighted_elastic   =   []
        
        self.kind = {   'ELASTIC': self.elastic,
                        'EFFECTIVE': self.effective,
                        'MOMENTUM': self.effective,
                        'ATTACHMENT': self.attachment,
                        'IONIZATION': self.ionization,
                        'EXCITATION': self.excitation,
                        'WEIGHTED_ELASTIC': self.weighted_elastic
                    }
        
        self.by_product = defaultdict(list)           

        logging.debug("Target %s created." % str(self))

    """
    The function add_process() is responsible for defining the target in the 
    system and also generate a data structure between target and product of the
    processes involved

    process -> used to define the type of target in the system, and to 
        associate the target and product
    """ 
    def add_process(self, process):

        # appends to the designated process, i.e, if its elastic, it will go
        # ahead and append to the variables in class Target
        kind = self.kind[process.kind]
        kind.append(process)                                                                                                                    

        # only applies for elastic and effective collisions 
        if process.mass_ratio is not None:                                                                                                     
            
            # verification process 
            if (self.mass_ratio is not None and self.mass_ratio != process.mass_ratio):                                                         
               raise ValueError("More than one mass ratio for target '%s'" % self.name)
            
            self.mass_ratio = process.mass_ratio

        # assigns to the process class itselt, ie, the target class            
        process.target = self                                                                                                                   
        
        # appends a key to the product in question 
        self.by_product[process.product].append(process)
        
    """
    The function ensure_elastic() makes sure a elastic component of the 
    reaction set exist, and if an effective cross-sectional area exist, 
    converts it effectively to an equivalent elastic term 

    process -> used to define the type of target in the system, and to 
        associate the target and product
    """
    def ensure_elastic(self):
        # the elastic and effective variables cannot be full at the same time                                                                                                                   
        if self.elastic and self.effective:                                                                                                     
            raise ValueError("In target '%s': EFFECTIVE/MOMENTUM and ELASTIC cross-sections are incompatible." % self)
            
        # passes 
        if self.elastic:                                                                                                                        
            return
        
        # only one effective when calculating the EEDF 
        if len(self.effective) > 1:                                                                                                             
            raise ValueError("In target '%s': Can't handle more than 1 EFFECTIVE/MOMENTUM for a given target" % self)
            
        # neither was encountered
        if not self.effective:                                                                                                                  
            logging.warning("Target %s has no ELASTIC or EFFECTIVE cross sections" % str(self))
            return 
        
        newdata = self.effective[0].data.copy()                                                                                                

        # substract cross sectional due to inelastic collisions if effective 
        # cross sectional area is used
        for p in self.inelastic:                                                                                                               
            newdata[:, 1] -= p.interp(newdata[:,0])                                                                                                                                                     
        
        # if cross sectional area in some instances of the data is less than 
        # zero, it automatically shorts to 0
        if np.amin(newdata[:,1]) < 0:                                                                                                           
            logging.warning("After substracting INELASTIC from effective, target %s has negative cross section." % self.name)
            logging.warning("Setting as max(0, ...)")                                                                                           
            newdata[:,1] = np.where(newdata[:,1]>0, newdata[:,1], 0)                                                                            
            
        # effective cross sectional area converted to elastic for calculation
        # purporses
        newelastic = Process(target=self.name, 
                             kind='ELASTIC', 
                             data=newdata, 
                             mass_ratio=self.effective[0].mass_ratio,
                             comment="Calculated from EFFECTIVE cross sections")                                                                
        
        
        logging.debug("EFFECTIVE -> ELASTIC for target %s" % str(self))
        
        # new process is added to target
        self.add_process(newelastic)                                                                                                            
        
        self.effective=[]                                                                                                                       
            
    """
    the function inelastic() is a getter for all the inelastic processes in the 
    system
    """
    
    # CHANGE TO CODE
    # @property                                                                                                                                   
    # def inelastic(self):
    #     return (self.excitation)
    
    # @property
    # def ioniztion(self):
    #     return (self.ionization)

    # # CHANGE TO CODE
    # @property                                                                                                                                   
    # def attachmnt(self):
    #     return (self.attachment)
    
    """
    the function everything() is a getter for all the processes in the system
    """
    @property
    def inelastic(self):
        """ An useful abbreviation. """
        return (self.attachment + self.ionization + self.excitation)

    @property
    def everything(self):
        return (self.attachment + self.elastic + self.excitation + self.ionization)
    
    """
    special function that returns a string representation of an object. This
    special function could be used as the default for most instances. There
    is no need to define the __str__ function for most users 
    """
    def __repr__(self):
        return "Target(%s)" % repr(self.name)

    def __str__(self):
        return self.name