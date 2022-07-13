import re
import numpy as np
import logging                                                                     

# re.compile - compile a regular expression pattern, returning a re.Pattern
# object. Efficient when the same regex used several times in a single program
RE_SEP = re.compile("-----+")                                                                   
RE_ARROW = re.compile("<?->")                                                      

"""
The function parse() parses a BOLSIG+ cross-section file 

fp -> file pointer pointing to a BOLSIG+ compatible cross file

note: as it executes the supporting functions, the lines in fp will move
further in the code. Hence, the for loop does in parse() will not execute all 
the lines as the supporting functions does that
"""  
def parse(fp):
    processes = []                                                                
    for line in fp:
        try:
            # checks for key, and then generates a pointer to the function 
            # desired at fread 
            key = line.strip()                                                      
            fread = KEYWORDS[key]

            # d is essentially the return of any of the _read_* function, which
            # returns a dictionary     
            d=fread(fp)

            # adding new element to dictionary d
            d['kind'] = key
            processes.append(d)                                                     
        except KeyError:                                                            
            pass                                                                    
        
    logging.info("Parsing complete. %d processes read." % len(processes))
    return processes


"""
The function _read_until_sep() reads the block of code last until the pattern 
corresponding to RE_SEP is found

fp -> file pointer pointing to a BOLSIG+ compatible cross file
""" 
def _read_until_sep(fp):
    lines = []
    for line in fp:
        if RE_SEP.match(line.strip()):                                              
            break                                                                   
        
        lines.append(line.strip())                                                   
        
    return lines 

"""
The function _read_block() reads the block of code corresponding to the type
of reaction process.

fp -> file pointer pointing to a BOLSIG+ compatible cross file
has_arg -> if the reaction rate is elastic, effective, excitation or ionization
    it would be considered for the creation of the dictionary
""" 
def _read_block(fp, has_arg=True):
    target = next(fp).strip()                                                       
    if has_arg:  
        # arg -> for elastic, effective, excitation or ionization                                                                    
        arg = next(fp).strip()                                                      
    else:
        # arg -> attachment processes
        arg = None
    
    #\n specifies to the list to become a block of text using newline. it 
    # starts at a line n+1 where defining arg left
    comment = "\n".join(_read_until_sep(fp))                                        
    
    logging.debug("Read process '%s'" % target)
    data = np.loadtxt(_read_until_sep(fp)).tolist()                                 
    
    return target, arg, comment, data                                               

"""
The function _read_momentum() reads the block of code corresponding to the
ELASTIC or MOMENTUM key.

fp -> file pointer pointing to a BOLSIG+ compatible cross file
""" 
def _read_momentum(fp):
    target, arg, comment, data = _read_block(fp, has_arg=True)
    mass_ratio = float(arg.split()[0])
    d = dict(target = target, mass_ratio=mass_ratio,comment=comment,data=data)
    
    return d 

"""
The function _read_excitation() reads the block of code corresponding to the
EXCITATION or IONIZATION key.

fp -> file pointer pointing to a BOLSIG+ compatible cross file
"""
def _read_excitation(fp):         
    target, arg, comment, data = _read_block(fp, has_arg=True)
    
    # left and right hand side gets splitted by regex expression
    lhs, rhs = [s.strip() for s in RE_ARROW.split(target)]
    d = dict(target=lhs, product=rhs, comment=comment, data=data)

    if '<->' in target.split():                                                     
        threshold, weight_ratio = float(arg.split()[0]), float(arg.split()[1])      
        d['weight_ratio'] = weight_ratio    
    else:                                                                           
        threshold = float(arg.split()[0])                                           
        
    d['threshold'] = threshold                
    return d                                                                     

"""
The function _read_attachment() reads the block of code corresponding to the
ATTACHMENT key.

fp -> file pointer pointing to a BOLSIG+ compatible cross file
"""    
def _read_attachment(fp):
    target, arg, comment, data = _read_block(fp, has_arg=False)
    
    d = dict(comment=comment,data=data,threshold=0.0)
    lr = [s.strip() for s in RE_ARROW.split(target)]

    if len(lr) == 2:
        d['target'] = lr[0]
        d['product'] = lr[1]
    else:
        d['target'] = target
        
    return d

KEYWORDS = {"MOMENTUM": _read_momentum, 
            "ELASTIC": _read_momentum, 
            "EFFECTIVE": _read_momentum,
            "EXCITATION": _read_excitation,
            "IONIZATION": _read_excitation,
            "ATTACHMENT": _read_attachment}
    
    
        
    