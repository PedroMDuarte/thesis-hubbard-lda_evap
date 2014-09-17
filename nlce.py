
"""
This file provides a way to obtain thermodynamic quantities from an 
interpolation of available NLCE solutions 
"""
import ldaconf
import numpy as np
import glob

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
rc('font', **{'family':'serif'})
rc('text', usetex=True)

from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

import logging
# create logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())
#logger.disabled = True

from qmc import get_qty_mu

def find_closest_nlce( U=8, T=0.67, mu=4., qty='dens', **kwargs):
    """
    This function finds the closest values of U and T in the NLCE data 
    that straddle the values U and T given as arguments.
    """

    msg0 = "U={:0.2f}, T={:0.2f}".format( U, T) 
   
    nUs = 3 
    us = [ float(u.split('/U')[-1]) for u in \
             glob.glob( ldaconf.basedir + 'NLCE8_FinalK/U*' ) ] 
    du = [ np.abs( U-u ) for u in us ] 
    index = np.argsort( du ) 
    Ulist0 = range( nUs ) 
    Upts = [ us[index[i]] for i in Ulist0 ]  
 
    
    # The T points are not uniformly spaced so we find the two closest ones
    # We start with a list of available T points: 
    Ts = np.array( sorted( [ float(g.split('/T')[1].split('.dat')[0]) for g in \
             glob.glob(ldaconf.basedir + 'NLCE8_FinalK/U00/T*') ] ) )
    
    
    diff = T-Ts

    error = False

    if np.all( diff < 0 ):
        print "Available temperatures do not make it this low:"
        print " T = ", T
        error = True  

    if not error:    
        order_pos =  np.argsort(np.abs( diff[diff>0] ))
        order_neg =  np.argsort(np.abs( diff[diff<0] ))
        Tpts = sorted( [  Ts[diff>0][ order_pos[0] ] ,  Ts[diff<0][ order_neg[0]] ] )  
    else:
        order = np.argsort( np.abs( diff ) ) 
        Tpts = sorted( [ Ts[order[0]], Ts[order[1]], Ts[order[2]] ] ) 
        #Ta = min(Ts[order[0]], Ts[order[1]])
        #Tb = max(Ts[order[0]], Ts[order[1]])
        #print "T in ", Ta, Tb
    
    datadir = ldaconf.basedir
    datfiles = []
    for Uval in Upts:
        for Tval in Tpts:
            fname =  datadir + \
                'NLCE8_FinalK/U{:02d}/T{:0.2f}.dat'.format(int(Uval),Tval)
            datfiles.append([ fname, Uval, Tval ])
            
    if qty == 'dens':
        COL = 1 
    elif qty == 'entr':
        COL = 2 
    elif qty == 'spi':
        COL = 3 
    elif qty == 'kappa':
        COL = 4 
    else:
        raise "Qty not defined:", qty
            
    MUCOL = 0 
    basedat = [] 

    qtyinterp = kwargs.get( 'qtyinterp', 'nearest' )

    for f in datfiles:
        msg = 'U={:0.2f}, T={:0.2f}'.format(U,T) + \
           ' mu={:0.2f}, Upt={:0.3f}, Tpt={:0.3f}'.\
           format(mu, f[1], f[2])

        try:
            dat = np.loadtxt(f[0])
            # careful = False, for NLCE data it is ok to not worry about
            # chemical potentials that are outside the data range
            qtyresult = get_qty_mu( dat, mu, MUCOL, COL, msg=msg, careful=False) 
            if qtyresult == 'out-of-bounds':
                print msg
                print "out-of-bounds"
                continue
 
            basedat.append( [f[1], f[2], qtyresult] )
                
        except Exception as e:
            print "Failed to get data from file = ", f

            print msg

            fig = plt.figure( figsize=(3.5,3.5))
            gs = matplotlib.gridspec.GridSpec( 1,1 ,\
                    left=0.15, right=0.96, bottom=0.12, top=0.88)
            ax = fig.add_subplot( gs[0] )
            ax.grid(alpha=0.5)
            ax.plot( dat[:,MUCOL], dat[:,COL], '.-')
            ax.axvline( mu )

            ax.text( 0.5, 1.05, msg, ha='center', va='bottom', \
                transform=ax.transAxes)
            if matplotlib.get_backend() == 'agg':
                fig.savefig('err_mu.png', dpi=200) 
            else:
                plt.show()

            raise e 
        
    basedat =   np.array(basedat)
    #print "Closest dat = ", basedat
    points = _ndim_coords_from_arrays(( basedat[:,0] , basedat[:,1]))
    
    
    
    #finterp = CloughTocher2DInterpolator(points, basedat[:,2])
    finterp = LinearNDInterpolator( points, basedat[:,2] )
    
    
    try:
        result = finterp( U,T )
        if np.isnan(result):
            if U >= 30 and U<=32.5:
                result = finterp( 29.99, T ) 
                logger.warning(" nlce: U={:0.1f} replaced to U=29.99 ".format(U) )
        if np.isnan(result):
            raise Exception("!!!! nlce: Invalid result !!!!\n" + msg0)
        
    except Exception as e:
        print e
        error = True
        
    if error or kwargs.get('showinterp',False):
        #print "Interp points:"
        #print basedat
        
        tri = Delaunay(points)
        fig = plt.figure( figsize=(3.5,3.5))
        gs = matplotlib.gridspec.GridSpec( 1,1 ,\
                left=0.15, right=0.96, bottom=0.12, top=0.88)
        ax = fig.add_subplot( gs[0] )
        ax.grid(alpha=0.5)
        ax.triplot(points[:,0], points[:,1], tri.simplices.copy())
        ax.plot(points[:,0], points[:,1], 'o')
        ax.plot( U, T, 'o', ms=6., color='red')
        xlim = ax.get_xlim()
        dx = (xlim[1]-xlim[0])/10.
        ax.set_xlim( xlim[0]-dx, xlim[1]+dx )
        ylim = ax.get_ylim()
        dy = (ylim[1]-ylim[0])/10.
        ax.set_ylim( ylim[0]-dy, ylim[1]+dy )
        ax.set_xlabel('$U/t$')
        ax.set_ylabel('$T/t$',rotation=0,labelpad=8)
        
        tt = kwargs.get('title_text','')
        ax.set_title( tt + '$U/t={:.2f}$'.format(U) + ',\ \ ' + '$T/t={:.2f}$'.format(T), \
                ha='center', va='bottom', fontsize=10)
        save_err = kwargs.get('save_err',None) 
        if save_err is not None:
            print "saving png to ", save_err 
            fig.savefig( save_err, dpi=300)

        if matplotlib.get_backend() == 'agg':
            print "saving png to err.png"
            fig.savefig('err.png', dpi=200) 
        else:
            plt.show()

        raise
    
    return result


QTYINTERP = 'linear' 

def nlce_dens( T, t, mu, U, ignoreLowT=False, verbose=True):
    U_ = U/t 
    T_ = T/t 
    mu_ = mu/t   

    result = np.empty_like(mu) 
    for i in range( len(mu_)):
        result[i] = find_closest_nlce( U=U_[i], T=T_[i], mu=mu_[i], \
                     qty='dens',  qtyinterp=QTYINTERP  ) 
    return result    
 
def nlce_entr( T, t, mu, U, ignoreLowT=False, verbose=True):
    U_ = U/t 
    T_ = T/t 
    mu_ = mu/t   

    result = np.empty_like(mu) 
    for i in range( len(mu_)):
        result[i] = find_closest_nlce( U=U_[i], T=T_[i], mu=mu_[i], \
                     qty='entr', qtyinterp=QTYINTERP ) 
    return result     

def nlce_spi( T, t, mu, U, ignoreLowT=False, verbose=True):
    U_ = U/t 
    T_ = T/t 
    mu_ = mu/t   

    result = np.empty_like(mu) 
    for i in range( len(mu_)):
        result[i] = find_closest_nlce( U=U_[i], T=T_[i], mu=mu_[i], \
                     qty='spi', qtyinterp=QTYINTERP ) 
    return result     
 
def nlce_cmpr( T, t, mu, U, ignoreLowT=False, verbose=True):
    U_ = U/t 
    T_ = T/t 
    mu_ = mu/t   

    result = np.empty_like(mu) 
    for i in range( len(mu_)):
        result[i] = find_closest_nlce( U=U_[i], T=T_[i], mu=mu_[i], \
                     qty='kappa',  qtyinterp=QTYINTERP  ) 
    return result     

