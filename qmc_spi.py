
"""
This file provides a way to calculate the bulk spin structure factor
from a given density distribution. 

The idea is to calculate the n(r) using NLCE then pass it over to 
QMC to get Spi
"""

import numpy as np

import matplotlib.pyplot as plt
import matplotlib
from matplotlib import rc
rc('font', **{'family':'serif'})
rc('text', usetex=True)

from scipy.spatial import Delaunay
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator
from scipy.interpolate.interpnd import _ndim_coords_from_arrays

import qmc

from scipy import integrate

import logging
# create logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler()) 

def integrate_sphere( r, qty):
    q = qty[ ~np.isnan(qty)]
    r = r[ ~np.isnan(qty) ]
    a = 1.064/2.
    return np.power( a,-3) * 2*np.pi * integrate.simps( q*(r**2), r)
    # Notice that the integral above is 2pi rather than 4pi because the
    # radial quantity q is defined for negative and positive radii, so 
    # the radial integral has an implicit factor of 2. 


def spi_bulk( r111, n111, mu111, T, t111, U111, **kwargs ): 
    """
    This function is used to calculate  the bulk
    spin structure factor using the QMC data
    """  

    # The max U and min t are used 
    U = U111.max()
    t = t111.min() 
    logger.info( "Calculating Spi_Bulk for " + \
          "U={:0.2f}, T={:0.2f}".format(U/t, T/t) ) 
    logger.info( "Central chemical potential = " + str(mu111.max()) ) 

    inhomog = kwargs.get('inhomog', False)
    spiextents = kwargs.get('spiextents', 100.) 
    entextents = kwargs.get('entextents', 100.)

    logger.info( "spiextents = {:0.2f},  entextents = {:0.2f}".\
           format(spiextents, entextents)  ) 

    #subset = np.where( np.abs(r111) < spiextents )[0] 

    spi = np.ones_like( mu111 ) 
    entropy = np.zeros_like( mu111 ) 
    density = np.zeros_like( mu111 ) 

    do_kappa =  kwargs.get('do_kappa', False)
    if do_kappa:
        compr = np.zeros_like(mu111) 
   

    posr111 = np.abs(r111) 

    # Variable to find the radius at which entropy cracks down
    entfail = True
 
    for i in range(len(r111)):
        mu = mu111[i] 
        # The corrrect thing to do here is 
        if inhomog == True:
            Uval = U111[i]/t111[i] 
            Tval = T/t111[i] 
        # But since we do not have enough QMC data yet 
        # we are sticking with a single U and T value
        else:
            Uval = U/t 
            Tval = T/t 


        title_text = r'$(U/t)_{{0}}={:0.2f}$,\ '.format(Uval) \
            + '$(T/t)_{{0}}={:0.2f}$,\ $\mu={:.2f}$,\ $r={:.2f}$'.\
                       format(Tval,mu,r111[i]) + '\n'
        

        if posr111[i] <= spiextents:
            # Find the Spi
            result = qmc.find_closest_qmc( U=Uval, T=Tval, mu=mu, \
                         title_text = title_text, radius=r111[i] )
            if result is None:
                print "Had problems finding Spi for " +  \
                  " U={:02d}, T={:0.2f}".format(int(Uval), Tval)
                continue
            spi[ i ] =  result

            # Find the kappa
            if do_kappa:
                result = qmc.find_closest_qmc( U=Uval, T=Tval, mu=mu, \
                             title_text = title_text, radius=r111[i], QTY='kappa',
                              error_nan=True)
                if result is None:
                    print "Had problems finding kappa for " +  \
                      " U={:02d}, T={:0.2f}".format(int(Uval), Tval)
                    continue
                compr[ i ] =  result



            # Find the density 
            result = qmc.find_closest_qmc( U=Uval, T=Tval, mu=mu, \
                                  title_text = title_text, \
                         QTY='density', radius=r111[i], error_nan=True)
            if result == 'out-of-bounds':
                density[ i ] = np.nan
                continue 
            elif result is None:
                print "Had problems finding Density for " +  \
                  " U={:0.2f}, T={:0.2f}".format(Uval, Tval)
                continue
            density[ i ] =  result

        else:
            density[ i ] = np.nan 

   
        if posr111[i] <= entextents:
            # Find the entropy 
            result = qmc.find_closest_qmc( U=Uval, T=Tval, mu=mu, \
                         title_text = title_text, radius=r111[i],\
                         QTY='entropy', error_nan=True)
            if result is None:
                print "Had problems finding Entropy for " +  \
                  " U={:0.2f}, T={:0.2f}".format(Uval, Tval)
                continue

            # Change True/False to use entropy extrapolation
            elif r111[i] >=0. and result is np.nan and True:

                warn =  'r={:.1f}, U={:0.2f}, T={:0.3f}, mu={:0.3f}'.\
                      format(r111[i],Uval, Tval, mu) + '  ==> s = nan'
                logger.warning( warn ) 
 
                printv = False

                Tabove = Tval + np.linspace(0.02, 0.2, 6)
                sabove = [] 
                for Tab in Tabove:
                    try:
                        sab = qmc.find_closest_qmc( U=Uval, T=Tab, mu=mu, \
                            title_text = title_text, radius=r111[i],\
                            QTY='entropy', error_nan=True)
                        sabove.append(sab) 
                    except:
                        sabove.append(np.nan) 
                extrapdat =  np.column_stack(( Tabove, sabove))
                if printv:
                    print extrapdat 
                valid =  ~np.isnan( extrapdat[:,1] )  
                if  np.sum( valid ) > 2:
                    x = extrapdat[:,0][ valid ] 
                    y = extrapdat[:,1][ valid ]
                    z = np.polyfit( x,y, 1) 
                    p = np.poly1d(z) 
                    extrap = max( float(p(Tval)), 0.0 )    
                    if printv:
                        print "Extrapolated = {:0.2f}".format( extrap )  
                    result =  extrap 
                    if not printv:
                        print 'r={:.1f}, U={:0.2f}, T={:0.3f}, mu={:0.3f}'.\
                              format(r111[i],Uval, Tval, mu),
                        print '  ==> s = {:0.2f}'.format(result)
                    if False:
                        fig = plt.figure( figsize=(3.5,3.5))
                        gs = matplotlib.gridspec.GridSpec( 1,1 ,\
                                left=0.18, right=0.95, bottom=0.13, top=0.88)
                        ax = fig.add_subplot( gs[0] )
                        ax.grid(alpha=0.5)
                        ax.plot( Tabove, sabove, 'o' ) 
                        ax.set_xlabel('$T/t$')
                        ax.set_ylabel('$s\ \mathrm{entropy}$')
                        plt.show()
                    
                else: 
                    logger.warning( "Error extrapolating entropy to lower T, "\
                                     + " all nans" )  
             
                
            entropy[ i ] =  result 

            if r111[i] > 20. and mu < 0. and result > 0.3:
                logger.info( '==== ALERT HIGH ENTROPY ====' )
                logger.info( 'r={:.1f}, U={:0.2f}, T={:0.3f}, mu={:0.3f}'.\
                      format(r111[i],Uval, Tval, mu) + \
                         '  ==> s={:0.2f}'.format( float(result ) ) )

            if result is not np.nan and entfail == True:
                entfail = False
                msg1 =  'r={:.1f}, U={:0.2f}, T={:0.3f}, mu={:0.3f}'.\
                      format(r111[i],Uval, Tval, mu),
             
                msg2 =  '  ==> s={:0.2f}'.format( float(result ) )
                logger.info(  '== First entropy valid radius ==' ) 
                logger.info(  msg1 ) 
                logger.info(  msg2 ) 


            
        else: 
            entropy[ i ] = np.nan
 

    number = integrate_sphere( r111, n111 ) 
    spibulk =  integrate_sphere( r111, spi * n111) / number 
    overall_entropy = integrate_sphere( r111, entropy ) / number

    lda_number = integrate_sphere( r111, density ) 
    
    if do_kappa:
        return spibulk, spi, overall_entropy, entropy, lda_number, density,\
                compr
    else:
        return spibulk, spi, overall_entropy, entropy, lda_number, density



