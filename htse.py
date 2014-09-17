
"""
This file provides the HTSE solution to the Hubbard model.  It gives only
the density, double occupancy and entropy
"""

import numpy as np
import numpy.ma as ma


minTt = 1.1
 
 
def check_valid( T, t, mu, U, ignoreLowT, verbose):
    """
    This function is used to enforce the validity limits of the HTSE. 

    Near half filling we have that T/t should be > 1.5, but away from 
    half filling this does not matter so much. 

    When running the LDA in our small beam waist setup the local value 
    of t increases as one moves away from the center (the lattice depth 
    gets shallower).   For fixed T this results in a very low T/t at 
    the edge.  

    Since the density is n<<1 at the edge then the temperature requirement
    is not so important there.    

    What this function does is it gets the local T/t profile.  Since the
    lattice depth decreases monotinically from the center,  T/t does also.  

    Usually the minimum T/t  along the profile is compared to minTt 
    (defined below)  to check if the HTSE is valid.   

    What this function does is it applies a function to the T/t profile
    such that it is unchanged  wherever the density is >=1  and it is 
    boosted up at low densities such that it has a better chance to be 
    higher than minTt
    
    """
    minTt = 1.2

    boostDelta = 0.9
    boost = ma.asarray( boostDelta - \
                        boostDelta*np.exp( -np.abs( ( mu-U/2.)/(3.*U))) ) 
    boost.mask = mu>U/2.
    #print boost
    boost =  boost.filled ( 0. ) 

    Tt = T/t 
    Tt_ =  Tt + boost

    # Make sure all quantities are arrays with the same dimension
    if type(Tt) is float or len(np.atleast_1d(Tt))==1 :
        Tt_arr = np.ones_like(Tt_)*Tt
    else: 
        Tt_arr = Tt 
    if type(mu) is float or len(np.atleast_1d(mu))==1 :
        mu_arr = np.ones_like(Tt_)*mu
    else:
        mu_arr = mu 
    if type(U) is float or len(np.atleast_1d(U))==1 :
        U_arr = np.ones_like(Tt_)*U
    else:
        U_arr = U 
   
    np.savetxt('errlog', np.column_stack(( \
                            Tt_arr, \
                            boost, \
                            Tt_ ,\
                            np.ones_like(Tt_)*minTt, \
                            mu_arr,\
                            U_arr) ) ) 

    nerror = np.sum(  Tt_ < minTt )
    if nerror > 0 :
        msg = "HTSE ERROR: T/t < %.2f =>  min(T/t) = %.2f, max(T/t) = %.2f"% \
                (minTt, Tt_.min(), Tt_.max())
        msg = msg + '\nmu0 = %.2f'%mu.max()
        
        if verbose:
            print msg 
        if not ignoreLowT:
            raise ValueError(msg)
  

    

def htse_dens( T, t, mu, U, ignoreLowT=False, verbose=True ):
    check_valid( T, t, mu, U, ignoreLowT,verbose) 

    z0  = 2.*np.exp(mu/T)  + np.exp(-U/T + 2.*mu/T)  + 1. 
    
    term1 = ( 2.*np.exp(mu/T) + 2.*np.exp(-U/T + 2.*mu/T) ) / z0 

    term2 = 6.*((t/T)**2.)*(-4.*np.exp(mu/T) - 4.*np.exp(-U/T + 2.*mu/T) ) * \
                           ( 2.*T*(1-np.exp(-U/t))*np.exp(2.*mu/T) / U \
                             + np.exp( mu/T ) + np.exp( -U/T + 3.*mu/T)) \
                      / z0**3.

    term3 = 6.*((t/T)**2.)*(4.*T*(1-np.exp(-U/T))*np.exp(2.*mu/T)/U \
                              + np.exp(mu/T) + 3.*np.exp( -U/T + 3.*mu/T) ) \
                      / z0**2.
        
    #print z0    
    #print term1
    #print term2
    #print term3 
    return  term1 + term2 + term3

def htse_doub( T, t, mu, U, ignoreLowT=False, verbose=True):
    check_valid( T, t, mu, U, ignoreLowT,verbose) 
 
    z0  = 2.*np.exp(mu/T)  + np.exp(-U/T + 2.*mu/T)  + 1. 
    
    term1 = np.exp(-U/T + 2.*mu/T) / z0
 
    term2 = -6.*((t/T)**2.)*( -2.*(T**2.)*(1-np.exp(-U/T))\
                                  *np.exp(2.*mu/T) / (U**2.) \
                              + 2.*np.exp(-U/T+2.*mu/T)*T/U  \
                              - np.exp(-U/T+3.*mu/T) ) \
                 / z0**2. 

    term3 = -12.*((t/T)**2.)*( 2.*T*(1-np.exp(-U/T))*np.exp(2.*mu/T) / U \
                               + np.exp(mu/T) + np.exp(-U/T+3.*mu/T) ) \
                             *np.exp(-U/T+2.*mu/T) \
                 / z0**3. 
            
    #print z0    
    #print term1
    #print term2
    #print term3 
    return  term1 + term2 + term3


def htse_entr( T, t, mu, U, ignoreLowT=False, verbose=True ):
    check_valid( T, t, mu, U, ignoreLowT,verbose) 
 
    z0  = 2.*np.exp(mu/T)  + np.exp(-U/T + 2.*mu/T)  + 1. 
    
    term0 = (  U*np.exp(-U/T+2.*mu/T) - 2.*mu*np.exp(mu/T) \
              - 2.*mu*np.exp(-U/T+2.*mu/T) ) / T / z0

    term1 = np.log( z0 ) 

    term2 = 6.*((t/T)**2.)*( -2.*U*np.exp(-U/T+2.*mu/T)/T \
                                 + 4.*mu*np.exp(mu/T)/T  \
                                 + 4.*mu*np.exp(-U/T+2.*mu/T)/T ) \
               * ( 2.*T*(1-np.exp(-U/T))*np.exp(2.*mu/T)/U  \
                   + np.exp(mu/T) + np.exp(-U/T+3.*mu/T) ) \
                         / z0**3. 

    term3 = 6.*((t/T)**2.)*( 2.*(1-np.exp(-U/T))*np.exp(2.*mu/T)*T/U \
                            -2.*np.exp(-U/T+2.*mu/T) \
                            -4.*mu*(1-np.exp(-U/T))*np.exp(2.*mu/T)/U \
                            +U*np.exp(-U/T+3.*mu/T)/T \
                            -mu*np.exp(mu/T)/T \
                            -3.*mu*np.exp(-U/T+3.*mu/T)/T ) \
                         / z0**2. 

    term4 = -6.*((t/T)**2.)*( 2.*T*(1-np.exp(-U/T))*np.exp(2.*mu/T)/U \
                             + np.exp(mu/T) + np.exp(-U/T+3.*mu/T) ) \
                         / z0**2. 
    
    #print z0   
    #print term0
    #print term1
    #print term2
    #print term3
    #print term4
    return  term0 + term1 + term2 + term3 + term4

#def htse_cmpr( T, t, mu, U, ignoreLowT=False, verbose=True ):
#    dmu = 0.001
#    n1 = htse_dens( T, t, mu+dmu, U, ignoreLowT=ignoreLowT, verbose=verbose)
#    n0 = htse_dens( T, t, mu-dmu, U, ignoreLowT=ignoreLowT, verbose=verbose)
#
#    dn =  n1**(2./3.) - n0**(2./3.) 
#    return dn/(2.*dmu)

def htse_cmpr( T, t, mu, U, ignoreLowT=False, verbose=True ):
    check_valid( T, t, mu, U, ignoreLowT,verbose)
 
    term0 =  2.*np.exp(mu/T) + 1. + np.exp(-U/T)*np.exp(2.*mu/T)
    
    term1 = (t/T) * (2*np.exp(mu/T) + 4.*np.exp(-U/T)*np.exp(2.*mu/T)) / term0 
    
    term2 = (t/T) * (-2.*np.exp(mu/T) - 2.*np.exp(-U/T)*np.exp(2.*mu/T) ) * \
                    ( 2.*np.exp(mu/T) + 2.*np.exp(-U/T)*np.exp(2.*mu/T) ) / \
                     (term0**2.) 
    
    term3 = ((t/T)**3.) * 6.0 * \
            (-4.*np.exp(mu/T) - 8.*np.exp(-U/T)*np.exp(2.*mu/T) ) * \
            ( 2.*(T/U)*(1. - np.exp(-U/T))*np.exp(2.*mu/T) + np.exp(mu/T) + \
                np.exp(-U/T)*np.exp(3.*mu/T)) / (term0**3.)
    
    term4 = ((t/T)**3.) *  6.0 * \
            (-6.*np.exp(mu/T) - 6.*np.exp(-U/T)*np.exp(2.*mu/T)) * \
            (-4.*np.exp(mu/T) - 4.*np.exp(-U/T)*np.exp(2.*mu/T)) * \
            ( 2.*(T/U)*(1 - np.exp(-U/T))*np.exp(2.*mu/T) + \
                 np.exp(mu/T) + np.exp(-U/T)*np.exp(3.*mu/T)) / ( term0**4.)
    
    term5 = ((t/T)**3.) * 12.0 * \
            (-4.*np.exp(mu/T) - 4.*np.exp(-U/T)*np.exp(2.*mu/T)) * \
            ( 4.*(T/U)*(1. - np.exp(-U/T))*np.exp(2.*mu/T) + np.exp(mu/T) + \
                  3.*np.exp(-U/T)*np.exp(3.*mu/T)) /  ( term0**3.)
    
    term6 = ((t/T)**3.) * 6.0 * \
            ( 8.*(T/U)*(1. - np.exp(-U/T))*np.exp(2.*mu/T) + \
                np.exp(mu/T) + 9.*np.exp(-U/T)*np.exp(3.*mu/T) ) / ( term0**2.) 
    
    dn_dmu = term1 + term2 + term3 + term4 + term5 + term6

    # Density : 
 
    z0  = 2.*np.exp(mu/T)  + np.exp(-U/T + 2.*mu/T)  + 1. 
    
    n_term1 = ( 2.*np.exp(mu/T) + 2.*np.exp(-U/T + 2.*mu/T) ) / z0 

    n_term2 = 6.*((t/T)**2.)*(-4.*np.exp(mu/T) - 4.*np.exp(-U/T + 2.*mu/T) ) * \
                           ( 2.*T*(1-np.exp(-U/t))*np.exp(2.*mu/T) / U \
                             + np.exp( mu/T ) + np.exp( -U/T + 3.*mu/T)) \
                      / z0**3.

    n_term3 = 6.*((t/T)**2.)*(4.*T*(1-np.exp(-U/T))*np.exp(2.*mu/T)/U \
                              + np.exp(mu/T) + 3.*np.exp( -U/T + 3.*mu/T) ) \
                      / z0**2.
        
    n = n_term1 + n_term2 + n_term3

    return (2./3.) /  n**(1./3.)   * dn_dmu  * t


def htse_cmpb( T, t, mu, U, ignoreLowT=False, verbose=True ):
    check_valid( T, t, mu, U, ignoreLowT,verbose)
 
    term0 =  2.*np.exp(mu/T) + 1. + np.exp(-U/T)*np.exp(2.*mu/T)
    
    term1 = (t/T) * (2*np.exp(mu/T) + 4.*np.exp(-U/T)*np.exp(2.*mu/T)) / term0 
    
    term2 = (t/T) * (-2.*np.exp(mu/T) - 2.*np.exp(-U/T)*np.exp(2.*mu/T) ) * \
                    ( 2.*np.exp(mu/T) + 2.*np.exp(-U/T)*np.exp(2.*mu/T) ) / \
                     (term0**2.) 
    
    term3 = ((t/T)**3.) * 6.0 * \
            (-4.*np.exp(mu/T) - 8.*np.exp(-U/T)*np.exp(2.*mu/T) ) * \
            ( 2.*(T/U)*(1. - np.exp(-U/T))*np.exp(2.*mu/T) + np.exp(mu/T) + \
                np.exp(-U/T)*np.exp(3.*mu/T)) / (term0**3.)
    
    term4 = ((t/T)**3.) *  6.0 * \
            (-6.*np.exp(mu/T) - 6.*np.exp(-U/T)*np.exp(2.*mu/T)) * \
            (-4.*np.exp(mu/T) - 4.*np.exp(-U/T)*np.exp(2.*mu/T)) * \
            ( 2.*(T/U)*(1 - np.exp(-U/T))*np.exp(2.*mu/T) + \
                 np.exp(mu/T) + np.exp(-U/T)*np.exp(3.*mu/T)) / ( term0**4.)
    
    term5 = ((t/T)**3.) * 12.0 * \
            (-4.*np.exp(mu/T) - 4.*np.exp(-U/T)*np.exp(2.*mu/T)) * \
            ( 4.*(T/U)*(1. - np.exp(-U/T))*np.exp(2.*mu/T) + np.exp(mu/T) + \
                  3.*np.exp(-U/T)*np.exp(3.*mu/T)) /  ( term0**3.)
    
    term6 = ((t/T)**3.) * 6.0 * \
            ( 8.*(T/U)*(1. - np.exp(-U/T))*np.exp(2.*mu/T) + \
                np.exp(mu/T) + 9.*np.exp(-U/T)*np.exp(3.*mu/T) ) / ( term0**2.) 
    
    dn_dmu = term1 + term2 + term3 + term4 + term5 + term6

    # Density : 
 
    z0  = 2.*np.exp(mu/T)  + np.exp(-U/T + 2.*mu/T)  + 1. 
    
    n_term1 = ( 2.*np.exp(mu/T) + 2.*np.exp(-U/T + 2.*mu/T) ) / z0 

    n_term2 = 6.*((t/T)**2.)*(-4.*np.exp(mu/T) - 4.*np.exp(-U/T + 2.*mu/T) ) * \
                           ( 2.*T*(1-np.exp(-U/t))*np.exp(2.*mu/T) / U \
                             + np.exp( mu/T ) + np.exp( -U/T + 3.*mu/T)) \
                      / z0**3.

    n_term3 = 6.*((t/T)**2.)*(4.*T*(1-np.exp(-U/T))*np.exp(2.*mu/T)/U \
                              + np.exp(mu/T) + 3.*np.exp( -U/T + 3.*mu/T) ) \
                      / z0**2.
        
    n = n_term1 + n_term2 + n_term3

    return 1. /  (n**2.)   * dn_dmu  * t
   

if __name__ == "__main__":
    print htse_dens( 2.4, 1., 10., 20.)
    print htse_doub( 2.4, 1., 10., 20.)
    print htse_entr( 2.4, 1., 10., 20.)
