
"""
This file provides a way to define a compensated simple cubic potential

The potential is characterized by: 

* local potential bottom
* local lattice depth along x,y,z
* interaction strength

From these two quantities one can calculate

* local band structure
* local tunneling rate, t
* onsite interactions, U 

With band structure,  tunneling and onsite in hand one can proceed to do the 
local density approximation. 
"""

import numpy as np
import udipole
import scipy.constants as C 


import ldaconf
# Load the interpolation data for band structure
basedir = ldaconf.basedir
 
v0 = np.loadtxt(basedir + 'banddat/interpdat_B1D_v0.dat')
NBands = 3
from scipy.interpolate import interp1d
interp0 = []
interp1 = []
for n in range( NBands ):
    interp0.append( interp1d(v0, \
               np.loadtxt(basedir + 'banddat/interpdat_B1D_0_%d.dat'%n) ))
    interp1.append( interp1d(v0, \
               np.loadtxt(basedir + 'banddat/interpdat_B1D_1_%d.dat'%n) ))
    

def bands3dvec( s0, NBand=0 ):
    """ 
    Using the interpolation data this function calculates the bottom and top
    of a band in a vectoreized way.   

    Parameters 
    ----------
    s0          :  array-like.  Has to have a len = 3, where each of the three
                   elements along the first dimension corresponds to the 
                   sX, sY, sZ lattice depths respectively. 

    Returns 
    -------
    bands       :  array of len = 2, which contains the band bottom and band 
                   top.

    Notes 
    ----  

    Examples
    --------
    """
    assert len(s0)==3
    bandbot = np.zeros_like( s0[0] ) 
    bandtop = np.zeros_like( s0[0] ) 
    if NBand == 0:
        nband = [0, 0, 0]
    elif NBand == 1:
        s0.sort(axis=0)
        nband = [1, 0, 0]
    else:
        return None
    for i in range(3):
        in1d = nband[i] 
        if in1d%2 ==0:
            bandbot += interp0[in1d](s0[i]) 
            bandtop += interp1[in1d](s0[i])
        else:
            bandbot += interp1[in1d](s0[i])
            bandtop += interp0[in1d](s0[i])
    return np.array((bandbot,bandtop))


#Here the interpolation data for the on-site interactions is loaded from disk
from scipy.interpolate import interp1d
wFInterp = interp1d( np.loadtxt(basedir + 'banddat/interpdat_wF_v0.dat'), \
                     np.loadtxt(basedir + 'banddat/interpdat_wF_wF.dat'))

# Using the interpolation data calculate a function that will get the on-site
# interactions in a vectorized way. 

def onsite( v0,  **kwargs ):
    """ 
    Using the interpolation data this function calculates the onsite
    interactions in a vectoreized way.   

    Parameters 
    ----------
    s0          :  array-like.  Has to have a len = 3, where each of the three
                   elements along the first dimension corresponds to the 
                   sX, sY, sZ lattice depths respectively.

    as          :  float.  scattering length.  here we set it to 1. so that this
                   code does not need to know the value of the scattering length
                   in the experiment.  The resuling U can be scaled up by the
                   scattering length outside of this function.   

    wavelength  :  float. the lattice wavelength.

    Returns 
    -------
    U           :  on-site interactions 

    Notes 
    ----  

    Examples
    --------
    """
    a_s = kwargs.get( 'as', 1. ) 
    wavelen = kwargs.get( 'wavelength', 1.064) 

    assert len(v0)==3
    wint = np.ones_like( v0[0] ) 
    for i in range(3):
        wint *= wFInterp( v0[i] )
    # The lattice spacing is given in um
    a0a = 5.29e-11 / (wavelen/2. *1e-6)
    return a_s * a0a * np.power(wint, 1./3.)




class sc( udipole.potential) :
    """
    This class defines a compensated simple cubic lattice potential.  The 
    initialization parameters are given as keyword arguments.   

    The purpose of this class is to calculate all of the local quantities 
    related ONLY to the potential, for use in the local density approximation. 

    The quantities that depend ONLY on the potential are:
  
    - envelope of lattice potential
    - lattice depths (x, y ,z ) 
    - band structure 

 
    """ 
    def __init__(self, **kwargs):
        # Initialize lattice part 
        axes= [ (np.pi/2,0.), (np.pi/2, np.pi/2), (0,0) ] 
        self.l  = kwargs.get('wavelength', 1.064)
        self.m  = kwargs.get('mass', 6.)
        self.w  = kwargs.get('waists', ((47.,47.), (47.,47.), (47.,47.)) )
        self.r  = kwargs.get('retro', (1.,1.,1.) )

        # This alpha is the retro factor, do not confuse with the 
        # waist ratio 
        self.alpha  = kwargs.get('alpha', (1.,1.,1.) )
 
        self.scale = kwargs.get('scale', 10.)
        self.Er0 = udipole.Erecoil(self.l, self.m)
        
        if 'allIR' in kwargs.keys():
            self.s0 = [kwargs.get('allIR', 7.0 )]*3
        else:
            self.s0 = kwargs.get('s0', (7.0, 7.0, 7.0) )

        if 'allIRw' in kwargs.keys():
            wIR = kwargs.get('allIRw', 47.) 
            self.w = ((wIR,wIR),(wIR,wIR),(wIR,wIR)) 
                 
        lattbeams = [ udipole.LatticeBeam( axis=axes[i], s0=self.s0[i], \
                                   wavelength=self.l, scale=self.scale,\
                                   waists=self.w[i], retro=self.r[i], \
                                   alpha=self.alpha[i] ) \
                      for i in range(3) ] 
        
        udipole.potential.__init__(self, lattbeams, units=('$E_{R}$', 1./self.Er0) )
        
        # Initialize compensation part  
        self.GRw  = kwargs.get('green_waists', ((40.,40.), (40.,40.), (40.,40.)) ) 
        if 'allGR' in kwargs.keys():
            self.g0 = [kwargs.get('allGR', 4.0 )]*3
        else:
            self.g0 = kwargs.get('green_Er', (4.0, 4.0, 4.0) )
 
        if 'allGRw' in kwargs.keys():
            wGR = kwargs.get('allGRw', 40.) 
            self.GRw = ((wGR,wGR),(wGR,wGR),(wGR,wGR)) 

        self.GRl  = kwargs.get('green_wavelength', 0.532)
        
        # Express the power requiered for each GR beam, given the compensation
        # value in units of the lattice recoil, and given the GR beam waists
        GRmW = [ 1000.* self.g0[i]  \
                      * self.Er0/np.abs(udipole.uL(self.GRl)*2/np.pi) \
                      * self.GRw[i][0]*self.GRw[i][1]  for i in range(3) ]
        
        self.greenbeams = [ udipole.GaussBeam( axis=axes[i], mW=GRmW[i], \
                                       waists=self.GRw[i], wavelength=self.GRl)\
                            for i in range(3) ]

    def EffAlpha(self):
        """
        Returns a latex string with the information for the effective
        alpha.  The effective alpha is the ratio of average IR waist to
        average green waist. 
        """
        effAlpha = np.mean(sum(self.w,())) / np.mean( sum(self.GRw,())) 
        return r'$\alpha_{w}=%.2f$'%effAlpha

    def LatticeWaist(self):
        """
        Returns the mean lattice waist of the sample
        """  
        return np.mean(sum(self.w,())) 

    def TrapFreqsInfo( self ):
        """
        Returns a latex string with the information about the trap 
        frequencies.  
        """
        freqs  = self.TrapFreqs()
        geomean = (freqs[0]*freqs[1]*freqs[2])**(1./3.)
        if np.isnan(geomean):
            return r'$\bar{\nu} = \mathrm{nan}$'
        try:
            return r'$\bar{\nu} = %d\,\mathrm{Hz}$'%geomean
        except:
            print geomean
            print type(geomean) 
            raise


    def TrapFreqs( self ): 
        """
        This function calculates the effective harmonic trapping frequencies
        of the potential.  It fits the bottom of the lowest band to a second
        degree polynomial. 
        """

        # Fit the first +/- 10 um of the band bottom to a second order
        # polynomial 
        direc100 = (np.pi/2., 0.) 
        direc010 = (np.pi/2., np.pi/2.) 
        direc001 = (0., np.pi) 
        nu = []
        for d in [direc100, direc010, direc001]: 
            # Make a cut line along d
            td, Xd, Yd, Zd, limsd = \
                udipole.linecut_points( direc=d, extents = 4.)

            # Obtain band structure along the d direction
            bandbot_d, bandtop_d,  Ezero_d, tunneling_d = \
                self.bandStructure( Xd, Yd, Zd, \
                    getonsite=False)

            # Fit with poly
            z = np.polyfit( td, bandbot_d , 2 )
            c2 = z[0]  

            # A factor equal to h/(m*lambda)  comes out in front 
            # here we use lengths in um and freqs in Hz.  
            factor =  C.h  / C.physical_constants['atomic mass constant'][0] \
                      * 1e12  \
                     / self.m  / self.l  

            nu.append( (1./2./np.pi) * np.sqrt(c2) * factor )
        return nu 


    def Info( self ):
        """
        Returns a latex string with the information that defines the 
        compensated simple cubic potential
        """
        # Lattice

        def beamlabels( V0, w , TypeV, Typew):
            if len(np.unique(V0))==1:
                Vlabel = '$%s=%.2fE_{R}$' % (TypeV, V0[0] )
            else:
                if 's' in TypeV:
                    Vlabel = '$s_{x}=%.2f, s_{y}=%.1f, s_{z}=%.1f$' % \
                          (V0[0],V0[1],V0[2] ) 
                else:
                    Vlabel = '$g_{x}=%.2f, g_{y}=%.1f, g_{z}=%.1f$' % \
                          (V0[0],V0[1],V0[2] ) 

             
            waists = sum( w, () ) 
            if len( np.unique( waists )) == 1: 
                wlabel = '$w_{%s}=%d\,\mu\mathrm{m}$' % (Typew, w[0][0] )
            else:
                coords = ['x','y','z']
                wlabel = ''
                for i,wp in enumerate(w):
                    wlabel += '$w_{%s%s}=(%d,%d)\,\mu\mathrm{m}%' % \
                               (Typew, coord[i], wp[0], wp[1] ) 
                    if i < 2 : wlabel += '$\mathrm{,}\ $'  

            return Vlabel + '$\mathrm{,}\ $' + wlabel
 
        Llabel = beamlabels( self.s0, self.w, 's_{0}', 'L')  
        Glabel = beamlabels( self.g0, self.GRw, 'g_{0}', 'C')

        return Llabel, Glabel 



    def Bottom( self, X, Y, Z):
        """ 
        Returns the envelope of the lattice potential.  Units depend on
        unitfactor. 

        Parameters 
        ----------
        X, Y, Z     :  can be floats or array-like. The potential is calculated
                       in a vectorized way.  X, Y, Z all need to have the same 
                       shape. 

        Returns 
        -------
        envelope of lattice potential in microKelvin.  Has the same shape as the 
        X, Y, Z. 

        Notes 
        ----  

        Examples
        --------
        """
        EVAL = np.zeros_like(X)
        for b in self.beams:
            EVAL += b.getBottom( X, Y, Z)
        for g in self.greenbeams:
            EVAL += g(X,Y,Z)
        return EVAL*self.unitfactor

    
    def S0( self, X, Y, Z):
        """ 
        Returns the local lattice depths.  There is a latttice depth along each
        lattice direction.  Units depend on unitfactor. 

        Parameters 
        ----------
        X, Y, Z     :  can be floats or array-like. The potential is calculated
                       in a vectorized way.  X, Y, Z all need to have the same 
                       shape. 

        Returns 
        -------
        lattice depths, which is a numpy array with shape = (3, shape(X) ) 
        the extradimensions with 3 elements is used for s0x, s0y, s0z.  

        Notes 
        ----  

        Examples
        --------
        """
        EVAL = []
        for b in self.beams:
            EVAL.append( b.getS0( X, Y, Z)*self.unitfactor )
        return np.array(EVAL)


    def LatticeMod( self, X, Y, Z):
        """ 
        Returns values that can be plotted to represent the local lattice 
        modulation.  This function is mainly for plotting purposes to help 
        visualize the potential.  

        Parameters 
        ----------
        X, Y, Z     :  can be floats or array-like. The potential is calculated
                       in a vectorized way.  X, Y, Z all need to have the same 
                       shape. 

        Returns 
        -------
        latticemod, which is an array with the same shape as X, Y, Z 

        Notes 
        ----  

        Examples
        --------
        """
        V0s = self.S0( X, Y, Z )
        Mod = np.amin(V0s, axis=0)
        return self.Bottom(X,Y,Z) + \
               Mod * np.power( np.cos( 2.*np.pi*np.sqrt(X**2 + Y**2 + Z**2 ) \
                                        / self.l / self.scale    ), 2)


    def bandStructure( self, X, Y, Z, **kwargs ):
        """ 
        This function calculates and returns the relevant quantities for the
        lowest band.   See the Returns section below for a list of these 
        quantities.  

        Parameters 
        ----------
        X, Y, Z     :  can be floats or array-like. The potential is calculated
                       in a vectorized way.  X, Y, Z all need to have the same 
                       shape. 

        Returns 
        -------
        bandbot     :  bottom of the lowest band 

        bandtop     :  top of the lowest band 

        Ezero       :  exactly half-way on the lowest band  
 
        tunneling   :  tunneling rate

        evapth      :  evaporation threshold 

        Notes 
        ----  

        Examples
        --------
        """
        self.bands = bands3dvec( self.S0( X,Y,Z), NBand=0) 
        
        self.Ezero     =  ( self.bands[1] + self.bands[0] )/2. \
                          + self.Bottom( X,Y,Z) 

        # Notice that this is an effective tunnling because in general there
        # will be a tunneling rate associated with each lattice direction
        # Along the lattice diagonal this is correct. 
        self.tunneling =  ( self.bands[1] - self.bands[0] )/12.

        getonsite = kwargs.get('getonsite',True) 
        if getonsite:
            self.Ut  = onsite( self.S0( X,Y,Z), wavelength = self.l ) \
                       / self.tunneling 
            return self.bands[0]+self.Bottom(X,Y,Z),\
                   self.bands[1]+self.Bottom(X,Y,Z),\
                   self.Ezero, self.tunneling, \
                   self.Ut
        else:
            return self.bands[0]+self.Bottom(X,Y,Z), \
                   self.bands[1]+self.Bottom(X,Y,Z), \
                   self.Ezero, self.tunneling

         
    def firstExcited( self, X, Y, Z ):
        """ 
        This function calculates and returns the bottom and top of the first
        excited band. 

        Parameters 
        ----------
        X, Y, Z     :  can be floats or array-like. The potential is calculated
                       in a vectorized way.  X, Y, Z all need to have the same 
                       shape. 

        Returns 
        -------
        bandbot     :  bottom of the first excited band 

        bandtop     :  top of the first excited  band 

        Notes 
        ----  

        Examples
        --------
        """
   
        bands1 = bands3dvec( self.S0( X,Y,Z), NBand=1) 
        return bands1[0]+self.Bottom(X,Y,Z), \
               bands1[1]+self.Bottom(X,Y,Z) 



def get_max_comp( pot, aS, T, verbose ): 
    """ 
    This function gets the maximum amount of compensation that will
    be tolerated by a setup.   

    Parameters 
    ----------
    pot    : a potential of class sc 
    aS     : scattering length
    T      : temperature in Er

    Returns 
    -------
    g0     :  maximum amount of compensation tolerated 

    """
    g0 = np.abs( pot.greenbeams[0](0.,0.,0.) * pot.unitfactor )
    s0 = np.abs( pot.beams[0](0.,0.,0.) * pot.unitfactor )

    
    wL = sum( pot.w, () ) 
    if len( np.unique( wL )) != 1: 
        raise ValueError('non-isotropic sample')   
    wL  = pot.w[0][0]

    wG = sum( pot.GRw, () ) 
    if len( np.unique( wG )) != 1: 
        raise ValueError('non-isotropic sample')  
    wG  = pot.GRw[0][0]

    # First find the depth of the band at the origin
    bandOrigin = -3.*s0 + ( bands3dvec( np.array([s0,s0,s0]) )[0] + 
                   bands3dvec( np.array([s0,s0,s0]) )[1] )/2.
    band100 =  bandOrigin / 3.
    if verbose: 
        print "bandOrigin = %.2f"% bandOrigin 
        print "band100 = %.2f" % band100
    
    U  = onsite( np.array([s0,s0,s0]) ) * aS
    if verbose: 
        print U/4.

    if wL <= wG:   # alpha < 1
        # 
        # bandOrigin + 3*g0 + U/2.  =  band100 + g0  
        #
        g0 = (band100 - bandOrigin - U/2. - T*1.2)/2.
        print "g0 = %.2f" % g0


