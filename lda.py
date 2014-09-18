
import logging
# create logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)


from vec3 import vec3, cross
import scipy.constants as C 


"""
This file provides a way of calculating trap profiles in the local density
approximation.    It needs to have a way of calculating:
 
* local band structure
* local tunneling rate, t
* local onsite interactions, U

From these thre quantities it can go ahead an use the solutions to the
homogeneous Fermi-Hubbard (FH)  model to calculate the LDA. 

In the homogeenous FH problem the chemical potential and the zero of
energy are always specified with respect to some point in the local band
structure.  This point depends on how the Hamiltonian is written down:  

A.  Traditional hamiltonian.   

  i, j  :  lattice sites 
  <i,j> :  nearest neighbors 
  s     :  spin 
  su    :  spin-up
  sd    :  spin-down


      Kinetic energy = -t \sum_{s} \sum_{<i,j>} a_{i,s}^{\dagger} a_{j,s}   

       Onsite energy =  U \sum_{i}  n_{i,su} n_{i,sd}   

  Using the traditional hamiltonian half-filling occurrs at a chemical 
  potential  mu = U/2.  

  The zero of energy in the traditional hamiltonian is exactly midway through
  the lowest band of the U=0 hamiltonian.


B.  Half-filling hamiltonian

      Kinetic energy = -t \sum_{s} \sum_{<i,j>} a_{i,s}^{\dagger} a_{j,s}   

       Onsite energy =  U \sum_{i} ( n_{i,su} - 1/2 )( n_{i,sd} - 1/2 )  

  Using the half-filling hamiltonian half-filling occurrs at a chemical
  potential  mu = 0,  a convenient value.   

  The zero of energy in the half-filling hamiltonian is shifted by U/2  
  with respect to the zero in the traditional hamiltonian. 

....
Considerations for LDA
....


When doing the local density approximation (LDA) we will essentially have a
homogenous FH model that is shifted in energy by the enveloping potential of
the trap and by the local band structure.  In the LDA the zero of energy  is
defined as the energy of an atom at a point where there are no external
potentials.   A global chemical potential will be defined with respect to the
LDA zero of energy.  

To calculate the local thermodynamic quantities, such as density, entropy,
double occupancy, etc.  we will use theoretical results for a homogeneous FH
model.  The local chemical potential will be determined based on the local
value of the enveloping potential and the local band structure (which can be
obtained from the local lattice depth).   

""" 

import udipole
import scubic
from mpl_toolkits.mplot3d import axes3d

from scipy import integrate
from scipy import optimize
from scipy.interpolate import interp1d


# Load up the HTSE solutions 
from htse import htse_dens, htse_doub, htse_entr, htse_cmpr
from nlce import nlce_dens, nlce_entr, nlce_spi, nlce_cmpr

import qmc, qmc_spi 



def get_dens( T, t, mu, U, select='htse', ignoreLowT=False, verbose=True):
    """ This function packages all three methods for obtaining
    the thermodynamic quantities: htse, nlce, qmc"""
    if select == 'htse':
        return htse_dens( T, t, mu, U, ignoreLowT=ignoreLowT, verbose=verbose)
    elif select == 'nlce':
        return nlce_dens( T, t, mu, U, ignoreLowT=ignoreLowT, verbose=verbose)
    

def get_entr( T, t, mu, U, select='htse', ignoreLowT=False, verbose=True):
    """ This function packages all three methods for obtaining
    the thermodynamic quantities: htse, nlce, qmc"""
    if select == 'htse':
        return htse_entr( T, t, mu, U, ignoreLowT=ignoreLowT, verbose=verbose)
    elif select == 'nlce':
        return nlce_entr( T, t, mu, U, ignoreLowT=ignoreLowT, verbose=verbose)

def get_spi( T, t, mu, U, select='htse', ignoreLowT=False, verbose=True):
    """ This function packages all three methods for obtaining
    the thermodynamic quantities: htse, nlce, qmc"""
    if select == 'htse':
        return np.ones_like( t ) 
    elif select == 'nlce':
        return nlce_spi( T, t, mu, U, ignoreLowT=ignoreLowT, verbose=verbose)
 

def get_doub( T, t, mu, U, select='htse', ignoreLowT=False, verbose=True):
    """ This function packages all three methods for obtaining
    the thermodynamic quantities: htse, nlce, qmc"""
    if select == 'htse':
        return htse_doub( T, t, mu, U, ignoreLowT=ignoreLowT, verbose=verbose)
    else:
        raise "doublons not defined" 

def get_cmpr( T, t, mu, U, select='htse', ignoreLowT=False, verbose=True):
    """ This function packages all three methods for obtaining
    the thermodynamic quantities: htse, nlce, qmc"""
    if select == 'htse':
        return htse_cmpr( T, t, mu, U, ignoreLowT=ignoreLowT, verbose=verbose)
    elif select == 'nlce':
        return nlce_cmpr( T, t, mu, U, ignoreLowT=ignoreLowT, verbose=verbose)
    


#...............
# LDA CLASS 
#...............


class lda:
    """ 
    This class provides the machinery to do the lda.  It provides a way to 
    determine the global chemical potential for a given number or for a half
    filled sample.  
    """ 
 
    def __init__( self, **kwargs ): 
        self.verbose = kwargs.get('verbose', False)
  
        # Flag to ignore errors related to the slope of the density profile
        # or the slope of the band bottom 
        self.ignoreSlopeErrors = kwargs.get( 'ignoreSlopeErrors',False)

        # Flag to ignore errors related to the global chemical potential
        # spilling into the beams 
        self.ignoreMuThreshold = kwargs.get('ignoreMuThreshold', False )

        # Flag to ignore errors related to low temperatures beyond the reach
        # of the htse  
        self.ignoreLowT = kwargs.get('ignoreLowT',False)

        # Flag to ignore errors related to a non-vanishing density 
        # distribution within the extents 
        self.ignoreExtents = kwargs.get('ignoreExtents',False)

        

        # The potential needs to offer a way of calculating the local band 
        # band structure via provided functions.  The following functions
        # and variables must exist:
        # 
        #  To calculate lda: 
        #  -  pot.l 
        #  -  pot.bandStructure( X,Y,Z )
        #
        #  To make plots  
        #  -  pot.unitlabel
        #  -  pot.Bottom( X,Y,Z )
        #  -  pot.LatticeMod( X,Y,Z )
        #  -  pot.Info() 
        #  -  pot.EffAlpha()
        #  -  pot.firstExcited( X,Y,Z )
        #  -  pot.S0( X,Y,Z )
 
        self.pot = kwargs.pop( 'potential', None) 
        if self.pot is None: 
            raise ValueError(\
                    'A potential needs to be defined to carry out the LDA')  
        # The potential also contains the lattice wavelength, which defines
        # the lattice spacing 
        self.a = self.pot.l / 2. 
 


        # Initialize temperature.  Temperature is specified in units of 
        # Er.  For a 7 Er lattice  t = 0.04 Er 
        self.T = kwargs.get('Temperature', 0.40 ) 
        # Initialize interactions.
        self.a_s = kwargs.get('a_s',300.)

        # Initialize extents
        self.extents = kwargs.pop('extents', 40.) 

        # Initialize the type of Hubbard solution
        # type can be: 'htse', 'nlce', 'qmc'
        self.select = kwargs.get('select','htse') 
 
        # Make a cut line along 111 to calculate integrals of the
        # thermodynamic quantities

        # set the number of points to use in the cut 
        if self.select == 'htse':
            NPOINTS = 320 
        else:
            NPOINTS = 80
	OVERRIDE_NPOINTS = kwargs.pop('override_npoints', None)
        if OVERRIDE_NPOINTS is not None:
            NPOINTS = OVERRIDE_NPOINTS 

        direc111 = (np.arctan(np.sqrt(2)), np.pi/4)
        unit = vec3(); th = direc111[0]; ph = direc111[1] 
        unit.set_spherical( 1., th, ph); 
        t111, self.X111, self.Y111, self.Z111, lims = \
            udipole.linecut_points( direc=direc111, extents=self.extents,\
            npoints=NPOINTS)
        # Below we get the signed distance from the origin
        self.r111 =  self.X111*unit[0] + self.Y111*unit[1] + self.Z111*unit[2]

 
        # Obtain band structure and interactions along the 111 direction
        bandbot_111, bandtop_111,  \
        self.Ezero_111, self.tunneling_111, self.onsite_t_111 = \
            self.pot.bandStructure( self.X111, self.Y111, self.Z111)

        # The onsite interactions are scaled up by the scattering length
        self.onsite_t_111 = self.a_s * self.onsite_t_111
        self.onsite_111 = self.onsite_t_111 * self.tunneling_111

        # Lowst value of E0 is obtained 
        self.LowestE0 = np.amin( bandbot_111 )  

        self.Ezero0_111 = self.Ezero_111.min()

        #---------------------
        # CHECK FOR NO BUMP IN BAND BOTTOM 
        #---------------------
        # Calculate first derivative of the band bottom at small radii, to 
        # assess whether or not the potential is a valid potential 
        # (no bum in the center due to compensation )
        positive_r =  np.logical_and( self.r111  > 0. ,  self.r111 < 10. ) 
        # absolute energy of the lowest band, elb
        elb = bandbot_111[ positive_r ]  
        elb_slope = np.diff( elb ) < -1e-4
        n_elb_slope = np.sum( elb_slope )
        if n_elb_slope > 0:
            msg = "ERROR: Bottom of the band has a negative slope"
            if self.verbose:
                print msg
                print elb
                print np.diff(elb) 
                print elb_slope
            if not self.ignoreSlopeErrors:  
                raise ValueError(msg) 
        else: 
            if self.verbose:
                print "OK: Bottom of the band has positive slope up to "\
                       + "r111 = 10 um"

        #------------------------------
        # SET GLOBAL CHEMICAL POTENTIAL 
        #------------------------------
        # Initialize global chemical potential and atom number
        # globalMu can be given directly or can be specified via the 
        # number of atoms.  If the Natoms is specified we calculate 
        # the required gMu using this function: 
        muHalfMott = self.onsite_111.max()/2. 
        if 'globalMu' in kwargs.keys(): 
            # globalMu is given in Er, and is measured from the value
            # of Ezero at the center of the potential
            # When using it in the phase diagram it has to be changed to
            # units of the tunneling
            self.globalMu = kwargs.get('globalMu', 0.15)
            if  self.globalMu == 'halfMott':
                self.globalMu = muHalfMott  \
                                + kwargs.get('halfMottPlus',0.)
        else :
            self.Number = kwargs.get('Natoms', 3e5)
            fN = lambda x : self.getNumber( muHalfMott + x,self.T, \
                                     verbose=False)- self.Number
            if self.verbose :
                print "Searching for globalMu => N=%.0f, "% self.Number,
            

            muBrent = kwargs.get('muBrent', (-0.2, 0.3)) # Maybe the default
                                                      # muBrent range should
                                                      # be U dependent
            muBrentShift = kwargs.get('muBrentShift', 0. ) 
            muBrent = ( muBrent[0] + muBrentShift * muHalfMott, \
                        muBrent[1] + muBrentShift * muHalfMott )

            try:
                muBrentOpt, brentResults = \
                    optimize.brentq(fN, muBrent[0], muBrent[1], \
                                xtol=2e-3, rtol=1e-2, full_output=True)
                #print "fN(muBrentOpt) = ", fN(muBrentOpt)
                self.globalMu =  muHalfMott + muBrentOpt  
            except Exception as e:
                errstr  = 'f(a) and f(b) must have different signs'
                if errstr in e.message:
                    print "Natoms = {:.4g}".format(self.Number)
                    print "mu0 = %.2f -->  Nlda = %.2g" % \
                              (muBrent[0], fN(muBrent[0]) + self.Number )
                    print "mu1 = %.2f -->  Nlda = %.2g" % \
                              (muBrent[1], fN(muBrent[1]) + self.Number )
                raise
 
            if self.verbose:
                print "gMu=%.3f, " % brentResults.root,
                print "n_iter=%d, " % brentResults.iterations,
                print "n eval=%d, " % brentResults.function_calls,
                print "converge?=", brentResults.converged

        #---------------------
        # EVAPORATION ENERGIES
        #---------------------
        # Calculate energies to estimate eta parameter for evaporation
        self.globalMuZ0 = self.Ezero0_111 + self.globalMu

        # Make a cut line along 100 to calculate the threshold for evaporation
        direc100 = (np.pi/2, 0.) 
        t100, self.X100, self.Y100, self.Z100, lims = \
            udipole.linecut_points( direc=direc100, extents = 1200.)

        # Obtain band structure along the 100 direction
        bandbot_100, bandtop_100,  self.Ezero_100, self.tunneling_100 = \
            self.pot.bandStructure( self.X100, self.Y100, self.Z100, \
                getonsite=False)
        self.Ezero0_100 = self.Ezero_100.min()

        # evapTH0_100 accounts for situations in which there is a local barrier 
        # as you move along 100 to the edge 
        self.evapTH0_100 = bandbot_100.max()

        # Once past the local barrier we calculate the bandbot energy along 
        # a beam
        self.beamBOT_100 = bandbot_100[-1]

        if self.verbose:
            #This obtains the value of g0, careful when using anisotropic params
            scubic.get_max_comp( self.pot, 650., self.T, verbose=False)  


        #------------------------------------------------
        # CONTROL THE CHEMICAL POTENTIAL SO THAT IT STAYS 
        # BELOW THE THRESHOLD FOR EVAPORATION
        #------------------------------------------------
        # For a valid scenario we need 
        #   self.globalMuZ0 < self.beamBOT_100
        #   self.globalMuZ0 < self.evapTH0_100  
        # Otherwise the density distribution will spill out into the beams
        # and the assumption of spherical symmetry won't be valid.
        if self.globalMuZ0 + self.T*1.2 > self.evapTH0_100:
            msg = "ERROR: Chemical potential exceeds the evaporation threshold "
            if self.verbose:
                print msg
                print " mu0 = %.3f" % self.globalMuZ0
                print "   T = %.3f" % (self.T*1.2)
                print " Eth = %.3f" % self.evapTH0_100  
            if not self.ignoreMuThreshold : 
                raise ValueError(msg) 
        elif self.verbose:
            print "OK: Chemical potential is below evaporation threshold."

        if self.globalMuZ0 + self.T*1.2 > self.beamBOT_100:
            msg = "ERROR: Chemical potential exceeds the bottom of the band " +\
                  "along 100"
            if self.verbose:
                print msg
                print " mu0 = %.3f" % self.globalMuZ0
                print "   T = %.3f" % (self.T*1.2)
                print "E100 = %.3f" % self.beamBOT_100  
            if not self.ignoreMuThreshold : 
                raise ValueError(msg) 
        elif self.verbose:
            print "OK: Chemical potential is below the bottom of the band " +\
                  "along 100"


        #-----------------------
        # ESTIMATION OF ETA EVAP
        #-----------------------
        mu = self.globalMuZ0 - self.LowestE0 
        th = self.evapTH0_100 - self.LowestE0
        self.EtaEvap = th/mu
        self.DeltaEvap = th - mu
        if False: 
            print "mu global = %.3g" % self.globalMuZ0 
            print "evap th   = %.3g" % self.evapTH0_100
            print "lowest E  = %.3g" % self.LowestE0
            print "mu = %.3g" % mu
            print "th = %.3g" % th
            print "eta = %.3g" % (th/mu)
            print "th-mu = %.3g" % (th-mu)

     
    
        
        # After the chemical potential is established the local chemical
        # potential along 111 can be defined.  It is used to calculate other
        # thermodynamic quantities
        gMuZero = self.Ezero0_111 + self.globalMu
        self.localMu_t_111= (gMuZero - self.Ezero_111) / self.tunneling_111
        self.localMu_111= (gMuZero - self.Ezero_111) 

        localMu = gMuZero - self.Ezero_111


        # If the global chemical potential is fixed then the lda
        # class can have an easier time calculating the necessary
        # temperature to obtain a certain entropy per particle.
        # This option is provided here: 
        if ( 'globalMu' in kwargs.keys() and 'SN' in kwargs.keys() ) \
             or kwargs.get('forceSN',False): 
            self.SN = kwargs.get('SN', 2.0)
           
            # Shut down density extent errors during the search
            igExt = self.ignoreExtents 
            self.ignoreExtents = True
            
 
            fSN = lambda x : self.getEntropy(x) / \
                             self.getNumber(self.globalMu, x  ) \
                                - self.SN
            if self.verbose: 
                print "Searching for T => S/N=%.2f, "% self.SN
            TBrent = kwargs.get('TBrent',(0.14,1.8))

            try:
                Tres, TbrentResults = \
                    optimize.brentq(fSN, TBrent[0], TBrent[1], \
                                xtol=2e-3, rtol=2e-3, full_output=True) 
                if self.verbose:
                    print "Brent T result = %.2f Er" % Tres                
                self.T = Tres 
            except Exception as e:
                errstr  = 'f(a) and f(b) must have different signs'
                if errstr in e.message:
                    print "T0 = %.3f -->  fSN = %.3f" % \
                              (TBrent[0], fSN(TBrent[0]) )
                    print "T1 = %.3f -->  fSN = %.3f" % \
                              (TBrent[1], fSN(TBrent[1]) )
                raise
                
                print "Search for S/N=%.2f did not converge"%self.SN
                print "Temperature will be set at T = %.2f Er" % self.T
                print "ERROR:"
                print e.message
                print self.pot.Info()
                print 

            self.ignoreExtents = igExt


        # Once the temperature is established we can calculate the ratio
        # of temperature to chemical potential, with the chem. potential
        # measured from the lowest energy state
        self.Tmu = self.T / mu 

        # We define an etaF_star which allows us to control for atoms 
        # spilling along the beams in situations with non-zero temperature
        # such as what we can access with HTSE
        self.etaF_star = self.EtaEvap - self.Tmu*1.4
            


        # Obtain trap integrated values of the thermodynamic quantities
        self.Number  = self.getNumber( self.globalMu, self.T )
        self.Entropy = self.getEntropy( self.T)

 

             

    def Info( self ):
        """
        Returns a latex string with the information pertinent to the 
        hubbard parameters
        """
        # Tunneling label 
        tmin = self.tunneling_111.min()  
        tmin_kHz = tmin * 29.2 
        tlabel = '$t=%.2f\,\mathrm{kHz}$'%(tmin_kHz)
        # Scattering length
        aslabel = '$a_{s}=%.0fa_{0}$' % self.a_s 
        # U/t label 
        Utlabel = '$U/t=%.1f$' % self.onsite_t_111.max()
        # Temperature label
        Tlabel = '$T/t=%.1f$' % (self.T/self.tunneling_111).max()

        LDAlabel = '\n'.join( [ aslabel, Utlabel, Tlabel, tlabel ] ) 
        return LDAlabel    

    def ThermoInfo( self ):
        """
        Returns a latex string with the information pertinent to the 
        calculated  thermodynamic quantities. 
        """
        wLs = self.pot.w 
        waists = sum( wLs, ())
        wL = np.mean(waists)  

        self.NumberD = self.getNumberD( self.T )

        rlabel = r'$\mathrm{HWHM} = %.2f\,w_{L}$' % ( self.getRadius()/wL  )  
        Nlabel = r'$N=%.2f\times 10^{5}$' % (self.Number/1e5)
        Dlabel = r'$D=%.3f$' % ( self.NumberD / self.Number )
        Slabel = r'$S/N=%.2fk_{\mathrm{B}}$' % ( self.Entropy / self.Number )
        return '\n'.join([rlabel, Nlabel, Dlabel, Slabel]) 
    
        
   
    def getRadius( self ):
        """
        This function calculates the HWHM (half-width at half max) of the 
        density distribution.    
        """
        gMu = self.globalMu
        T   = self.T

        gMuZero = self.Ezero0_111 + gMu
        localMu = gMuZero - self.Ezero_111
        density = get_dens( T, self.tunneling_111, localMu, \
                      self.onsite_111, select=self.select,\
                      ignoreLowT=self.ignoreLowT, \
                      verbose=self.verbose)
        posradii = self.r111 >= 0. 
        r111pos = self.r111[ posradii]  
        posdens =  density[ posradii ]
 
        try:
            hwhm = r111pos[ posdens - posdens[0]/2. < 0.][0]
            return  hwhm 
        except:
            print r111pos
            print posdens 
            raise


    def get_localMu_t( self, gMu):
        gMuZero = self.Ezero0_111 + gMu
        localMu = gMuZero - self.Ezero_111
        localMu_t = localMu / self.tunneling_111
        return localMu_t
        

    def getDensity( self, gMu, T ):
        """
        This function calculates and returns the density along
        the 111 direction  

        Parameters 
        ----------
        gMu         :  global chemical potential
 
        """
        gMuZero = self.Ezero0_111 + gMu
        localMu = gMuZero - self.Ezero_111
        localMu_t = localMu / self.tunneling_111
        density = get_dens( T, self.tunneling_111, localMu, \
                      self.onsite_111, select=self.select,\
                      ignoreLowT=self.ignoreLowT, \
                      verbose=self.verbose)
        return self.r111 ,  density

    def getEntropy111( self, gMu, T ):
        """
        This function calculates and returns the entropy along
        the 111 direction  

        Parameters 
        ----------
        gMu         :  global chemical potential
 
        """
        gMuZero = self.Ezero0_111 + gMu
        localMu = gMuZero - self.Ezero_111
        localMu_t = localMu / self.tunneling_111
        entropy = get_entr( T, self.tunneling_111, localMu, \
                      self.onsite_111, select=self.select,\
                      ignoreLowT=self.ignoreLowT, \
                      verbose=self.verbose)
        return self.r111 ,  entropy
 
    def getSpi111( self, gMu, T ):
        """
        This function calculates and returns the structure factor along
        the 111 direction  

        Parameters 
        ----------
        gMu         :  global chemical potential
 
        """
        gMuZero = self.Ezero0_111 + gMu
        localMu = gMuZero - self.Ezero_111
        localMu_t = localMu / self.tunneling_111
        spi = get_spi( T, self.tunneling_111, localMu, \
                      self.onsite_111, select=self.select,\
                      ignoreLowT=self.ignoreLowT, \
                      verbose=self.verbose)
        return self.r111 ,  spi

    def getBulkSpi( self, **kwargs ):
        r111, n111 = self.getDensity( self.globalMu, self.T )

        t0 = self.tunneling_111.min() 
        
        Tspi = kwargs.get( 'Tspi', self.T / t0  )
        logger.info(  "Tspi in units of t0 = " + str(Tspi) )  
        Tspi = Tspi * t0
        logger.info( "Tspi in units of Er = " + str(Tspi) ) 
        logger.info( "  t0 in units of Er = " + str( t0 ) ) 
    

        gMuZero = self.Ezero0_111 + self.globalMu
        localMu = gMuZero - self.Ezero_111
        localMu_t = localMu / self.tunneling_111

        # Get the bulk Spi and the Spi profile
        # ALSO  
        # Get the overall S/N and the s profiles,  both s per lattice site
        # and s per particle 
        spibulk, spi, overall_entropy, entropy, lda_number, density =  \
             qmc_spi.spi_bulk( r111, n111, localMu_t, Tspi, \
             self.tunneling_111, self.onsite_111, **kwargs )

        do_k111 = kwargs.get('do_k111', False)
        if do_k111:
            # Get the compressibility
            k111 = get_cmpr( self.T, self.tunneling_111, localMu, \
                          self.onsite_111, select=self.select,\
                          ignoreLowT=self.ignoreLowT, \
                          verbose=self.verbose)
            
            k111htse_list = [] 
            for Thtse in [ 1.8, 2.3, 2.8]:
                k111htse = get_cmpr( Thtse*t0, self.tunneling_111, localMu, \
                          self.onsite_111, select='htse',\
                          ignoreLowT=self.ignoreLowT, \
                          verbose=self.verbose)
                k111htse_list.append( [Thtse, k111htse] )
        else:
            k111 = None
            k111htse_list = []  
        

        U111 = self.onsite_111 / self.tunneling_111

        return spibulk, spi, r111, n111, U111, self.tunneling_111, \
               overall_entropy, entropy, lda_number, density, k111, \
               k111htse_list

    def getSpiFineGrid( self, **kwargs):
        direc111 = (np.arctan(np.sqrt(2)), np.pi/4)
        unit = vec3(); th = direc111[0]; ph = direc111[1] 
        unit.set_spherical( 1., th, ph); 

        numpoints = kwargs.pop('numpoints', 80 ) 
        t111, X111_, Y111_, Z111_, lims_ = \
            udipole.linecut_points( direc=direc111, extents=self.extents,\
            npoints=numpoints)
        # Below we get the signed distance from the origin
        r111_ =  X111_*unit[0] + Y111_*unit[1] + Z111_*unit[2]

 
        # Obtain band structure and interactions along the 111 direction
        bandbot_111_, bandtop_111_,  \
        Ezero_111_, tunneling_111_, onsite_t_111_ = \
            self.pot.bandStructure( X111_, Y111_, Z111_)


        # The onsite interactions are scaled up by the scattering length
        onsite_t_111_ = self.a_s * onsite_t_111_
        onsite_111_ = onsite_t_111_ * tunneling_111_

        # Lowst value of E0 is obtained 
        LowestE0_ = np.amin( bandbot_111_ )  
        Ezero0_111_ = Ezero_111_.min()

        t0 = tunneling_111_.min() 
        Tspi = kwargs.get( 'Tspi', self.T / t0  )
        Tspi = Tspi * t0

        localMu_ = self.globalMu +  Ezero0_111_ - Ezero_111_
        localMu_t_ = localMu_ / tunneling_111_

        # Get the density 
        density_ = get_dens( self.T, tunneling_111_, localMu_, \
                      onsite_111_, select=self.select,\
                      ignoreLowT=self.ignoreLowT, \
                      verbose=self.verbose)

        # Get the bulk Spi and the Spi profile
        # ALSO  
        # Get the overall S/N and the s profiles,  both s per lattice site
        # and s per particle 
        kwargs['do_kappa']=True 
        spibulk, spi, overall_entropy, entropy, \
            lda_number, density, compr =  \
             qmc_spi.spi_bulk( r111_, density_, localMu_t_, Tspi, \
             tunneling_111_, onsite_111_, **kwargs )


        U111 = onsite_111_ / tunneling_111_

        #return spibulk, spi, r111, n111, U111, self.tunneling_111, \
        #       overall_entropy, entropy, lda_number, density
        return r111_, spi, density_, compr,  localMu_t_ * tunneling_111_


    def getNumber( self, gMu, T, **kwargs):
        """ 
        This function calculates and returns the total number of atoms.  
        It integrates along 111 assuming a spherically symmetric sample. 

        Parameters 
        ----------
        gMu         :  global chemical potential
 
        """

        kwverbose = kwargs.get('verbose', None)
        if kwverbose is not None:
            NVerbose = kwverbose
        else:
            NVerbose = self.verbose


        gMuZero = self.Ezero0_111 + gMu
        localMu = gMuZero - self.Ezero_111
        localMu_t = localMu / self.tunneling_111
        density = get_dens( T, self.tunneling_111, localMu, \
                      self.onsite_111, select=self.select,\
                      ignoreLowT=self.ignoreLowT, \
                      verbose=self.verbose)


        # Under some circumnstances the green compensation can 
        # cause dips in the density profile.  This can happen only 
        # if the green beam waist is smaller than the IR beam waist 
        # Experimentally we have seen that we do not handle these very
        # well, so we want to avoid them at all cost 
        # The occurence of this is flagged by a change in the derivative
        # of the radial density.  This derivative should always be negative. 

        # If the green beam waist is larger than the IR beam waist, then 
        # the problem with the non-monotonic density can also be found
        # when trying to push the compensation such that muGlobal gets 
        # close to the evaporation threshold 
        # This can be pathological because it leads to an accumulation of atoms
        # that are not trapped but this lda code integrates over them and counts
        # them anyways.  
        
        # To avoid any of the two situations desribed above we force the
        # density to decrease monotonically over the extent of our calculation. 

        # If the density slope is positive the we report it as an error 
        # 
        # find the point at which the density changes derivative
        radius_check = 1e-3 
        posradii = self.r111 > radius_check 
        posdens =  density[ posradii ]
        neg_slope = np.diff( posdens ) > 1e-4
        n_neg_slope = np.sum( neg_slope )

 


        if n_neg_slope > 0:  
            msg = "ERROR: Radial density profile along 111 " + \
                  "has a positive slope"
            if NVerbose:
                print msg
                print "\n\nradius check start = ", radius_check
                print posdens
                print np.diff( posdens ) > 1e-4
            if not self.ignoreSlopeErrors:
                raise ValueError(msg)
        elif NVerbose:
            print "OK: Radial density profile along 111 decreases " + \
                  "monotonically."
        if False:
            print " posdens len = ",len(posdens)
            print " n_neg_slope = ",n_neg_slope

        # Checks that the density goes to zero within the current extents
        if kwverbose is not None and kwverbose is False:
            edgecuttof = 10.
        else: 
            edgecuttof = 2e-2

        if posdens[-1]/posdens[0] > edgecuttof:
            msg = "ERROR: Density does not vanish within current " + \
                  "extents"
            if not self.ignoreExtents:
                print msg
                print posdens[0]
                print posdens[-1]
                print posdens
                print self.pot.g0
                #print "etaF = ", self.EtaEvap
                #print "etaFstar = ", self.etaF_star
                #print "extents = ", self.extents
                raise ValueError(msg)
            if NVerbose:
                print msg
                print posdens[0]
                print posdens[-1]
                print self.pot.g0

         
        dens = density[~np.isnan(density)]
        r = self.r111[~np.isnan(density)]
        self.PeakD = dens.max()
        return np.power(self.a,-3)*2*np.pi*integrate.simps(dens*(r**2),r)

    def getNumberD( self, T):
        """ 
        This function calculates and returns the total number of doublons. 
        It integrates along 111 assuming a spherically symmetric sample. 

        """
        doublons = get_doub( T, self.tunneling_111, self.localMu_111,\
                              self.onsite_111, select=self.select,\
                              ignoreLowT=self.ignoreLowT,\
                              verbose=self.verbose) 
        doub = doublons[~np.isnan(doublons)]
        r = self.r111[~np.isnan(doublons)]
        return 2.*np.power(self.a,-3)*2*np.pi*integrate.simps(doub*(r**2),r)
    
    def getEntropy( self, T):
        """ 
        This function calculates and returns the total entropy.  
        It integrates along 111 assuming a spherically symmetric sample. 

        """
        entropy  = get_entr( T, self.tunneling_111, self.localMu_111,\
                              self.onsite_111, select=self.select,\
                              ignoreLowT=self.ignoreLowT,\
                              verbose=self.verbose) 
        entr = entropy[~np.isnan(entropy)]
        r = self.r111[~np.isnan(entropy)]
        return np.power(self.a,-3)*2*np.pi*integrate.simps(entr*(r**2),r)

    def column_density(  self ):
        """
        This function calculates and returns the column density of the
        cloud 
        """ 

        return None
        



def plotLine(  lda0, **kwargs):
    # Flag to ignore errors related to low temperatures beyond the reach
    # of the htse  
    ignoreLowT = kwargs.get('ignoreLowT',False)

    scale = 0.9
    figGS = plt.figure(figsize=(6.0*scale,4.2*scale))
    gs3Line = matplotlib.gridspec.GridSpec(2,2,\
                 width_ratios=[1.6, 1.], height_ratios=[2.0,1.4],\
                 wspace=0.25, 
                 left=0.13, right=0.90,
                 bottom=0.15, top=0.78) 
    tightrect = [0.,0.00, 0.95, 0.84]

    Ax1 = []; 
    Ymin =[]; Ymax=[]

    line_direction  = kwargs.pop('line_direction', '111')
    direcs = { \
               '100':(np.pi/2, 0.), \
               '010':(np.pi/2, np.pi/2), \
               '001':(0., np.pi), \
               '111':(np.arctan(np.sqrt(2)), np.pi/4) } 
    labels = { \
               '100':'$(\mathbf{100})$', \
               '010':'$(\mathbf{010})$', \
               '001':'$(\mathbf{001})$', \
               '111':'$(\mathbf{111})$' } 

    cutkwargs = kwargs.pop( 'cutkwargs', {  } ) 
    cutkwargs['direc'] = direcs[ line_direction ] 
    cutkwargs['ax0label']= labels[ line_direction ]   
    cutkwargs['extents']= kwargs.pop('extents', 40.)
    t, X,Y,Z, lims = udipole.linecut_points( **cutkwargs ) 

    potkwargs = kwargs.pop( 'potkwargs', {  } ) 
    potkwargs['direc'] = direcs[ line_direction ] 
    potkwargs['ax0label']= labels[ line_direction ]   
    potkwargs['extents']= kwargs.pop('x1lims', (lims[0],lims[1]))[1]
    tp, Xp,Yp,Zp, lims = udipole.linecut_points( **potkwargs )

     


    kwargs['suptitleY'] = 0.96
    kwargs['foottextY'] = 0.84
  
    x1lims = kwargs.get('x1lims', (lims[0],lims[1])) 
    
    ax1 = figGS.add_subplot( gs3Line[0:3,0] )
    ax1.set_xlim( *x1lims )
    ax1.grid()
    ax1.grid(which='minor')
    ax1.set_xlabel('$\mu\mathrm{m}$ '+cutkwargs['ax0label'], fontsize=13.)
    ax1.set_ylabel( lda0.pot.unitlabel, rotation=0, fontsize=13., labelpad=15 )
    ax1.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(20) ) 
    ax1.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) ) 

    ax1.yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(7) ) 
    ax1.yaxis.set_minor_locator( matplotlib.ticker.AutoMinorLocator() ) 

    ax2 = figGS.add_subplot( gs3Line[0,1] )
    ax3 = None
    #ax2.grid()
    ax2.set_xlabel('$\mu\mathrm{m}$', fontsize=12, labelpad=0)
    #ax2.set_ylabel('$n$', rotation=0, fontsize=14, labelpad=11 )
    ax2.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(20) ) 
    ax2.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) ) 
        

    #----------------------------------
    # CALCULATE ALL RELEVANT QUANTITIES
    #----------------------------------
    # All the relevant lines are first calculated here 
    bandbot_XYZ, bandtop_XYZ,  \
    Ezero_XYZ, tunneling_XYZ, onsite_t_XYZ = \
        lda0.pot.bandStructure( X, Y, Z ) 
    # The onsite interactions are scaled up by the scattering length
    onsite_t_XYZ = lda0.a_s * onsite_t_XYZ

    onsite_XYZ = onsite_t_XYZ * tunneling_XYZ
    Ezero0_XYZ = Ezero_XYZ.min()

    bottom = lda0.pot.Bottom( X, Y, Z ) 
    lattmod = lda0.pot.LatticeMod( X, Y, Z ) 

    excbot_XYZ, exctop_XYZ = lda0.pot.firstExcited( X, Y, Z ) 

    # Offset the chemical potential for use in the phase diagram
    localMu_XYZ =  ( lda0.globalMu + lda0.Ezero0_111 - Ezero_XYZ )
                      

    # Obtain the thermodynamic quantities
    density_XYZ = get_dens( lda0.T, tunneling_XYZ,  localMu_XYZ, \
                      onsite_XYZ, select=lda0.select, ignoreLowT=ignoreLowT ) 
    doublon_XYZ = get_doub( lda0.T, tunneling_XYZ,  localMu_XYZ, \
                      onsite_XYZ, select=lda0.select, ignoreLowT=ignoreLowT ) 
    entropy_XYZ = get_entr( lda0.T, tunneling_XYZ,  localMu_XYZ, \
                      onsite_XYZ, select=lda0.select, ignoreLowT=ignoreLowT )


    # All the potential lines are recalculated to match the potential
    # xlims
    bandbot_XYZp, bandtop_XYZp,  \
    Ezero_XYZp, tunneling_XYZp, onsite_t_XYZp = \
        lda0.pot.bandStructure( Xp, Yp, Zp ) 
    # The onsite interactions are scaled up by the scattering length
    onsite_t_XYZp = lda0.a_s * onsite_t_XYZp

    onsite_XYZp = onsite_t_XYZp * tunneling_XYZp
    Ezero0_XYZp = Ezero_XYZp.min()

    bottomp = lda0.pot.Bottom( Xp, Yp, Zp ) 
    lattmodp = lda0.pot.LatticeMod( Xp, Yp, Zp ) 

    excbot_XYZp, exctop_XYZp = lda0.pot.firstExcited( Xp, Yp, Zp ) 

    # Offset the chemical potential for use in the phase diagram
    localMu_XYZp =  ( lda0.globalMu + lda0.Ezero0_111 - Ezero_XYZp )

     
    #--------------------------
    # SETUP LINES TO BE PLOTTED 
    #--------------------------
    # A list of lines to plot is generated 
    # Higher zorder puts stuff in front
    toplot = [ 
             {'x':tp,\
              'y':(bandbot_XYZp, Ezero_XYZp ), 'color':'blue', 'lw':2., \
              'fill':True, 'fillcolor':'blue', 'fillalpha':0.75,\
               'zorder':10, 'label':'$\mathrm{band\ lower\ half}$'},
             
             {'x':tp,\
              'y':(Ezero_XYZp + onsite_XYZp, bandtop_XYZp + onsite_XYZp), \
              'color':'purple', 'lw':2., \
              'fill':True, 'fillcolor':'plum', 'fillalpha':0.75,\
              'zorder':10, 'label':'$\mathrm{band\ upper\ half}+U$'},
              
             {'x':tp,\
              'y':(excbot_XYZp, exctop_XYZp ), 'color':'red', 'lw':2., \
              'fill':True, 'fillcolor':'pink', 'fillalpha':0.75,\
               'zorder':2, 'label':'$\mathrm{first\ excited\ band}$'},
             
             {'x':tp,\
              'y':np.ones_like(Xp)*lda0.globalMuZ0, 'color':'limegreen',\
              'lw':2,'zorder':1.9, 'label':'$\mu_{0}$'},

             {'x':tp,\
              'y':np.ones_like(Xp)*lda0.evapTH0_100, 'color':'#FF6F00', \
              'lw':2,'zorder':1.9, 'label':'$\mathrm{evap\ threshold}$'},
             
             {'x':tp,\
              'y':bottomp,'color':'gray', 'lw':0.5,'alpha':0.5},
             {'x':tp,\
              'y':lattmodp,'color':'gray', 'lw':1.5,'alpha':0.5,\
              'label':r'$\mathrm{lattice\ potential\ \ }\lambda\times10$'} \
             ]  
 
    toplot = toplot + [
         {'y':density_XYZ, 'color':'blue', 'lw':1.75, \
          'axis':2, 'label':'$n$'},

         {'y':doublon_XYZ, 'color':'red', 'lw':1.75, \
          'axis':2, 'label':'$d$'},

         {'y':entropy_XYZ, 'color':'black', 'lw':1.75, \
          'axis':2, 'label':'$s_{L}$'},

         #{'y':density-2*doublons,  'color':'green', 'lw':1.75, \
         # 'axis':2, 'label':'$n-2d$'},

         #{'y':self.localMu_t,  'color':'cyan', 'lw':1.75, \
         # 'axis':2, 'label':r'$\mu$'},

         ]

    toplot = toplot + [                 
         {'y':entropy_XYZ/density_XYZ,  'color':'gray', 'lw':1.75, \
          'axis':3, 'label':'$s_{N}$'} ]

    lattlabel = '\n'.join(  list( lda0.pot.Info() ) + \
                            [lda0.pot.TrapFreqsInfo() + r',\ ' \
                             + lda0.pot.EffAlpha(), \
                             '$\eta_{F}=%.2f$'%lda0.EtaEvap + '$,$ ' \
                             '$\Delta_{F}=%.2fE_{R}$'%lda0.DeltaEvap, \
                        
                         ] )
    toplot = toplot + [ {'text':True, 'x': -0.1, 'y':1.02, 'tstring':lattlabel,
                         'ha':'left', 'va':'bottom', 'linespacing':1.4} ]

    toplot = toplot + [ {'text':True, 'x': 1.0, 'y':1.02, 'tstring':lda0.Info(),
                         'ha':'right', 'va':'bottom', 'linespacing':1.4} ]
 
    toplot = toplot + [ {'text':True, 'x': 0., 'y':1.02, \
                         'tstring':lda0.ThermoInfo(), \
                         'ha':'left', 'va':'bottom', 'axis':2, \
                         'linespacing':1.4} ] 

    #--------------------------
    # ITERATE AND PLOT  
    #--------------------------
        
    Emin =[]; Emax=[]
    for p in toplot:
        if not isinstance(p,dict):
            ax1.plot(t,p); Emin.append(p.min()); Emax.append(p.max())
        else:
            if 'text' in p.keys():
                whichax = p.get('axis',1)
                axp = ax2 if whichax ==2 else ax1
                 

                tx = p.get('x', 0.)
                ty = p.get('y', 1.)
                ha = p.get('ha', 'left')
                va = p.get('va', 'center')
                ls = p.get('linespacing', 1.)
                tstring = p.get('tstring', 'empty') 

                axp.text( tx,ty, tstring, ha=ha, va=va, linespacing=ls,\
                    transform=axp.transAxes)
            

            elif 'figprop' in p.keys():
                figsuptitle = p.get('figsuptitle',  None)
                figGS.suptitle(figsuptitle, y=kwargs.get('suptitleY',1.0),\
                               fontsize=14)
 
                figGS.text(0.5,kwargs.get('foottextY',1.0),\
                           p.get('foottext',None),fontsize=14,\
                           ha='center') 

            elif 'y' in p.keys():

                if 'x' in p.keys():
                    x = p['x'] 
                else:
                    x = t
 
                labelstr = p.get('label',None)
                porder   = p.get('zorder',2)
                fill     = p.get('fill', False)
                ydat     = p.get('y',None)

                whichax = p.get('axis',1)
                if whichax == 3:
                    if ax3 is None:
                        ax3 = ax2.twinx()  
                    axp = ax3
                            
                else: 
                    axp = ax2 if whichax ==2 else ax1


                if ydat is None: continue
                if fill:
                    axp.plot(x,ydat[0],
                             lw=p.get('lw',2.),\
                             color=p.get('color','black'),\
                             alpha=p.get('fillalpha',0.5),\
                             zorder=porder,\
                             label=labelstr
                             )
                    axp.fill_between( x, ydat[0], ydat[1],\
                                      lw=p.get('lw',2.),\
                                      color=p.get('color','black'),\
                                      facecolor=p.get('fillcolor','gray'),\
                                      alpha=p.get('fillalpha',0.5),\
                                      zorder=porder
                                    )
                    if whichax == 1: 
                        Emin.append( min( ydat[0].min(), ydat[1].min() ))
                        Emax.append( max( ydat[0].max(), ydat[1].max() )) 
                else:
                    axp.plot( x, ydat,\
                              lw=p.get('lw',2.),\
                              color=p.get('color','black'),\
                              alpha=p.get('alpha',1.0),\
                              zorder=porder,\
                              label=labelstr
                            )
                    if whichax == 1: 
                        Emin.append( ydat.min() ) 
                        Emax.append( ydat.max() )
                if whichax == 3:
                    ax3.tick_params(axis='y', colors=p.get('color','black'))
                #print labelstr
                #print Emin
                #print Emax 
                  
    if ax3 is not None:
        ax3.yaxis.set_major_locator( \
            matplotlib.ticker.MaxNLocator(6, prune='upper') ) 

       
    handles2, labels2 = ax2.get_legend_handles_labels()
    handles3, labels3 = ax3.get_legend_handles_labels()
    handles = handles2 + handles3 
    labels  = labels2 + labels3
    ax2.legend( handles, labels,  bbox_to_anchor=(1.25,1.0), \
        loc='lower right', numpoints=1, labelspacing=0.2, \
         prop={'size':10}, handlelength=1.1, handletextpad=0.5 )
        
        
    Emin = min(Emin); Emax=max(Emax)
    dE = Emax-Emin
    
    
    # Finalize figure
    x2lims = kwargs.get('x2lims', (lims[0],lims[1])) 
    ax2.set_xlim( *x2lims )
    y0,y1 = ax2.get_ylim()
    if y1 == 1. :
        ax2.set_ylim( y0 , y1 + (y1-y0)*0.05)
    y2lims = kwargs.get('y2lims', None) 
    if y2lims is not None:
        ax2.set_ylim( *y2lims) 

    y3lims = kwargs.get('y3lims', None)
    if y3lims is not None:
        ax3.set_ylim( *y3lims)


    ymin, ymax =  Emin-0.05*dE, Emax+0.05*dE

    Ymin.append(ymin); Ymax.append(ymax); Ax1.append(ax1)

    Ymin = min(Ymin); Ymax = max(Ymax) 
    #print Ymin, Ymax
    for ax in Ax1:
        ax.set_ylim( Ymin, Ymax)
  
    if 'ax1ylim' in kwargs.keys():
        ax1.set_ylim( *kwargs['ax1ylim'] ) 
    
        
    Ax1[0].legend( bbox_to_anchor=(1.1,-0.15), \
        loc='lower left', numpoints=1, labelspacing=0.2,\
         prop={'size':9.5}, handlelength=1.1, handletextpad=0.5 )
        
    #gs3Line.tight_layout(figGS, rect=tightrect)
    return figGS



def plotMathy(  lda0, **kwargs):
    # Flag to ignore errors related to low temperatures beyond the reach
    # of the htse  
    ignoreLowT = kwargs.get('ignoreLowT',False)

    scale = 0.9
    figGS = plt.figure(figsize=(6.0*scale,4.2*scale))
    #figGS = plt.figure(figsize=(5.6,4.2))
    gs3Line = matplotlib.gridspec.GridSpec(3,2,\
                 width_ratios=[1.6, 1.], height_ratios=[2.2,0.8,1.2],\
                 wspace=0.2, hspace=0.24, 
                 left = 0.15, right=0.95, bottom=0.14, top=0.78)
    #tightrect = [0.,0.00, 0.95, 0.88]

    Ax1 = []; 
    Ymin =[]; Ymax=[] 

    line_direction  = kwargs.pop('line_direction', '111')
    direcs = { \
               '100':(np.pi/2, 0.), \
               '010':(np.pi/2, np.pi/2), \
               '001':(0., np.pi), \
               '111':(np.arctan(np.sqrt(2)), np.pi/4) } 
    labels = { \
               '100':'$(\mathbf{100})$', \
               '010':'$(\mathbf{010})$', \
               '001':'$(\mathbf{001})$', \
               '111':'$(\mathbf{111})$' } 

    cutkwargs = kwargs.pop( 'cutkwargs', {} ) 
    cutkwargs['direc'] = direcs[ line_direction ] 
    cutkwargs['ax0label']= labels[ line_direction ]   
    cutkwargs['extents']= kwargs.pop('extents', 40.)
    t, X,Y,Z, lims = udipole.linecut_points( **cutkwargs ) 


   
    
    ax1 = figGS.add_subplot( gs3Line[0:2,0] )
    ax1.grid()
    ax1.grid(which='minor')
    ax1.set_ylabel( lda0.pot.unitlabel, rotation=0, fontsize=16, labelpad=15 )
    ax1.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(7) ) 
    #ax1.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(20) ) 

    ax1.yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(7) ) 
    #ax1.yaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(1.) ) 

    ax2 = figGS.add_subplot( gs3Line[0,1] )
    ax2.grid()
    #ax2.set_ylabel('$n$', rotation=0, fontsize=14, labelpad=11 )
    ax2.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(6) ) 
    #ax2.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) )

    ax3 = figGS.add_subplot( gs3Line[2,0] ) 
    ax3.grid() 
    ax3.yaxis.set_major_locator( matplotlib.ticker.MaxNLocator(3) ) 
    ax3.xaxis.set_major_locator( matplotlib.ticker.MaxNLocator(7) ) 
        

    #----------------------------------
    # CALCULATE ALL RELEVANT QUANTITIES
    #----------------------------------
    # All the relevant lines are first calculated here
   
    # In the Mathy plot the x-axis is the local lattice depth
    s0_XYZ = lda0.pot.S0( X, Y, Z)[0] 

    ax1.set_xlim( s0_XYZ.min(), s0_XYZ.max() )
    ax3.set_xlim( s0_XYZ.min(), s0_XYZ.max() )


    x2lims = kwargs.get('x2lims', None) 
    if x2lims is not None:
        ax2.set_xlim( *x2lims) 
    else:
        ax2.set_xlim( s0_XYZ.min(), s0_XYZ.max() )

    ax3.set_xlabel('$s_{0}\,(E_{R}) $', fontsize=13)
    ax2.set_xlabel('$s_{0}\,(E_{R}) $', fontsize=12, labelpad=0)
 
    bandbot_XYZ, bandtop_XYZ,  \
    Ezero_XYZ, tunneling_XYZ, onsite_t_XYZ = \
        lda0.pot.bandStructure( X, Y, Z ) 
    # The onsite interactions are scaled up by the scattering length
    onsite_t_XYZ = lda0.a_s * onsite_t_XYZ

    onsite_XYZ = onsite_t_XYZ * tunneling_XYZ
    Ezero0_XYZ = Ezero_XYZ.min()

    bottom = lda0.pot.Bottom( X, Y, Z ) 
    lattmod = lda0.pot.LatticeMod( X, Y, Z )

    Mod = np.amin( lda0.pot.S0( X, Y, Z), axis=0 ) 
    deltas0 =  ( s0_XYZ.max()-s0_XYZ.min() )
    lattmod = lda0.pot.Bottom( X, Y, Z ) + \
        Mod*np.power( np.cos( 2.*np.pi* s0_XYZ *10./deltas0 ), 2)

    excbot_XYZ, exctop_XYZ = lda0.pot.firstExcited( X, Y, Z ) 

    # Offset the chemical potential for use in the phase diagram
    localMu_XYZ =  ( lda0.globalMu + lda0.Ezero0_111 - Ezero_XYZ ) 
                     

    # Obtain the thermodynamic quantities
    density_XYZ = get_dens( lda0.T, tunneling_XYZ,  localMu_XYZ, \
                      onsite_XYZ, select=lda0.select, ignoreLowT=ignoreLowT ) 
    doublon_XYZ = get_doub( lda0.T, tunneling_XYZ,  localMu_XYZ, \
                      onsite_XYZ, select=lda0.select, ignoreLowT=ignoreLowT ) 
    entropy_XYZ = get_entr( lda0.T, tunneling_XYZ,  localMu_XYZ, \
                      onsite_XYZ, select=lda0.select, ignoreLowT=ignoreLowT ) 

     
    #--------------------------
    # SETUP LINES TO BE PLOTTED 
    #--------------------------
    # A list of lines to plot is generated 
    # Higher zorder puts stuff in front
    toplot = [ 
             {'y':(bandbot_XYZ, Ezero_XYZ ), 'color':'blue', 'lw':2., \
              'fill':True, 'fillcolor':'blue', 'fillalpha':0.5,\
               'zorder':10, 'label':'$\mathrm{band\ lower\ half}$'},
             
             {'y':(Ezero_XYZ + onsite_XYZ, bandtop_XYZ + onsite_XYZ), \
              'color':'purple', 'lw':2., \
              'fill':True, 'fillcolor':'plum', 'fillalpha':0.5,\
              'zorder':10, 'label':'$\mathrm{band\ upper\ half}+U$'},

             {'y':(Ezero_XYZ, Ezero_XYZ + onsite_XYZ), \
              'color':'black', 'lw':2., \
              'fill':True, 'fillcolor':'gray', 'fillalpha':0.85,\
              'zorder':10, 'label':'$\mathrm{mott\ gap}$'},
              
             #{'y':(excbot_XYZ, exctop_XYZ ), 'color':'red', 'lw':2., \
             # 'fill':True, 'fillcolor':'pink', 'fillalpha':0.75,\
             #  'zorder':2, 'label':'$\mathrm{first\ excited\ band}$'},
             
             {'y':np.ones_like(X)*lda0.globalMuZ0, 'color':'limegreen',\
              'lw':2,'zorder':1.9, 'label':'$\mu_{0}$'},

             {'y':np.ones_like(X)*lda0.evapTH0_100, 'color':'#FF6F00', \
              'lw':2,'zorder':1.9, 'label':'$\mathrm{evap\ threshold}$'},
             
             #{'y':bottom,'color':'gray', 'lw':0.5,'alpha':0.5, 'axis':3},
             {'y':lattmod,'color':'gray', 'lw':1.5,'alpha':0.5, \
              'axis':3,\
              'label':r'$\mathrm{lattice\ potential\ \ }\lambda\times10$'} \
             ]  

    entropy_per_particle = kwargs.pop('entropy_per_particle', False)
    if entropy_per_particle:
        toplot = toplot + [                 
             {'y':entropy_XYZ/density_XYZ,  'color':'black', 'lw':1.75, \
              'axis':2, 'label':'$s_{N}$'} ] 
    else:
        toplot = toplot + [
             {'y':density_XYZ, 'color':'blue', 'lw':1.75, \
              'axis':2, 'label':'$n$'},

             {'y':doublon_XYZ, 'color':'red', 'lw':1.75, \
              'axis':2, 'label':'$d$'},

             {'y':entropy_XYZ, 'color':'black', 'lw':1.75, \
              'axis':2, 'label':'$s_{L}$'},

             #{'y':density-2*doublons,  'color':'green', 'lw':1.75, \
             # 'axis':2, 'label':'$n-2d$'},

             #{'y':self.localMu_t,  'color':'cyan', 'lw':1.75, \
             # 'axis':2, 'label':r'$\mu$'},

             ]

    lattlabel = '\n'.join(  list( lda0.pot.Info() ) + \
                            [lda0.pot.TrapFreqsInfo() + r',\ ' \
                             + lda0.pot.EffAlpha(), \
                             '$\eta_{F}=%.2f$'%lda0.EtaEvap + '$,$ ' \
                             '$\Delta_{F}=%.2fE_{R}$'%lda0.DeltaEvap, \
                        
                         ] )
    toplot = toplot + [ {'text':True, 'x': 0., 'y':1.02, 'tstring':lattlabel,
                         'ha':'left', 'va':'bottom', 'linespacing':1.4} ]

    toplot = toplot + [ {'text':True, 'x': 1.0, 'y':1.02, 'tstring':lda0.Info(),
                         'ha':'right', 'va':'bottom', 'linespacing':1.4} ]
 
    toplot = toplot + [ {'text':True, 'x': 0., 'y':1.02, \
                         'tstring':lda0.ThermoInfo(), \
                         'ha':'left', 'va':'bottom', 'axis':2, \
                         'linespacing':1.4} ] 


    #--------------------------
    # ITERATE AND PLOT  
    #--------------------------
    kwargs['suptitleY'] = 0.96
    kwargs['foottextY'] = 0.84
     
    # For every plotted quantity I use only lthe positive radii  
    Emin =[]; Emax=[]
    positive = t > 0.
    xarray = s0_XYZ[ positive ] 
    for p in toplot:
        if not isinstance(p,dict):
            p = p[positive] 
            ax1.plot(xarray,p); Emin.append(p.min()); Emax.append(p.max())
        else:
            if 'text' in p.keys():
                whichax = p.get('axis',1)
                axp = ax2 if whichax ==2 else ax1

                tx = p.get('x', 0.)
                ty = p.get('y', 1.)
                ha = p.get('ha', 'left')
                va = p.get('va', 'center')
                ls = p.get('linespacing', 1.)
                tstring = p.get('tstring', 'empty') 

                axp.text( tx,ty, tstring, ha=ha, va=va, linespacing=ls,\
                    transform=axp.transAxes)

            elif 'figprop' in p.keys():
                figsuptitle = p.get('figsuptitle',  None)
                figGS.suptitle(figsuptitle, y=kwargs.get('suptitleY',1.0),\
                               fontsize=14)
 
                figGS.text(0.5,kwargs.get('foottextY',1.0),\
                           p.get('foottext',None),fontsize=14,\
                           ha='center') 

            elif 'y' in p.keys():
                whichax = p.get('axis',1)
                #if whichax == 2 : continue
                axp = ax2 if whichax ==2 else ax3 if  whichax == 3 else ax1

                labelstr = p.get('label',None)
                porder   = p.get('zorder',2)
                fill     = p.get('fill', False)
                ydat     = p.get('y',None)

                if ydat is None: continue

                if fill:
                    ydat = ( ydat[0][positive], ydat[1][positive] ) 
                    axp.plot(xarray,ydat[0],
                             lw=p.get('lw',2.),\
                             color=p.get('color','black'),\
                             alpha=p.get('fillalpha',0.5),\
                             zorder=porder,\
                             label=labelstr
                             )
                    axp.fill_between( xarray, ydat[0], ydat[1],\
                                      lw=p.get('lw',2.),\
                                      color=p.get('color','black'),\
                                      facecolor=p.get('fillcolor','gray'),\
                                      alpha=p.get('fillalpha',0.5),\
                                      zorder=porder
                                    )
                    if whichax == 1: 
                        Emin.append( min( ydat[0].min(), ydat[1].min() ))
                        Emax.append( max( ydat[0].max(), ydat[1].max() )) 
                else:
                    ydat = ydat[ positive ] 
                    axp.plot( xarray, ydat,\
                              lw=p.get('lw',2.),\
                              color=p.get('color','black'),\
                              alpha=p.get('alpha',1.0),\
                              zorder=porder,\
                              label=labelstr
                            )
                    if whichax == 1: 
                        Emin.append( ydat.min() ) 
                        Emax.append( ydat.max() ) 
                  
        
    ax2.legend( bbox_to_anchor=(0.03,1.02), \
        loc='upper left', numpoints=1, labelspacing=0.2, \
         prop={'size':10}, handlelength=1.1, handletextpad=0.5 )
        
        
    Emin = min(Emin); Emax=max(Emax)
    dE = Emax-Emin
    
    
    # Finalize figure
    y0,y1 = ax2.get_ylim()
    ax2.set_ylim( y0 , y1 + (y1-y0)*0.1)


    ymin, ymax =  Emin-0.05*dE, Emax+0.05*dE

    Ymin.append(ymin); Ymax.append(ymax); Ax1.append(ax1)

    Ymin = min(Ymin); Ymax = max(Ymax)
    for ax in Ax1:
        ax.set_ylim( Ymin, Ymax)

    if 'ax1ylim' in kwargs.keys():
        ax1.set_ylim( *kwargs['ax1ylim'] ) 
        
    Ax1[0].legend( bbox_to_anchor=(1.1,0.1), \
        loc='upper left', numpoints=1, labelspacing=0.2,\
         prop={'size':11}, handlelength=1.1, handletextpad=0.5 )
        
    #gs3Line.tight_layout(figGS, rect=tightrect)
    return figGS


 
def CheckInhomog( lda0, **kwargs ):
    """This function will make a plot along 111 of the model parameters:
       U, t, U/t, v0.  

       It is useful to assess the degree of inhomogeneity in our system"""
    
    # Prepare the figure
    fig = plt.figure(figsize=(9.,4.2))
    lattlabel = '\n'.join(  list( lda0.pot.Info() ) )
    lattlabel = '\n'.join( [ i.split( r'$\mathrm{,}\ $' )[0].replace('s','v') \
                                 for i in lda0.pot.Info() ] )
 
    Nlabel = r'$N=%.2f\times 10^{5}$' % (lda0.Number/1e5)
    Slabel = r'$S/N=%.2fk_{\mathrm{B}}$' % ( lda0.Entropy / lda0.Number )
    thermolabel =  '\n'.join([Nlabel, Slabel])

    ldainfoA = '\n'.join(lda0.Info().split('\n')[:2])
    ldainfoB = '\n'.join(lda0.Info().split('\n')[-2:])

 
    fig.text( 0.05, 0.98,  lattlabel, ha='left', va='top', linespacing=1.2)
    fig.text( 0.48, 0.98,  ldainfoA, ha='right', va='top', linespacing=1.2)
    fig.text( 0.52, 0.98,  ldainfoB, ha='left', va='top', linespacing=1.2)
    fig.text( 0.95, 0.98,  thermolabel, ha='right', va='top', linespacing=1.2)

    #fig.text( 0.05, 0.86, "Sample is divided in 5 bins, all containing" +\
    #   " the same number of atoms (see panel 2).\n" + \
    #   "Average Fermi-Hubbard parameters $n$, $U$, $t$, " +\
    #   "and $U/t$ are calculated in each bin (see panels 1, 3, 4, 5 )" )
    
    gs = matplotlib.gridspec.GridSpec( 2,4, wspace=0.18,\
             left=0.1, right=0.9, bottom=0.05, top=0.98)
    
    # Setup axes
    axn  = fig.add_subplot(gs[0,0])
    axnInt = fig.add_subplot(gs[0,3])
    axU  = fig.add_subplot(gs[1,0])
    axt  = fig.add_subplot(gs[1,1])
    axUt = fig.add_subplot(gs[1,2]) 
    axv0 = fig.add_subplot(gs[1,3])

    axEntr = fig.add_subplot( gs[0,1] ) 
    axSpi = fig.add_subplot( gs[0,2] )

    # Set xlim
    x0 = -40.; x1 = 40.
    axn.set_xlim( x0, x1)
    axEntr.set_xlim( x0, x1)
    axEntr.set_ylim( 0., 1.0)
    axSpi.set_xlim( x0, x1)
    axSpi.set_ylim( 0., 3.0)

    axnInt.set_xlim( 0., x1 )
    axU.set_xlim( x0, x1 )
    axU.set_ylim( 0., np.amax( lda0.onsite_t_111 * lda0.tunneling_111 *1.05 ) )
    axt.set_xlim( x0, x1 )
    axt.set_ylim( 0., 0.12)
    axUt.set_xlim( x0, x1 )
    axUt.set_ylim( 0., np.amax( lda0.onsite_t_111 * 1.05 )) 
    axv0.set_xlim( x0, x1 )
    
    lw0 = 2.5
    # Plot relevant quantities 
    r111_, density_111 = lda0.getDensity( lda0.globalMu, lda0.T )
    r111_Entr, entropy_111 = lda0.getEntropy111( lda0.globalMu, lda0.T) 
    r111_Spi, spi_111 = lda0.getSpi111( lda0.globalMu, lda0.T) 
    V0_111 = lda0.pot.S0( lda0.X111, lda0.Y111, lda0.Z111 ) 

    # density, entropy and spi
    axn.plot( lda0.r111, density_111, lw=lw0 , color='black')
    axEntr.plot( lda0.r111, entropy_111, lw=lw0 , color='black')
    axSpi.plot( lda0.r111, spi_111, lw=lw0 , color='black')
    # U 
    axU.plot( lda0.r111, lda0.onsite_t_111 * lda0.tunneling_111 , \
                  lw=lw0, label='$U$', color='black') 
    # t 
    axt.plot( lda0.r111, lda0.tunneling_111,lw=lw0, label='$t$', \
                  color='black')
    # U/t 
    axUt.plot( lda0.r111, lda0.onsite_t_111, lw=lw0, color='black')
 
    # Lattice depth 
    #print "shape of V0 = ", V0_111.shape
    axv0.plot( lda0.r111, V0_111[0], lw=lw0, color='black', \
               label='$\mathrm{Lattice\ depth}$')

    # Band gap 
    bandgap_111 = bands = scubic.bands3dvec( V0_111, NBand=1 )[0] \
                          - scubic.bands3dvec( V0_111, NBand=0 )[1] 
    axv0.plot( lda0.r111, bandgap_111, lw=lw0, linestyle=':', color='black', \
                label='$\mathrm{Band\ gap}$') 

    axv0.legend( bbox_to_anchor=(0.03,0.02), \
        loc='lower left', numpoints=3, labelspacing=0.2,\
         prop={'size':6}, handlelength=1.5, handletextpad=0.5 )


    # Define function to calculate cummulative atom number
    def NRadius( Radius ):
        """
        This function calculates the fraction of the atom number 
        up to a certain Radius
        """
        valid = np.logical_and( np.abs(lda0.r111) < Radius, \
                                ~np.isnan(density_111) )
        r    = lda0.r111[ valid ] 
        dens = density_111[ valid ] 
        return np.power( lda0.pot.l/2, -3) * \
               2 * np.pi*integrate.simps( dens*(r**2), r) / lda0.Number
    
    # Plot the cummulative atom number 
    radii = lda0.r111[ lda0.r111 > 4. ] 
    NInt = []
    for radius in radii:
        NInt.append( NRadius( radius ) ) 
    NInt = np.array( NInt ) 
    axnInt.plot( radii, NInt, lw=lw0, color='black') 

   
    # Define function to numerically solve for y in a pair of x,y arrays     
    def x_solve( x_array, y_array,  yval ):
        """  
        This function solves for x0 in the equation y0=y(x0)  
        where the function y(x) is defined with data arrays. 
        """
        # Convert the array to a function and then solve for y==yval
        yf = interp1d( x_array, y_array-yval, kind='cubic') 
        return optimize.brentq( yf, x_array.min(), x_array.max() )

    def y_solve( x_array, y_array, xval ):
        yf = interp1d( x_array, y_array, kind='cubic')
        return yf(xval) 
     

    radius1e = x_solve( lda0.r111[ lda0.r111 > 0 ] , \
                        density_111[ lda0.r111 > 0 ] , \
                        density_111.max()/np.exp(1.) )

   
    pos_r111 = lda0.r111[ lda0.r111 > 0 ]  
    pos_dens111 = density_111[ lda0.r111 > 0 ]

 
    #slice_type = 'defined_bins'
    slice_type = 'percentage'
    
    if slice_type == 'defined_bins':
        print pos_dens111.max() 
        cutoffs = [ 1.20, 1.05, 0.95, 0.75, 0.50, 0.25, 0.00 ]   
        if pos_dens111.max() < 1.20 :
            cutoffs = cutoffs[1:] 
        if pos_dens111.max() < 1.05 : 
            cutoffs = cutoffs[1:]
     
        nrange0 = [ pos_dens111.max() ] + cutoffs[:-1] 
        nrange1 = cutoffs 
        print nrange0
        print  nrange1
    
        rbins = [] 
        for i in range(len(nrange1)-1):
            if np.any( pos_dens111 > nrange1[i] ): 
                rbins.append(( (nrange1[i] + nrange0[i])/2., \
                              x_solve( pos_r111, pos_dens111, nrange1[i] ) ))
        print rbins
        rcut = [ b[1] for b in rbins ] 
        print " Bins cut radii = ", rcut
  
    elif slice_type == 'percentage':
        # Find the various radii that split the cloud into slots of 20% atom number
        rcut = []
        nrange0 = [ pos_dens111[0] ]
        nrange1 = [] 
        for Ncut in [0.2, 0.4, 0.6, 0.8 ]:
            sol = x_solve( radii, NInt, Ncut )
            rcut.append( sol  )
            
            denssol =  y_solve( pos_r111, pos_dens111,  sol )
            nrange0.append( denssol )  
            nrange1.append( denssol ) 
        nrange1.append(0.)
            


    # get the number of atoms in each bin
    binedges = rcut + [rcut[-1]+20.]    
    Nbin = []
    for b in range(len(rcut) + 1 ):
        if b == 0:
            Nbin.append( NRadius( binedges[b] ) ) 
        else:
            Nbin.append( NRadius(binedges[b]) - NRadius(binedges[b-1]) )
    Nbin = np.array( Nbin )

    Nbinsum = Nbin.sum()
    if np.abs( Nbinsum - 1.0 ) > 0.01:
        print "Total natoms from adding bins = ", Nbinsum
        raise ValueError("Normalization issue with density distribution.")
    
      

    # Define functions to average over the shells        
    def y_average( y_array,  x0, x1):
        # Average y_array over the radii x0 to x1,  weighted by density 
        valid = np.logical_and( np.abs(lda0.r111) < 70., ~np.isnan(density_111) )
        
        r    = lda0.r111[ valid ] 
        dens = density_111[ valid ]
        y    = y_array[ valid ] 
        
        shell = np.logical_and( r >= x0, r<x1 ) 
        r    = r[shell]
        dens = dens[shell]
        y    = y[shell] 
        
        num = integrate.simps( y* dens*(r**2), r) 
        den = integrate.simps(  dens*(r**2), r) 
        return num/den 
    
    # Define a function here that makes a piecewise function with the average
    # values of a quantity so that it can be plotted
    def binned( x, yqty ):
        x = np.abs(x)
        yavg = [] 
        cond = []
        for x0,x1 in zip( [0.]+rcut,  rcut+[rcut[-1]+20.]):
            cond.append(np.logical_and( x >= x0 , x<x1 ) )
            yavg.append( y_average( yqty, x0, x1) ) 
        return np.piecewise( x, cond, yavg ), yavg

    # Calculate and plot the binned quantities
    dens_binned = binned( lda0.r111, density_111 ) 
    entr_binned = binned( lda0.r111, entropy_111 ) 
    spi_binned = binned( lda0.r111, spi_111 ) 
    Ut_binned   = binned( lda0.r111, lda0.onsite_t_111 )
    U_binned    = binned( lda0.r111, lda0.onsite_t_111 * lda0.tunneling_111 )
    t_binned    = binned( lda0.r111, lda0.tunneling_111 )

    peak_dens = np.amax( density_111 )
    peak_t = np.amin( lda0.tunneling_111 )
    
    axn.fill_between( lda0.r111, dens_binned[0], 0., \
                      lw=2, color='red', facecolor='red', \
                      zorder=2, alpha=0.8)
    axEntr.fill_between( lda0.r111, entr_binned[0], 0., \
                      lw=2, color='red', facecolor='red', \
                      zorder=2, alpha=0.8)
    axSpi.fill_between( lda0.r111, spi_binned[0], 0., \
                      lw=2, color='red', facecolor='red', \
                      zorder=2, alpha=0.8)
    axUt.fill_between( lda0.r111, Ut_binned[0],  0., \
                      lw=2, color='red', facecolor='red', \
                      zorder=2, alpha=0.8  )
    axU.fill_between( lda0.r111, U_binned[0], 0., \
                      lw=2, color='red', facecolor='red',label='$U$', \
                      zorder=2, alpha=0.8) 
    axt.fill_between( lda0.r111, t_binned[0], 0., \
                      lw=2, color='red', facecolor='red',linestyle=':',\
                      label='$t$', zorder=2, alpha=0.8)
                     
       
    
    # Set y labels
    axn.set_ylabel(r'$n$')
    axEntr.set_ylabel(r'$s$')
    axSpi.set_ylabel(r'$S_{\pi}$')
    axnInt.set_ylabel(r'$N_{<R}$')
    axU.set_ylabel(r'$U\,(E_{R})$')
    axt.set_ylabel(r'$t\,(E_{R})$')
    axUt.set_ylabel(r'$U/t$')
    axv0.set_ylabel(r'$E_{R}$')

    # Set y lims 
    n_ylim =  kwargs.get('n_ylim',None)
    if n_ylim is not None: axn.set_ylim( *n_ylim) 
   
    letters = [\
               r'\textbf{a}',\
               r'\textbf{b}',\
               r'\textbf{c}',\
               r'\textbf{d}',\
               r'\textbf{e}',\
               r'\textbf{f}',\
               r'\textbf{g}',\
               r'\textbf{h}',\
              ]
    for i,ax in enumerate([axn, axEntr, axSpi, axnInt, axU, axt, axUt, axv0]):
        ax.text( 0.08,0.86, letters[i] , transform=ax.transAxes, fontsize=14)
        ax.yaxis.grid()
        ax.set_xlabel(r'$\mu\mathrm{m}$')
        for n,r in enumerate(rcut):
            if n % 2 == 0:
                if n == len(rcut)  - 1: 
                    r2 = 60.
                else:
                    r2 = rcut[n+1 ]  
                ax.axvspan( r, r2, facecolor='lightgray') 
                if i != 3:
                    ax.axvspan(-r2, -r, facecolor='lightgray') 
            ax.axvline( r, lw=1.0, color='gray', zorder=1 )
            if i != 3:
                ax.axvline(-r, lw=1.0, color='gray', zorder=1 )
            
        ax.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(20) ) 
        ax.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) )
        
        #labels = [item.get_text() for item in ax.get_xticklabels()]
        #print labels
        #labels = ['' if float(l) % 40 != 0 else l for l in labels ] 
        #ax.set_xticklabels(labels)

    axnInt.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(10) ) 
    axnInt.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(5) )
    
    # Finalize figure
    gs.tight_layout(fig, rect=[0.,0.0,1.0,0.94])

    if kwargs.get('closefig', False):
        plt.close()

    #dens_set = np.array( [ b[0] for b in rbins ] + [dens_binned[1][-1]] ) 
    binresult  = np.column_stack(( 
                     np.round( Nbin, decimals=3),\
                     np.round( nrange1, decimals=3),\
                     np.round( nrange0, decimals=3),\
                     np.round( dens_binned[1], decimals=2),\
                     np.round( t_binned[1], decimals=3),\
                     np.round( U_binned[1], decimals=3),\
                     np.round( Ut_binned[1], decimals=3) ))

    from tabulate import tabulate 
    
    output =  tabulate(binresult, headers=[\
           "Atoms in bin", \
           "n min", \
           "n max", \
           "Mean n", \
           "Mean t", \
           "Mean U", \
           "Mean U/t", ]\
          , tablefmt="orgtbl", floatfmt='.3f')
          #, tablefmt="latex", floatfmt='.3f')

    #print
    #print output
   
    if kwargs.get('return_profile', False):
        return fig, binresult,\
           peak_dens, radius1e, peak_t, output, r111_, density_111
    else:
        return fig, binresult,\
           peak_dens, radius1e, peak_t, output

        
def CheckInhomogSimple( lda0, **kwargs ):
    """This function will make a plot along 111 of the density, U/t
       and T/t 

       It is useful to assess the degree of inhomogeneity in our system"""
    
    # Prepare the figure
    fig = plt.figure(figsize=(9.,4.2))
    lattlabel = '\n'.join(  list( lda0.pot.Info() ) )
    lattlabel = '\n'.join( [ i.split( r'$\mathrm{,}\ $' )[0].replace('s','v') \
                                 for i in lda0.pot.Info() ] )
 
    Nlabel = r'$N=%.2f\times 10^{5}$' % (lda0.Number/1e5)
    Slabel = r'$S/N=%.2fk_{\mathrm{B}}$' % ( lda0.Entropy / lda0.Number )
    thermolabel =  '\n'.join([Nlabel, Slabel])

    ldainfoA = '\n'.join(lda0.Info().split('\n')[:2])
    ldainfoB = '\n'.join(lda0.Info().split('\n')[-2:])

 
    fig.text( 0.05, 0.98,  lattlabel, ha='left', va='top', linespacing=1.2)
    fig.text( 0.48, 0.98,  ldainfoA, ha='right', va='top', linespacing=1.2)
    fig.text( 0.52, 0.98,  ldainfoB, ha='left', va='top', linespacing=1.2)
    fig.text( 0.95, 0.98,  thermolabel, ha='right', va='top', linespacing=1.2)

    gs = matplotlib.gridspec.GridSpec( 1,3, wspace=0.18,\
             left=0.1, right=0.9, bottom=0.05, top=0.98)
    
    # Setup axes
    axn  = fig.add_subplot(gs[0,0])
    axU  = fig.add_subplot(gs[0,1])
    axT  = fig.add_subplot(gs[0,2])


    # Set xlim
    x0 = -40.; x1 = 40.
    axn.set_xlim( x0, x1)

    axU.set_xlim( x0, x1 )
    axU.set_ylim( 0., np.amax( lda0.onsite_t_111 * lda0.tunneling_111 *1.05 ) )
    axT.set_xlim( x0, x1 )
    axT.set_ylim( 0., 1.0)
    
    lw0 = 2.5
    # Plot relevant quantities 
    r111_, density_111 = lda0.getDensity( lda0.globalMu, lda0.T )

    # density,
    axn.plot( lda0.r111, density_111, lw=lw0 , color='black')
    # U
    Ut_111 = lda0.onsite_t_111 
    axU.plot( lda0.r111, Ut_111  , \
                  lw=lw0, label='$U$', color='black') 
    # T
    Tt_111 = lda0.T / lda0.tunneling_111
    axT.plot( lda0.r111, Tt_111, lw=lw0, label='$T$', \
                  color='black')
 
    peak_dens = np.amax( density_111 )
    peak_t = np.amin( lda0.tunneling_111 )
    
    # Set y labels
    axn.set_ylabel(r'$n$')
    axU.set_ylabel(r'$U/t$')
    axT.set_ylabel(r'$T/t$')

    # Set y lims 
    n_ylim =  kwargs.get('n_ylim',None)
    if n_ylim is not None: axn.set_ylim( *n_ylim) 
   
    letters = [\
               r'\textbf{a}',\
               r'\textbf{b}',\
               r'\textbf{c}',\
              ]
    for i,ax in enumerate([axn, axU, axT]):
        ax.text( 0.08,0.86, letters[i] , transform=ax.transAxes, fontsize=14)
        ax.yaxis.grid()
        ax.set_xlabel(r'$\mu\mathrm{m}$')
            
        ax.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(20) ) 
        ax.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(10) )
        
        #labels = [item.get_text() for item in ax.get_xticklabels()]
        #print labels
        #labels = ['' if float(l) % 40 != 0 else l for l in labels ] 
        #ax.set_xticklabels(labels)

    # Finalize figure
    gs.tight_layout(fig, rect=[0.,0.0,1.0,0.94])

    if kwargs.get('closefig', False):
        plt.close()

    if kwargs.get('return_profile', False):
        return fig, peak_dens, peak_t, r111_, density_111, Ut_111 ,Tt_111
    else:
        return fig, peak_dens, peak_t


def CheckInhomogTrap( lda0, **kwargs ):
    """This function will make a plot along 111 of U, t, U/t, v0, W, and W/U 
       (where W is the band gap) 

       It is useful to assess the degree of inhomogeneity in our system"""
    
    # Prepare the figure
    fig = plt.figure(figsize=(8.,4.2))
    lattlabel = '\n'.join(  list( lda0.pot.Info() ) )
    lattlabel = '\n'.join( [ i.split( r'$\mathrm{,}\ $' )[0].replace('s','v') \
                                 for i in lda0.pot.Info() ] )
 

    ldainfoA = '\n'.join(lda0.Info().split('\n')[:2])
    ldainfoB = '\n'.join(lda0.Info().split('\n')[-2:])

 
    fig.text( 0.05, 0.98,  lattlabel, ha='left', va='top', linespacing=1.2)
    fig.text( 0.48, 0.98,  ldainfoA, ha='right', va='top', linespacing=1.2)
    fig.text( 0.52, 0.98,  ldainfoB, ha='left', va='top', linespacing=1.2)

    gs = matplotlib.gridspec.GridSpec( 2,4, wspace=0.18,\
             left=0.1, right=0.9, bottom=0.05, top=0.98)
    
    # Setup axes
    axU    = fig.add_subplot(gs[0,0])
    axt    = fig.add_subplot(gs[0,1])
    ax12t  = fig.add_subplot(gs[0,2])
    axUt   = fig.add_subplot(gs[0,3])
    axv0   = fig.add_subplot(gs[1,0])
    axW    = fig.add_subplot(gs[1,1])
    axWU   = fig.add_subplot(gs[1,2])
    axW12t = fig.add_subplot(gs[1,3])
    axs = [axU, axt, ax12t, axUt, axv0, axW, axWU, axW12t] 

    # Set xlim
    x0 = 0.; x1 = 40.
    for ax in axs:
        ax.set_xlim( x0, x1) 

    # Set y labels
    axU.set_ylabel(r'$U\,(E_{R})$')
    axt.set_ylabel(r'$t\,(\mathrm{kHz})$')
    ax12t.set_ylabel(r'$12t\,(E_{R})$')
    axUt.set_ylabel(r'$U/t$')
    axv0.set_ylabel(r'$v_{0}\,(E_{R})$')
    axW.set_ylabel(r'$W\,(E_{R})$')
    axWU.set_ylabel(r'$W/U$') 
    axW12t.set_ylabel(r'$W/(12t)$') 
 
    #axU.set_ylim( 0., np.amax( lda0.onsite_t_111 * lda0.tunneling_111 *1.05 ) )
    
    lw0 = 2.5



    # U
    U_111 = lda0.onsite_t_111 * lda0.tunneling_111
    axU.plot( lda0.r111, U_111  , \
                  lw=lw0, label='$U/t$', color='black') 
    # t
    t_111 = lda0.tunneling_111 
    axt.plot( lda0.r111, t_111*29., \
                  lw=lw0, label='$t$', color='black') 
    # 12t
    t_111 = lda0.tunneling_111 
    ax12t.plot( lda0.r111, 12.*t_111 , \
                  lw=lw0, label='$t$', color='black') 
    # U/t
    Ut_111 = lda0.onsite_t_111
    axUt.plot( lda0.r111, Ut_111  , \
                  lw=lw0, label='$U$', color='black') 
    # v0 
    V0_111 = lda0.pot.S0( lda0.X111, lda0.Y111, lda0.Z111 ) 
    axv0.plot( lda0.r111, V0_111[0], lw=lw0, color='black', \
               label='$\mathrm{Lattice\ depth}$')
    # Band gap 
    bandgap_111 = bands = scubic.bands3dvec( V0_111, NBand=1 )[0] \
                          - scubic.bands3dvec( V0_111, NBand=0 )[1] 
    axW.plot( lda0.r111, bandgap_111, lw=lw0, color='black', \
                label='$\mathrm{Band\ gap},\,W$')   
    # Band gap / U 
    axWU.plot( lda0.r111, bandgap_111 / U_111, lw=lw0, color='black', \
                label='$W/U$')   
    # Band gap / 12t 
    axW12t.plot( lda0.r111, bandgap_111 / (12.*t_111), lw=lw0, color='black', \
                label='$W/(12t)$')   


    letters = [\
               r'\textbf{a}',\
               r'\textbf{b}',\
               r'\textbf{c}',\
               r'\textbf{d}',\
               r'\textbf{e}',\
               r'\textbf{f}',\
              ]
    for i,ax in enumerate(axs):
        #ax.text( 0.08,0.86, letters[i] , transform=ax.transAxes, fontsize=14)
        ax.yaxis.grid()
        ax.set_xlabel(r'$\mu\mathrm{m}$')
            
        ax.xaxis.set_major_locator( matplotlib.ticker.MultipleLocator(10) ) 
        ax.xaxis.set_minor_locator( matplotlib.ticker.MultipleLocator(5) )
        
        #labels = [item.get_text() for item in ax.get_xticklabels()]
        #print labels
        #labels = ['' if float(l) % 40 != 0 else l for l in labels ] 
        #ax.set_xticklabels(labels)

    # Finalize figure
    gs.tight_layout(fig, rect=[0.,0.0,1.0,0.94])

    if kwargs.get('closefig', False):
        plt.close()

    return fig 
