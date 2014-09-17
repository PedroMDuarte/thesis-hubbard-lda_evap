

""" 
Potentials in ultracold atom experiments are typically constructed using
Gaussian laser beams, here we provide some definitions that will make it easy
to assemble a generic optical dipole potential. 

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from vec3 import vec3, cross
import scipy.constants as C 
from mpl_toolkits.mplot3d import axes3d


def beam(xb,yb,zb,wx,wy,wavelen):
    """ 
    Normalized intensity profile of a Gaussian beam which propagates along z and 
    has its waist at the origin 

    Parameters 
    ----------
    xb, yb, zb  :  These can be single floats, or can be array-like for full 
                   vectorization.   

 
    Returns 
    -------
    intensity   :  The intensity of the gaussian beam.  Normalized, so it is 
                   equal to 1. at the origin.


    Notes 
    ----  


    Examples
    --------
    """

    zRx = np.pi * wx**2 / wavelen
    zRy = np.pi * wy**2 / wavelen 
    
    sqrtX = np.sqrt( 1 + np.power(zb/zRx,2) ) 
    sqrtY = np.sqrt( 1 + np.power(zb/zRy,2) ) 
    intensity = np.exp( -2.*( np.power(xb/(wx*sqrtX ),2) \
                            + np.power(yb/(wy*sqrtY),2) )) / sqrtX / sqrtY
    return intensity 


def uL( wavelen, **kwargs ):
    """ 
    Calculates the factor uL which allows conversion from intensity to depth
    in microKelvin.  

    Parameters 
    ----------
    wavelen     :  wavelength of the light used to create the dipole potential. 
 
    Returns 
    -------
    uL          :  intensity to uK factor

    Notes 
    ----  

    Examples
    --------
    """
    Cc = C.c * 1e6 # speed of light um s^-1

    Gamma   = kwargs.get('Gamma',  2*np.pi *5.9e6 )  # linewidth s^-1
    lambda0 = kwargs.get('lambda0', 0.671 )  # transition wavelength in microns  
    
    omega0  = 2*np.pi*Cc / lambda0
    omegaL  = 2*np.pi*Cc / wavelen
    intensity = 1.0 
    depthJ  = (intensity)* -3*np.pi* Cc**2*Gamma / ( 2*omega0**3) * \
              ( 1/(omega0 - omegaL )  + 1/(omega0 + omegaL ) ) # Joule
    depthuK = depthJ / C.k  *1e6 # C.k is Boltzmann's constant
    return depthuK



def Erecoil( wavelen, mass):
    """ 
    Calculates the recoil energy in microKelvin for a given photon wavelength 
    and atom mass  

    Parameters 
    ----------
    wavelen     :  wavelength of the photon in microns
    mass        :  mass of the atom in atomic mass units
 
    Returns 
    -------
    Er          :  recoil energy in microKelvin

    Notes 
    ----  

    Examples
    --------
    """
    inJ  = C.h**2 / ( 2* \
            mass*C.physical_constants['unified atomic mass unit'][0] * \
           (wavelen*1e-6)**2 ) 
    inuK = inJ / C.k *1e6
    return inuK
    
                  

class GaussBeam:
    """
    This class defines the potential created for a Gaussian beam.  The 
    initialization parameters can be given to the class as keyword arguments 
    (kwargs)  


    Parameters 
    ----------

    mW         :   float.  Power in the beam in milliWatts 

    waists     :   tuple.  ( waistx, waisty )   

    wavelength :   float.  wavelength of the light 

    axis       :   tuple.  ( theta, pi )  polar coordinates specifying 
                   direction of propagation of the beam 

    origin     :   tuple.  ( x, y, z )  cartesian coordinates spicifying the
                   location of the beam waist 

    """ 
    def __init__( self,
                 **kwargs ):
        self.mW = kwargs.get('mW',1000.0 )
        self.w  = kwargs.get('waists', (30.,30.) )
        self.l  = kwargs.get('wavelength', 1.064 )
        #
        self.axis   = kwargs.get('axis', (np.pi/2,0.) )
        self.origin = kwargs.get('origin', vec3(0,0,0) )
        
        
        # Make sure vectors are of type(vec3)
        self.axisvec = vec3()
        th = self.axis[0]
        ph = self.axis[1]
        self.axisvec.set_spherical( 1, th, ph) 
        self.origin = vec3(self.origin)
        
        # Calculate two orthogonal directions 
        # which will be used to specify the beam waists
        self.orth0 = vec3( np.cos(th)*np.cos(ph) , \
                           np.cos(th)*np.sin(ph), -1.*np.sin(th) )
        self.orth1 = vec3( -1.*np.sin(ph), np.cos(ph), 0. )
        
    def transform(self, X, Y, Z):
        # coordinates into beam coordinates 
        zb = X*self.axisvec[0] + Y*self.axisvec[1] + Z*self.axisvec[2]
        xb = X*self.orth0[0]   + Y*self.orth0[1]   + Z*self.orth0[2]
        yb = X*self.orth1[0]   + Y*self.orth1[1]   + Z*self.orth1[2]
        return xb,yb,zb
        
    def __call__( self, X, Y, Z):
        """ 
        Returns the depth in microKelvin of the potential produced by the
        Gaussian beam.  

        Parameters 
        ----------
        X, Y, Z     :  can be floats or array-like. The potential is calculated
                       in a vectorized way.  

        Returns 
        -------
        potential in microKelvin

        Notes 
        ----  

        Examples
        --------
        """
        xb,yb,zb = self.transform( X,Y,Z)
        
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] *gauss  # W um^-2
        
        return uL(self.l)*intensity


        
class LatticeBeam(GaussBeam):
    """
    This class defines the lattice potential created by two retroreflected 
    Gaussian beams.   
    
    The initialization parameters can be given to the class as keyword 
    arguments (kwargs).

    It is assumed that the input beam and retro beam have the same beam waists 


    Parameters 
    ----------
    waists     :   tuple.  ( waistx, waisty )   

    wavelength :   float.  wavelength of the light 

    axis       :   tuple.  ( theta, pi )  polar coordinates specifying 
                   direction of propagation of the beam 

    origin     :   tuple.  ( x, y, z )  cartesian coordinates spicifying the
                   location of the beam waist 

    s0         :   float.  The lattice depth at the waist in units of the 
                   recoil energy Er

    scale      :   The periodicity of the lattice potential is increased by 
                   this scale, for visibility when plotting it.  This does not
                   affect other results, just the plotting. 
  
    mass       :   float.  The mass of the atom in atomic mass units. 

    retro      :   The retro factor.  This is the percentage of power that is
                   retroreflected.   The losses on the retro-reflection amount
                   to   losses =  1 - retro .   

    alpha      :   Used to specify the amount of the retro beam that can 
                   interfere with the input beam to form a lattice.  
                   If alpha=1 all of the retro beam intereres, if alpha = 0 
                   none of the retro beam interferes.  
                   
    """ 
    def __init__(self, **kwargs):
        """Lattice beam, with retro factor and polarization """
        GaussBeam.__init__(self, **kwargs)
        self.scale = kwargs.get('scale',10.)
        self.mass  = kwargs.get('mass', 6.0)
        self.s0    = kwargs.get('s0', 7.0)
        self.retro = kwargs.get('retro', 1.0)
        self.alpha = kwargs.get('alpha', 1.0)
        self.Er0  = Erecoil( self.l , self.mass)  
        self.mW = 1000 * (self.s0 * self.Er0 ) \
                 * np.abs( np.pi / 8. / uL(self.l) )\
                 * self.w[0]*self.w[1] / self.retro 
        
        
    def __call__( self, X, Y, Z):
        """ 
        Returns the lattice potential in microKelvin.

        Parameters 
        ----------
        X, Y, Z     :  can be floats or array-like. The potential is calculated
                       in a vectorized way.  

        Returns 
        -------
        lattice potential in microKelvin

        Notes 
        ----  

        Examples
        --------
        """
        xb,yb,zb = self.transform( X,Y,Z)
         
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] *gauss  # W um^-2

        cosSq = np.power(np.cos(2*np.pi/self.l * zb/self.scale ),2)
        
        lattice =  cosSq *4*np.sqrt(self.retro*self.alpha)\
                 + ( 1 + self.retro - 2*np.sqrt(self.retro*self.alpha) )
        
        return uL(self.l)*intensity*lattice
    
    def getBottom( self, X, Y, Z):
        """ 
        Returns the envelope of the lattice potential in microKelvin.

        Parameters 
        ----------
        X, Y, Z     :  can be floats or array-like. The potential is calculated
                       in a vectorized way.  

        Returns 
        -------
        envelope of the lattice potential in microKelvin

        Notes 
        ----  

        Examples
        --------
        """
        xb,yb,zb = self.transform( X,Y,Z)
        
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] *gauss  # W um^-2
        
        latticeBot = 4*np.sqrt(self.retro*self.alpha)  \
             + 1 + self.retro - 2*np.sqrt(self.retro*self.alpha)

        return uL(self.l)*intensity * latticeBot
    
    def getS0( self, X, Y, Z):
        """ 
        Returns the lattice depth in microKelvin 

        Parameters 
        ----------
        X, Y, Z     :  can be floats or array-like. The potential is calculated
                       in a vectorized way.  

        Returns 
        -------
        lattice depth in microKelvin

        Notes 
        ----  

        Examples
        --------
        """
        xb,yb,zb = self.transform( X,Y,Z)
        
        gauss = beam( xb,yb,zb, self.w[0], self.w[1], self.l)
        intensity = (2/np.pi)* self.mW/1000. /self.w[0]/self.w[1] \
                    * gauss  # W um^-2
    
        latticeV0  = 4*np.sqrt(self.retro*self.alpha) 
        return np.abs(uL(self.l)*intensity * latticeV0)


class potential:
    """
    A potential is defined as a collection of beams that do not interfere 
    with each other.  

    The sum of the potential crated by each beam is the total potential. 

    Parameters 
    ----------
    units      :   tuple, two elements.  
                   - First element is the string which will be used for 
                     labeling plots.  
                   - Second element is the multiplication factor required to 
                     obtain the desired units. Beams are by default in 
                     microKelvin.   
  
    beams      :   list,  this is the list of beams that makes up the 
                   potential

    """ 
    def __init__(self, beams, **kwargs ):
        self.units = kwargs.get('units', ('$\mu\mathrm{K}$', 1.))
        self.unitlabel = self.units[0]
        self.unitfactor = self.units[1] 
        self.beams = beams 
        
    def evalpotential( self, X, Y, Z):
        """ 
        Evaluates the total potential by summing over beams  

        Parameters 
        ----------
        X, Y, Z     :  can be floats or array-like. The potential is calculated
                       in a vectorized way.  

        Returns 
        -------
        total potential.  The units used depend on self.unitfactor.  

        Notes 
        ----  

        Examples
        --------
        """
        EVAL = np.zeros_like(X) 
        for b in self.beams:
            EVAL += b(X,Y,Z)
        return EVAL* self.unitfactor


""" 
Below we include functions to make cuts through the geometry.  These can be
line cuts or plane cuts. 
""" 

def linecut_points( **kwargs ):
    """ 
    Defines an line cut through the potential geometry.  Parameters are given
    as keyword arguments (kwargs).

    All distances are in microns. 

    Parameters 
    ----------
    npoints     :  number of points along the cut
 
    extents     :  a way of specifying the limits for a cut that is symmetric
                   about the cut origin.  the limits will be 
                   lims = (-extents, extents)  

    lims        :  used only if extents = None.   limits are specified using
                   a tuple  ( min, max ) 

    direc       :  tuple, two elements.  polar coordinates for the direcdtion 
                   of the cut 
 
    origing     :  tuple, three elements.  cartesian coordinates for the origin
                   of the cut  


    Returns 
    -------
    t           :  array which parameterizes the distance along the cut   
  
    X, Y, Z     :  each of X,Y,Z is an array with the same shape as t. 
                   They correspond to the cartesian coordinates of all the 
                   points along the cut 
                   
    Notes 
    ----  

    Examples
    --------
    """
    npoints = kwargs.get('npoints', 320)
    extents = kwargs.get('extents',None)
    lims    = kwargs.get('lims', (-80.,80.))
    direc   = kwargs.get('direc', (np.pi/2, 0.))
    origin  = kwargs.get('origin', vec3(0.,0.,0.))

    if extents is not None:
        lims = (-extents, extents)

    # Prepare set of points for plot 
    t = np.linspace( lims[0], lims[1], npoints )
    unit = vec3()
    th = direc[0]
    ph = direc[1] 
    unit.set_spherical(1, th, ph) 
    # Convert vec3s to ndarray
    unit = np.array(unit)
    origin = np.array(origin) 
    #
    XYZ = origin + np.outer(t, unit)
    X = XYZ[:,0]
    Y = XYZ[:,1]
    Z = XYZ[:,2]
 
    return t, X, Y, Z, lims



    
def surfcut_points(**kwargs):
    """ 
    Defines an surface cut through the potential geometry.  Parameters are given
    as keyword arguments (kwargs).

    All distances are in microns. 

    Parameters 
    ----------
    npoints     :  number of points along the cut
 
    extents     :  a way of specifying the limits for a cut that is symmetric
                   about the cut origin.  the limits will be 
                   lims = (-extents, extents)  

    lims        :  used only if extents = None.   limits are specified using
                   a tuple  ( min, max ) 

    direc       :  tuple, two elements.  polar coordinates for the direcdtion 
                   of the cut 
 
    origin      :  tuple, three elements.  cartesian coordinates for the origin
                   of the cut  

    ax0         :  optional axes where the reference surface for the surface 
                   cut can be plotted 


    Returns 
    -------
    T0, T1      :  arrays which parameterizes the position on the cut surface
  
    X, Y, Z     :  each of X,Y,Z is an array with the same shape as T0 and T1. 
                   They correspond to the cartesian coordinates of all the 
                   points on the cut surface.  
                   
    Notes 
    ----  

    Examples
    --------
    """
    npoints = kwargs.get( 'npoints', 240  )
    origin = kwargs.get( 'origin', vec3(0.,0.,0.)) 
    normal = kwargs.get( 'normal', (np.pi/2., 0.) )  
    lims0  = kwargs.get( 'lims0', (-50., 50.) ) 
    lims1  = kwargs.get( 'lims1', (-50., 50.) ) 
    extents = kwargs.get( 'extents', None)  
    
    if extents is not None:
        lims0 = (-extents, extents)
        lims1 = (-extents, extents)
    
    # Make the unit vectors that define the plane
    unit = vec3()
    th = normal[0]
    ph = normal[1]
    unit.set_spherical( 1, th, ph) 
    orth0 = vec3( -1.*np.sin(ph), np.cos(ph), 0. )
    orth1 = cross(unit,orth0)
    
    t0 = np.linspace( lims0[0], lims0[1], npoints )
    t1 = np.linspace( lims1[0], lims1[1], npoints ) 
    
    # Obtain points on which function will be evaluated
    T0,T1 = np.meshgrid(t0,t1)
    X = origin[0] + T0*orth0[0] + T1*orth1[0] 
    Y = origin[1] + T0*orth0[1] + T1*orth1[1]
    Z = origin[2] + T0*orth0[2] + T1*orth1[2] 
    

    # If given an axes it will plot the reference surface to help visusalize
    # the surface cut
    
    # Note that the axes needs to be created with a 3d projection. 
    # For example: 
    #    fig = plt.figure( figsize=(4.,4.) ) 
    #    gs = matplotlib.gridspec.GridSpec( 1,1 ) 
    #    ax0 = fig.add_subplot( gs[0,0], projection='3d' )  
 
    ax0 = kwargs.get( 'ax0', None ) 
    if ax0 is not None: 

        # Plot the reference surface
        ax0.plot_surface(X, Y, Z, rstride=8, cstride=8, alpha=0.3, linewidth=0.)
        ax0.set_xlabel('X')
        ax0.set_ylabel('Y')
        ax0.set_zlabel('Z')
        lmin = min([ ax0.get_xlim()[0], ax0.get_ylim()[0], ax0.get_zlim()[0] ] )
        lmax = max([ ax0.get_xlim()[1], ax0.get_ylim()[1], ax0.get_zlim()[1] ] )
        ax0.set_xlim( lmin, lmax )
        ax0.set_ylim( lmin, lmax )
        ax0.set_zlim( lmin, lmax )
        ax0.set_yticklabels([])
        ax0.set_xticklabels([])
        ax0.set_zticklabels([])
    
    # If given an axes and a potential it will plot the surface cut of the 
    # potential   

    ax1 = kwargs.get( 'ax1', None) 
    pot = kwargs.get( 'potential', None)  

    if (ax1 is not None) and (pot is not None):
        # Evaluate function at points and plot
        EVAL = pot.evalpotential(X,Y,Z)

        im =ax1.pcolormesh(T0, T1, EVAL, cmap = plt.get_cmap('jet')) 
        # cmaps:  rainbow, jet

        plt.axes( ax1)
        cbar = plt.colorbar(im)
        cbar.set_label(pot.unitlabel, rotation=0 )#self.unitlabel
    
    return T0, T1, X, Y, Z  
        
        

def plot3surface( pot, **kwargs ): 
    """
    This is a packaged function to quickly plot a potential along
    three orthogonal planes that intersecdt at the origin. 

    Parameters 
    ----------
    pot     :  potential to be plotted 
 
    Returns 
    -------
                   
    Notes 
    ----  

    Examples
    --------
    """   
        
    fig = plt.figure( figsize = (8., 8.) ) 
    gs = matplotlib.gridspec.GridSpec( 3,2, wspace=0.2) 
    
    # Make a list with three perpendicular directions which 
    # will define the three surface cuts 
    perp = [(np.pi/2., 0.), (np.pi/2., -np.pi/2.), (0., -1.*np.pi/2.) ]
    
    # Iterate to plot the three surface cuts
    yMin = 1e16
    yMax = -1e16 
    Ims = []
    for i in range(3):
        ax0 = fig.add_subplot( gs[i,0], projection='3d')
        ax1 = fig.add_subplot( gs[i,1])  
        
        T0, T1, X, Y, Z = surfcut_points( normal = perp[i], \
                                          ax0=ax0, **kwargs ) 
        
        EVAL = pot.evalpotential(X,Y,Z)
        im = ax1.pcolormesh( T0, T1, EVAL, \
                       cmap=plt.get_cmap('jet') ) 
        plt.axes( ax1 )  
        cbar = plt.colorbar(im)
        cbar.set_label( pot.unitlabel, rotation=0)  
        
        ymin = EVAL.min()
        ymax = EVAL.max()
        
        Ims.append(im) 
        if ymin < yMin : yMin = ymin
        if ymax > yMax : yMax = ymax 
        
    for im in Ims:
        im.set_clim( vmin=yMin, vmax=yMax)             
        
        
        
        
        
        
        
        
        
    
