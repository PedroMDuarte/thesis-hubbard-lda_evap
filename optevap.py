
import scubic 
import lda
from scipy.optimize import minimize_scalar, brentq
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from matplotlib import rc
rc('font',**{'family':'serif'})
rc('text', usetex=True)

 

def optimal( **kwargs ) :
    """ 
    This function takes fixed values of s0, wL, wC and optimizes the 
    evaporation figure of merit, eta_F,  by using the green compensation
    as a variable. 
    """

    s0 = kwargs.get('s0', 7. ) 
    wL = kwargs.get('wL', 47. )
    wC = kwargs.get('wC', 40. )
    alpha = wL/wC
    a_s = kwargs.get('a_s', 650. )
    T_Er= kwargs.get('T_Er', 0.2 )
    extents = kwargs.get('extents', 40.)

    if 'Number' in kwargs.keys():  
        N0 = kwargs['Number']  

    def Npenalty( Num ):
        p = 4.
        if Num > N0: 
            return np.exp( (Num - N0)/1e5 )**p 
        else:
            return 1. 

    def penalty(x,p):
        """
        This function is used to penalyze  EtaF < 1 , which amounts to 
        spilling out along the lattice beams.
        """ 
        if x < 1.:
            return np.exp(-(x-1.))**p
        else:
            return x 
    
        #return np.piecewise(x, [x < 1., x >= 1.], \
        #           [lambda x: , lambda x: x])        

    def merit( g0 ) :
        try:
            pot = scubic.sc(allIR  = s0, \
                        allGR  = g0, \
                        allIRw = wL, \
                        allGRw = wC )

            # The lda within the optimization loop is told to ignore errors
            # of the density distribution going beyond the extents.  
            # This will be checked after the optmization is done.
            lda0 = lda.lda(potential = pot, Temperature = T_Er,\
                            a_s=a_s, globalMu='halfMott', extents=extents,\
                            ignoreExtents=True, select='htse' )
    
            etaFstar = penalty( lda0.etaF_star , 5 ) 
            if 'Number' in kwargs.keys():
                return  etaFstar * Npenalty( lda0.Number) 
            else:
                return  etaFstar
        except Exception as e :
            negslope = 'Bottom of the band has a negative slope'
            posslope = 'Radial density profile along 111 has a positive slope'
            threshol = 'Chemical potential exceeds the evaporation threshold'
            thresh100= 'Chemical potential exceeds the bottom of the band along 100'

            if negslope in e.message:
                return 1e4 # this is caused by too much green 
                           # return large value to asign penalty
            elif posslope in e.message:
                return 1e4 # this is caused by too much green
                           # as the chemical potential comes to close
                           # to the threshold and atoms accumulate on 
                           # the beams 
                           # return large value to asign penalty
            elif threshol in e.message:
                return 1e4 # this is caused by too much green
                           # the chemical potential is above the evap th. 
                           # return large value to asign penalty
            elif thresh100 in e.message:
                return 1e4 # this is caused by too much green
                           # the chemical potential is above the bottom of 
                           # the band along 100 
                           # return large value to asign penalty
                              
 
            elif 'vanish' in e.message : 
                # this is is caused by insufficient extents
                print "extents = %.1f"%  extents  
                raise 
            else:
                raise 
            #print "Fail at g0=%.2f"% g0 
            #raise

    g0bounds =  (0., min(s0,s0/(alpha**2.)))   
    res = minimize_scalar( merit, bounds=g0bounds, tol=4e-2, \
              method='bounded' )
    gOptimal =  res.x 

    #print "gOpt=%.2f"%gOptimal

    potOpt = scubic.sc( allIR=s0, allGR=gOptimal, allIRw=wL, allGRw=wC ) 
    ldaOpt = lda.lda( potential = potOpt, Temperature=T_Er, \
                      a_s=a_s, globalMu='halfMott', extents=extents)  
    return [ gOptimal, ldaOpt.EtaEvap, ldaOpt.Number, \
             ldaOpt.Entropy/ldaOpt.Number, ldaOpt.getRadius(), ldaOpt.getRadius()/wL,  ldaOpt.DeltaEvap ]



def get_trap_results( **kwargs  ):
    """
    If the parameters for the trap are known, the trap results can be obtained 
    directly with this function 
    """
    s0 = kwargs.get('s0', 7. ) 
    wL = kwargs.get('wL', 47. )
    wC = kwargs.get('wC', 40. )
    alpha = wL/wC
    a_s = kwargs.get('a_s', 650. )
    T_Er= kwargs.get('T_Er', 0.2 )
    extents = kwargs.get('extents', 40.)
  
    gOptimal = kwargs.get('g0',4.304)
    
    potOpt = scubic.sc( allIR=s0, allGR=gOptimal, allIRw=wL, allGRw=wC ) 
    ldaOpt = lda.lda( potential = potOpt, Temperature=T_Er, \
                      a_s=a_s, globalMu='halfMott', extents=extents)  
    return [ gOptimal, ldaOpt.EtaEvap, ldaOpt.Number, \
             ldaOpt.Entropy/ldaOpt.Number, ldaOpt.getRadius(), ldaOpt.getRadius()/wL,  ldaOpt.DeltaEvap ]


def optimal_FixedRadius( **kwargs ) :
    """ 
    This function takes fixed values of s0, wL, wC and finds the value of
    green that would be required to make a sample with a radius that is
    a fixed fraction of the lattice beam waist. 
 
    The value of the fraction is hardcoded in the function
    """
    fraction = 0.32

    s0 = kwargs.get('s0', 7. ) 
    wL = kwargs.get('wL', 47. )
    wC = kwargs.get('wC', 40. )
    alpha = wL/wC
    a_s = kwargs.get('a_s', 650. )
    T_Er= kwargs.get('T_Er', 0.2 )
    extents = kwargs.get('extents', 40.)

    def merit( g0 ) :
        try:
            pot = scubic.sc(allIR  = s0, \
                        allGR  = g0, \
                        allIRw = wL, \
                        allGRw = wC )

            # The lda within the optimization loop is told to ignore errors
            # of the density distribution going beyond the extents.  
            # This will be checked after the optmization is done.
            lda0 = lda.lda(potential = pot, Temperature = T_Er,\
                            a_s=a_s, globalMu='halfMott', extents=extents,\
                            ignoreExtents=True )

            return   (fraction*wL -   lda0.getRadius())**2.
            #return   fraction -   lda0.getRadius()/wL
        except Exception as e :
            negslope = 'Bottom of the band has a negative slope'
            posslope = 'Radial density profile along 111 has a positive slope'
            threshol = 'Chemical potential exceeds the evaporation threshold'
            thresh100= 'Chemical potential exceeds the bottom of the band along 100'

            if negslope in e.message:
                return 1e6 # this is caused by too much green 
                           # return large value to asign penalty
            elif posslope in e.message:
                return 1e6 # this is caused by too much green
                           # as the chemical potential comes to close
                           # to the threshold and atoms accumulate on 
                           # the beams 
                           # return large value to asign penalty
            elif threshol in e.message:
                return 1e6 # this is caused by too much green
                           # the chemical potential is above the evap th. 
                           # return large value to asign penalty
            elif thresh100 in e.message:
                return 1e6 # this is caused by too much green
                           # the chemical potential is above the bottom of 
                           # the band along 100 
                           # return large value to asign penalty
                              
 
            elif 'vanish' in e.message : 
                # this is is caused by insufficient extents
                print "extents = %.1f"%  extents  
                raise 
            else:
                raise 
            #print "Fail at g0=%.2f"% g0 
            #raise

    g0bounds =  (1., min(s0, (4.*s0-2.*np.sqrt(s0))/(4.*(alpha**2.)) ) ) 
 

    
    #(x, res) = brentq( merit, g0bounds[0], g0bounds[1] )  
    #gOptimal =  x
 
    res = minimize_scalar( merit, bounds=g0bounds, tol=1e-4, \
              method='bounded' )
    gOptimal =  res.x 

    #print "gOpt=%.2f"%gOptimal

    potOpt = scubic.sc( allIR=s0, allGR=gOptimal, allIRw=wL, allGRw=wC ) 
    ldaOpt = lda.lda( potential = potOpt, Temperature=T_Er, \
                      a_s=a_s, globalMu='halfMott', extents=extents)  
    return [ gOptimal, ldaOpt.EtaEvap, ldaOpt.Number, \
             ldaOpt.Entropy/ldaOpt.Number, ldaOpt.getRadius(), ldaOpt.getRadius()/wL ]



def meshplot( ax, results, i, j, k, contours = None, dashed=None, base=1., **kwargs):
    x = results[:,i]
    y = results[:,j]
    z = results[:,k]/base
    xi = np.linspace( x.min(), x.max(), 300)
    yi = np.linspace( y.min(), y.max(), 300)
    zq = matplotlib.mlab.griddata(x, y, z, xi,yi)
    vlim = kwargs.get('vlim',None)
    if vlim is None:
        im0 =ax.pcolormesh( xi, yi, zq , cmap = plt.get_cmap('rainbow'))
    else:
        im0 =ax.pcolormesh( xi, yi, zq , cmap = plt.get_cmap('rainbow'),\
                vmin=vlim[0], vmax=vlim[1])
    plt.axes( ax)
    plt.colorbar(im0) 

    x0,x1 = kwargs.get('xlim',(44.,80.))
    y0,y1 = kwargs.get('ylim',(28.,71.))

    #ax.axhline( 40., lw=3., color='lightgray', alpha=0.30)
    #ax.axvline( 47., lw=3., color='lightgray', alpha=0.30)
    
    if contours is not None:
        c0 = ax.contour(xi, yi, zq, contours, linewidths = 0.5, colors = 'k')
        plt.clabel(c0, inline=1, fontsize=8)
    if dashed is not None:
        for m in dashed:
            dx  = kwargs.get('dx', 3.8) 
            xm  = kwargs.get('xm',55. )
            x = np.linspace( x0, xm-dx, 100)
            ax.plot( x, x/m['slope'], '--',lw=1.5, color=m['c'], alpha=1.0)
            x = np.linspace( xm+dx, x1, 100)
            ax.plot( x, x/m['slope'], '--',lw=1.5, color=m['c'], alpha=1.0)
            ax.text( xm, xm/m['slope'], r'$\alpha_{w}=$%.2f'%m['slope'], \
                     ha='center', va='center', color=m['c'], fontsize=13, \
                     rotation = np.arctan( m['slope']*26./44.)*180./np.pi - \
                                (m['slope']-1.)*60. )

 
    blackout = kwargs.get('blackout', None)
    if blackout is not None:
        x = np.linspace( x0, x1, 100)
        ax.fill_between( x, x/blackout[0], x/blackout[1], color='white' ) 
        
 
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)

def plotOptSingle( datfile, k, **kwargs): 
    results = np.loadtxt( datfile )

    dashed = kwargs.pop('dashed',[{'slope':1.0,'c':'black'}]) 
    contours = kwargs.pop('contours', [1.87,2.8, 4.4, 6. ])

    x0,x1 = kwargs.get('xlim',(44.,80.))
    y0,y1 = kwargs.get('ylim',(28.,71.))
    fig = plt.figure(figsize=(4.,3.25))
    gs  = matplotlib.gridspec.GridSpec(1,1, wspace=0.2, hspace=0.3,\
            left=0.14, right=0.96, bottom=0.14, top=0.90)

    ax  = fig.add_subplot( gs[0,0] )
    i=1; j=2;
    meshplot( ax, results, i, j, k, contours=contours, dashed=dashed, \
               **kwargs )     
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1) 
    title = kwargs.get('title', None) 
    ax.set_title(title, fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ beam\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ beam\ waist}\ w_{C}\ (\mu\mathrm{m})$')
    return fig  

def plotOptimal( datfile, **kwargs ) : 
 
    results = np.loadtxt( datfile )
 
    dashed_eta = kwargs.get('dashed_eta',[{'slope':1.0,'c':'black'}] ) 
    dashed_g0 = kwargs.get('dashed_g0',[{'slope':1.0,'c':'black'}] ) 
    dashed_sn = kwargs.get('dashed_sn',[{'slope':1.0,'c':'black'}] ) 
    dashed_num = kwargs.get('dashed_num',[{'slope':1.0,'c':'black'}] ) 
    x0,x1 = kwargs.get('xlim',(44.,80.))
    y0,y1 = kwargs.get('ylim',(28.,71.))
    
        
    fig = plt.figure(figsize=(8.,6.5))
    gs  = matplotlib.gridspec.GridSpec(2,2, wspace=0.2, hspace=0.3,\
            left=0.07, right=0.98, bottom=0.08, top=0.94)
    
    
    eta_contours = kwargs.get('eta_contours', [1.87,2.8, 4.4, 6. ])
    ax  = fig.add_subplot( gs[0,0] )
    i=1; j=2; k=4
    meshplot( ax,results, i, j, k, contours = eta_contours, \
              dashed=dashed_eta , **kwargs)     
    ax.set_title('$\eta_{F}$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ beam\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ beam\ waist}\ w_{C}\ (\mu\mathrm{m})$')

    g0_contours = kwargs.get('g0_contours', [1.87,2.8, 4.4, 6. ])
    ax  = fig.add_subplot( gs[0,1] )
    i=1; j=2; k=3
    meshplot( ax,results, i, j, k, contours = g0_contours, \
              dashed=dashed_g0, **kwargs )     
    ax.set_title('$g_{0}$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ beam\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ beam\ waist}\ w_{C}\ (\mu\mathrm{m})$')
   
    sn_contours = kwargs.get('sn_contours', [1.2, 1.4, 2.4, 3.] ) 
    ax  = fig.add_subplot( gs[1,0] )
    i=1; j=2; k=6
    meshplot( ax,results, i, j, k, contours = sn_contours, \
              dashed=dashed_sn, **kwargs )     
    ax.set_title('$S/N$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ waist}\ w_{C}\ (\mu\mathrm{m})$')
    
    num_contours = kwargs.get('num_contours', [1.0, 2.0, 3.4, 4.8, 5.8])
    ax  = fig.add_subplot( gs[1,1] )
    i=1; j=2;  k=5 
    meshplot( ax,results, i, j, k, contours = num_contours, \
              dashed=dashed_num, base=1e5, **kwargs )     
    ax.set_title('$N/10^{5}$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ waist}\ w_{C}\ (\mu\mathrm{m})$')
   
    return fig  
    

def plotOptimal_6( datfile, **kwargs ) : 
 
    results = np.loadtxt( datfile )
 
    dashed_eta = kwargs.get('dashed_eta',[{'slope':1.0,'c':'black'}] ) 
    dashed_delta = kwargs.get('dashed_delta',[{'slope':1.0,'c':'black'}] ) 
    dashed_g0 = kwargs.get('dashed_g0',[{'slope':1.0,'c':'black'}] ) 
    dashed_sn = kwargs.get('dashed_sn',[{'slope':1.0,'c':'black'}] ) 
    dashed_num = kwargs.get('dashed_num',[{'slope':1.0,'c':'black'}] ) 
    dashed_hwhm = kwargs.get('dashed_hwhm',[{'slope':1.0,'c':'black'}] ) 
    x0,x1 = kwargs.get('xlim',(44.,80.))
    y0,y1 = kwargs.get('ylim',(28.,71.))
    
        
    fig = plt.figure(figsize=(12.,6.5))
    gs  = matplotlib.gridspec.GridSpec(2,3, wspace=0.2, hspace=0.3,\
            left=0.07, right=0.98, bottom=0.08, top=0.94)
    
    
    eta_contours = kwargs.get('eta_contours', [1.87,2.8, 4.4, 6. ])
    ax  = fig.add_subplot( gs[0,0] )
    i=1; j=2; k=4
    meshplot( ax,results, i, j, k, contours = eta_contours, \
              dashed=dashed_eta , **kwargs)     
    ax.set_title('$\eta_{F}$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ beam\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ beam\ waist}\ w_{C}\ (\mu\mathrm{m})$')

    delta_contours = kwargs.get('delta_contours', [1.87,2.8, 4.4, 6. ])
    ax  = fig.add_subplot( gs[0,1] )
    i=1; j=2; k=9
    meshplot( ax,results, i, j, k, contours = delta_contours, \
              dashed=dashed_delta , **kwargs)     
    ax.set_title('$\Delta_{F}\,(E_{R})$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ beam\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ beam\ waist}\ w_{C}\ (\mu\mathrm{m})$')

    g0_contours = kwargs.get('g0_contours', [1.87,2.8, 4.4, 6. ])
    ax  = fig.add_subplot( gs[0,2] )
    i=1; j=2; k=3
    meshplot( ax,results, i, j, k, contours = g0_contours, \
              dashed=dashed_g0, **kwargs )     
    ax.set_title('$g_{0}\,(E_{R})$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ beam\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ beam\ waist}\ w_{C}\ (\mu\mathrm{m})$')
   
    sn_contours = kwargs.get('sn_contours', [1.2, 1.4, 2.4, 3.] ) 
    ax  = fig.add_subplot( gs[1,0] )
    i=1; j=2; k=6
    meshplot( ax,results, i, j, k, contours = sn_contours, \
              dashed=dashed_sn, **kwargs )     
    ax.set_title('$S/N\,(k_{\mathrm{B}})$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ waist}\ w_{C}\ (\mu\mathrm{m})$')
    
    num_contours = kwargs.get('num_contours', [1.0, 2.0, 3.4, 4.8, 5.8])
    ax  = fig.add_subplot( gs[1,1] )
    i=1; j=2;  k=5 
    meshplot( ax,results, i, j, k, contours = num_contours, \
              dashed=dashed_num, base=1e5,  **kwargs )     
    ax.set_title('$N/10^{5}$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ waist}\ w_{C}\ (\mu\mathrm{m})$')

    hwhm_contours = kwargs.get('hwhm_contours', [1.0, 2.0, 3.4, 4.8, 5.8])
    ax  = fig.add_subplot( gs[1,2] )
    i=1; j=2;  k=8
    meshplot( ax,results, i, j, k, contours = hwhm_contours, \
              dashed=dashed_hwhm, base=1.,  **kwargs )     
    ax.set_title('$\mathrm{HWHM}/w_{L}$', fontsize=16)
    ax.set_xlabel('$\mathrm{Lattice\ waist}\ w_{L}\ (\mu\mathrm{m})$')
    ax.set_ylabel('$\mathrm{Compensation\ waist}\ w_{C}\ (\mu\mathrm{m})$')
   
    return fig  

