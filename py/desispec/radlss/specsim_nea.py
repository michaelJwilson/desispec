import pylab             as     pl    
import astropy.io.fits   as     fits       
import matplotlib.pyplot as     plt
import numpy             as      np

from   scipy.stats       import multivariate_normal


def calc_nea(fwhm_wave, fwhm_spat, plot=False):
    var_wave  = (fwhm_wave / 2.355)**2.
    var_spat  = (fwhm_spat / 2.355)**2.

    dx, dy    = .01, .01
    dA        =  dx * dy

    x, y      = np.mgrid[-5 : 5 : dx, -5 : 5 : dy]
    pos       = np.dstack((x, y))
    rv        = multivariate_normal([0.0, 0.0], [[var_wave, 0.0], [0.0, var_spat]])

    psf       = rv.pdf(pos)

    norm      = np.sum(psf) * dA
    psf      /= norm

    nea       = 1. / np.sum(psf ** 2.) / dA
    
    if plot:
        fig2  = plt.figure()
        ax2   = fig2.add_subplot(111)        
        ax2.contourf(x, y, psf)         

    return  nea

##
fig, axes     = plt.subplots(2, 1, figsize=(10,10))

for band in ['B', 'R', 'Z']:
    dat       = fits.open('/global/common/software/desi/cori/desiconda/20200801-1.4.0-spec/code/desimodel/master/data/specpsf/psf-quicksim.fits') 

    sampling  = 100
    
    wave      = dat['QUICKSIM-{}'.format(band)].data['wavelength'][::sampling]

    # 0.62425197, ..., 0.55040059 A/pix.
    angperpix = dat['QUICKSIM-{}'.format(band)].data['angstroms_per_row'][::sampling]
    
    fwhm_wave  = dat['QUICKSIM-{}'.format(band)].data['fwhm_wave'][::sampling]    # [AA]
    fwhm_wave /= angperpix                                                        # [pixels]

    fwhm_spat  = dat['QUICKSIM-{}'.format(band)].data['fwhm_spatial'][::sampling] # [pixels]
    
    if band == 'B':
        axes[0].plot(wave, fwhm_wave, label=r'wavelength [pixels]')
        axes[0].plot(wave, fwhm_spat, label=r'spatial [pixels]')

    else:
        axes[0].plot(wave, fwhm_wave, label='')
        axes[0].plot(wave, fwhm_spat, label='')
        
    neas      = []
    
    for i, f_wave in enumerate(fwhm_wave):        
        neas.append(calc_nea(f_wave, fwhm_spat[i], plot=False))

    neas      = np.array(neas)

    axes[1].plot(wave, neas)

axes[0].set_ylabel('PSF FWHM')
axes[1].set_ylabel('NEA')

for ax in axes:
    ax.legend(frameon=False)  

axes[1].set_xlabel(r'Wavelength $[\AA]$')

fig.suptitle('Specsim.')

pl.savefig('plots/specsim_nea.pdf')
