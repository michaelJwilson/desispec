import glob
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits

from   astropy.table import Table, join


np.random.seed(314)

conds     = Table.read('sv1-exposures.fits')

rads      = glob.glob('/global/cscratch1/sd/mjwilson/radlss/blanc/tiles/*/*/radweights*.fits')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes      = {'b': axes[0], 'r': axes[1], 'z': axes[2]}

data      = []

colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']

expids    = []

for rad in rads:
    # radweights-7-00069593
    parts = rad.split('/')[-1].split('-')

    petal = np.int(parts[1])
    
    expid = np.int(parts[2].replace('.fits',''))
    expids.append(expid)
        
    rad   = fits.open(rad)
    rad   = rad['BGS'].data    

    # Cut out masked pixels.
    rad   = rad[rad > 0.0]
    
    mean  = np.mean(rad)
    std   =  np.std(rad)
    
    mjd   = conds['MJDOBS'][conds['EXPID'] == expid][0]
    mjd  /= 1.e4

    # 15 min. scatter
    shift = 0.01

    # 5 min. scatter, 1/3 of a percent. 
    mjd   = mjd * (1. + np.random.uniform(-shift / 3., shift / 3.))

    for band in ['B', 'R', 'Z']:
        # _EBVAIR
        dep  = conds['{}_DEPTH'.format(band)][conds['EXPID'] == expid]    
        
        axes[band.lower()].errorbar(dep, mean, yerr=std, marker='^', markersize=4, alpha=0.5)
        axes[band.lower()].set_xscale('log')

        axes[band.lower()].set_xlabel('{}_DEPTH_EBVAIR'.format(band)) 
        axes[band.lower()].set_ylabel(r'$\tau$SNR')
        
        # axes[band].set_ylim(14., 19.)
        # axes[band].set_xlabel(r'MJD $[10^4]$')
        # axes[band].set_ylabel(r'NEA [PIX]')
        # axes[band].set_title('{:.3f} AA'.format(psfwave))
    
expids = np.array(expids)
expids = np.unique(expids)
expids = 'EXPIDS\n' + '\n'.join(expids.astype(str))

print(expids)

axes['z'].text(1.1, 1.0, expids, transform=axes['z'].transAxes, verticalalignment='top', fontsize=8.)

plt.tight_layout()
pl.show()
