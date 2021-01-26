import glob
import pylab as pl
import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits

from   astropy.table import Table, join


np.random.seed(314)

conds     = Table.read('sv1-exposures.fits') 

rds       = glob.glob('/global/cscratch1/sd/mjwilson/radlss/blanc/tiles/*/*/rdnoise*.fits')

fig, axes = plt.subplots(3, 1, figsize=(15, 15))

axes      = {'b': axes[0], 'r': axes[1], 'z': axes[2]}

data      = []

colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']

expids    = []

for rd in rds:
    parts = rd.split('/')[-1].split('_')

    cam   = parts[1]
    band  = cam[0]
    petal = np.int(cam[1])
    
    expid = np.int(parts[2].replace('.fits',''))

    expids.append(expid)
    
    psfwave      = fits.open(rd)[1].header['PSFWAVE']
    
    # [('FIBER', '>i4'), ('NEA', '>f8'), ('ANGSTROMPERPIXEL', '>f8')])) 
    rd          = Table.read(rd)
    rd['EXPID'] = expid
    
    rd    = join(rd, conds['EXPID', 'MJDOBS'], join_type='left', keys='EXPID')
    
    mean  = np.mean(rd['RDNOISE'])
    std   =  np.std(rd['RDNOISE'])

    mjd   = np.mean(rd['MJDOBS'])
    # mjd  /= 1.e4

    # 15 min. scatter
    shift = 0.01

    # 5 min. scatter, 1/3 of a percent. 
    # mjd = mjd + (1. + np.random.uniform(-shift / 3., shift / 3.))
    
    axes[band].errorbar(mjd, mean, yerr=std, color=colors[petal], marker='^', markersize=4, alpha=0.5)
    # axes[band].set_ylim(2., 6.)

    axes[band].set_ylabel(r'RDNOISE [e / PIX]')

    axes[band].set_title('{:.3f} AA'.format(psfwave))
    
expids = np.array(expids)
expids = np.unique(expids)
expids = 'EXPIDS\n' + '\n'.join(expids.astype(str))

print(expids)

# https://github.com/desihub/specsim/blob/56e203f1ca0bd606390da351ac9d6b207c6bed08/specsim/data/config/desi.yaml#L197
axes['b'].axhline(y=3.41, c='k', alpha=0.5)
axes['r'].axhline(y=2.60, c='k', alpha=0.5)
axes['z'].axhline(y=2.60, c='k', alpha=0.5)

axes['z'].set_xlabel(r'MJD $[10^4]$')
# axes['z'].text(1.1, 1.0, expids, transform=axes['z'].transAxes, verticalalignment='top', fontsize=8.)

plt.tight_layout()
pl.show()