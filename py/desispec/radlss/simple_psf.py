import numpy             as np
import pylab             as pl
import matplotlib.pyplot as plt

from   RadLSS import RadLSS                                                                                                                                                                                                              


night = '20201223'
expid = 69611

rads  = RadLSS(night, expid, cameras=None, rank=0, shallow=False, outdir='.')
rads.compute(tracers=['BGS'])                                                                                                                                                                                                          
swave  = np.arange(4.e3, 1.e4, 1.e2) 
fibers = np.arange(500)

fig, axes = plt.subplots(3, 1, figsize=(15, 15))
axes      = {'b': axes[0], 'r': axes[1], 'z': axes[2]}

data      = []
colors    = plt.rcParams['axes.prop_cycle'].by_key()['color']

for cam in rads.cameras:
    mjd       = rads.mjd
    band      = cam[0]
    petal     = np.int(cam[1])
    
    psf       = rads.psfs[cam]

    psf_wave  = np.median(rads.cframes[cam].wave)

    wdisps    = []
    
    for ifiber in fibers:
        wdisps.append(psf.wdisp(ifiber, wavelength=psf_wave).tolist())

    wdisps = np.array(wdisps)
        
    # psf_2d  = psf.pix(ispec=0, wavelength=psf_wave)
    # xslice, yslice, pixels = psf.xypix(0, psf_wave)

    mean   = np.mean(wdisps)
    std    =  np.std(wdisps)

    axes[band].errorbar(mjd, mean, yerr=std, color=colors[petal], marker='^', markersize=4, alpha=0.5)
    # axes[band].set_ylim(14., 19.)
    # axes[band].set_ylabel(r'NEA [PIX dA]')
    # axes[band].set_title('{:.3f} AA'.format(psfwave))
    # break

pl.show()
