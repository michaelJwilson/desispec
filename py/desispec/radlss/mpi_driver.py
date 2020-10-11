#- srun -n 2 python mpi_driver.py 
#- Initialize MPI ASAP before proceeding with other imports.
import multiprocessing

from   mpi4py                   import  MPI


nproc = multiprocessing.cpu_count() // 2
comm  = MPI.COMM_WORLD
rank  = comm.Get_rank()
size  = comm.Get_size()

# -----------------------------------------------------------

import os
import sys
import time
import glob
import fitsio

import itertools
import warnings

import numpy as np
import pylab as pl

import desisim.templates
import astropy.io.fits           as      fits

import desispec.io
import redrock.templates
import matplotlib.pyplot         as      plt

from   astropy.convolution       import  convolve, Box1DKernel
from   desispec.spectra          import  Spectra
from   desispec.frame            import  Frame
from   desispec.resolution       import  Resolution
from   desispec.io.meta          import  findfile
from   desispec.io               import  read_frame, read_fiberflat, read_flux_calibration, read_sky, read_fibermap 
from   desispec.interpolation    import  resample_flux
from   astropy.table             import  Table
from   desispec.io.image         import  read_image
from   specter.psf.gausshermite  import  GaussHermitePSF
from   scipy.signal              import  medfilt
from   desispec.calibfinder      import  CalibFinder
from   astropy.utils.exceptions  import  AstropyWarning
from   scipy                     import  stats
from   pathlib                   import  Path
from   templateSNR               import	 templateSNR
from   RadLSS                    import  RadLSS
from   desispec.parallel         import  stdouterr_redirected


andes     = '/global/cfs/cdirs/desi/spectro/redux/andes'

def get_expids(night, andes='/global/cfs/cdirs/desi/spectro/redux/andes'):
    return  np.sort(np.array([x.split('/')[-1] for x in glob.glob(andes + '/exposures/{}/*'.format(night))]).astype(np.int))

nmax      = 1
night     = '20200315'
tracers   = ['ELG']

expids    = get_expids(night)

blacklist = np.array([55556]) # 55556, 55557, 55587, 55592, 55668, 55670, 55672, 55674, 55676, 55678, 55680, 55682, 55684, 55686, 55688, 55690, 55692, 55705, 55706, 55707, 55708, 55709, 55713, 55714, 55715, 55716, 55717])
is_black  = np.isin(expids, blacklist)

expids    = expids[~is_black]

blacklist = blacklist.tolist()

cameras   = ['b6', 'r6', 'z6']

comm.barrier()

expids    = comm.bcast(expids, root=0)

if rank == 0:
  print('Number of exposures to process: {}'.format(len(expids)))

# https://github.com/desihub/desitarget/blob/master/bin/mpi_select_mock_targets
iexp      = np.linspace(0, len(expids), size+1, dtype=int)
rankexp   = expids[iexp[rank]:iexp[rank+1]]

print('rank {} processes {} exposures {}'.format(rank, iexp[rank+1]-iexp[rank], rankexp))

sys.stdout.flush()

comm.barrier()

if len(rankexp) > 0:
  start   = time.perf_counter()
    
  for nexp, expid in enumerate(rankexp):
    if expid not in blacklist:
      logdir   = '/global/cscratch1/sd/mjwilson/radlss/logs/{:08d}/'.format(expid)
      logfile  = logdir + '/{:08d}.log'.format(expid) 

      Path(logdir).mkdir(parents=True, exist_ok=True)

      print('Rank {}:  Writing log for {:08d} to {}.'.format(rank, expid, logfile))
      
      with stdouterr_redirected(to=logfile):
          print('Rank {}: Solving for EXPID {:08d} ({} of {})'.format(rank, expid, nexp, len(expids)))
        
          rads = RadLSS(night, expid, cameras=None, rank=rank)
    
          rads.compute(templates=True)

          #  if nexp == nmax:
          #    break

          print('Rank {}:  SUCCESS.  Processed EXPID {:08d} in {:.3f} minutes.'.format(rank, expid, (time.perf_counter() - start) / 60.))
      
          del rads

          #blacklist.append(expid)

          #print('Rank {} blacklisted {} {}'.format(rank, expid, andes + '/exposures/{}/{:08d}/'.format(night, expid)))
          
          sys.stdout.flush()