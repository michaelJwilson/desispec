import os

import time
import glob
import pickle
import fitsio

import itertools
import warnings

import numpy                     as      np
import pylab                     as      pl

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
from   astropy.table             import  Table, join
from   desispec.io.image         import  read_image
from   specter.psf.gausshermite  import  GaussHermitePSF
from   scipy.signal              import  medfilt
from   desispec.calibfinder      import  CalibFinder
from   astropy.utils.exceptions  import  AstropyWarning
from   scipy                     import  stats
from   pathlib                   import  Path
from   templateSNR               import  templateSNR
from   bitarray                  import  bitarray
from   astropy.convolution       import  convolve, Box1DKernel
from   os                        import  path
from   dtemplateSNR              import  dtemplateSNR, dtemplateSNR_modelivar

warnings.simplefilter('ignore', category=AstropyWarning)

class RadLSS(object):
    def __init__(self, night, expid, cameras=None, shallow=False, aux=True, prod='/global/cfs/cdirs/desi/spectro/redux/blanc/', calibdir=os.environ['DESI_SPECTRO_CALIB'], desimodel=os.environ['DESIMODEL'],\
                                                                                                                                outdir='/global/cscratch1/sd/mjwilson/radlss/blanc/', rank=0):
        """
        Creates a spectroscopic rad(ial) weights instance.
        
        Args:
            night: night to solve for
            expid: 
            cameras: subset of cameras to reduce, else all (None) 
            prod: path to spec. pipeline reductions.
            shallow:
            odir: path to write to. 
            aux: loading auxiliary data required for e.g. varying psf & psf local rdnoise.  
            rank: given rank, for logging.  
        """
                    
        # E.g. export DESI_SPECTRO_CALIB=/global/cfs/cdirs/desi/spectro/desi_spectro_calib/trunk/.
        #
        # E.g. $DESI_MODEL:  global/common/software/desi/cori/desiconda/20200801-1.4.0-spec/code/desimodel/0.13.0.
        # E.g. $DESI_BASIS_TEMPLATES:  /global/scratch/mjwilson/desi-data.dm.noao.edu/desi/spectro/templates/basis_templates/v3.1/

        self.aux              = aux
        self.prod             = prod
        self.night            = night
        self.expid            = expid
        self.rank             = rank
        self.calibdir         = calibdir
        self.desimodel        = desimodel
        
        if cameras is None:
            petals            = np.arange(10)
            self.cameras      = [x[0] + x[1] for x in itertools.product(['b', 'r', 'z'], petals.astype(np.str))]

        else:
            self.cameras      = cameras

        if self.aux:
            # Time to retrieve aux. data, e.g. preprocs. 
            self.auxtime      = 0.0
            
        self.fail             = False
        self.flavor           = 'unknown'
        
        self.templates        = {}
   
        self.nmodel           = {}
        self.ensemble_tracers = []
        self.ensemble_flux    = {}
        self.ensemble_dflux   = {} # Stored (F - ~F_100A) flux for the ensemble.
        self.ensemble_meta    = {}
        self.ensemble_objmeta = {}
        self.template_snrs    = {}
                
        # Establish if cframes exist & science exposure, from header info.  
        self.get_data(shallow=True)
            
        if (self.flavor == 'science') and (not shallow):
            self.gfa_dir          = '{}/gfa/'.format(outdir)
            self.ensemble_dir     = '{}/ensemble/'.format(outdir)
            self.expdir           = '{}/exposures/{}/{:08d}/'.format(outdir, self.night, self.expid)
            self.outdir           = '{}/tiles/{}/{}/'.format(outdir, self.tileid, self.night)
            self.qadir            = self.outdir + '/QA/{:08d}/'.format(self.expid)

            Path(self.gfa_dir).mkdir(parents=True, exist_ok=True)
            Path(self.ensemble_dir).mkdir(parents=True, exist_ok=True)
            Path(self.outdir).mkdir(parents=True, exist_ok=True)

            # Caches spectra matched gfas to file. 
            self.get_gfas()

            # TO DO: Get GFA estimated depths for Anand's SV1 exposures.             
            try:
                self.get_data(shallow=False)

            except Exception as e:
                print('Failed to retrieve raw and reduced data for expid {} and cameras {}'.format(expid, self.cameras))
                print('\n\t{}'.format(e))
                
                self.fail = True
          
        else:
            self.fail = True

            print('{:08d}:  Non-science ({}) exposure.'.format(expid, self.flavor))  
        
    def get_data(self, shallow=False):    
        '''
        Grab the raw and reduced data for a given (SV0) expid.
        Bundle by camera (band, petal) for each exposure instance.

        args:
            shallow:  if True, populate class with header derived exptime, mjd, tileid,
                      flavour & program for this exposure, but do not gather corresponding
                      pipeline data, e.g. cframe flux etc.  

            aux:
                      gather data required for a non-basic tsnr, e.g. varying spec. psf.
        '''
        
        # For each camera of a given exposure.
        self.frames      = {}
        self.cframes     = {}

        self.psfs        = {}
        self.skies       = {} 
        self.fluxcalibs  = {}
        self.fiberflats  = {}  # NOT including in the flux calibration. 
        self.fibermaps   = {} 
        self.psf_nights  = {}
        self.preprocs    = {}

        # Start the clock.
        start_getdata    = time.perf_counter()

        # Of the requested cameras, which are actually reduced. 
        reduced_cameras  = glob.glob(self.prod + '/exposures/{}/{:08d}/cframe-*'.format(self.night, self.expid))
        reduced_cameras  = [x.split('-')[1] for x in reduced_cameras]

        bit_mask         = [x in reduced_cameras for x in self.cameras]

        if np.all(np.array(bit_mask) == False):
            self.fail    = True

            print('Requested cameras {} are not successfully reduced.  Try any of {}.'.format(self.cameras, reduced_cameras))

        # Solve for only reduced & requested cameras. 
        self.cameras     = np.array(self.cameras)[bit_mask]
        self.cameras     = self.cameras.tolist()

        self.cam_bitmask = bitarray(bit_mask)

        # All petals with an available camera.
        self.petals      = np.unique(np.array([x[1] for x in self.cameras]))

        if shallow:
            print('Rank {}:  Shallow gather of data, header info. for existence & (flavor == science) checks.'.format(self.rank))
            
        for cam in self.cameras:            
            # E.g.  /global/cfs/cdirs/desi/spectro/redux/andes/tiles/67230/20200314/cframe-z9-00055382.fits; 'cframe' | 'frame'.
            self.cframes[cam]    = read_frame(findfile('cframe', night=self.night, expid=self.expid, camera=cam, specprod_dir=self.prod))
                     
            self.exptime         = self.cframes[cam].meta['EXPTIME']        
            self.mjd             = self.cframes[cam].meta['MJD-OBS']
            self.tileid          = self.cframes[cam].meta['TILEID']
            self.flavor          = self.cframes[cam].meta['FLAVOR']
            self.program         = self.cframes[cam].meta['PROGRAM']
            
            if shallow:
                del self.cframes[cam]
                continue

            print('Rank {}:  Grabbing camera {}.'.format(self.rank, cam))
                                            
            self.skies[cam]      = read_sky(findfile('sky', night=self.night, expid=self.expid, camera=cam, specprod_dir=self.prod))
        
            # https://desidatamodel.readthedocs.io/en/latest/DESI_SPECTRO_CALIB/fluxcalib-CAMERA.html
            self.fluxcalibs[cam] = read_flux_calibration(findfile('fluxcalib', night=self.night, expid=self.expid, camera=cam, specprod_dir=self.prod))            
            self.fiberflats[cam] = read_fiberflat(self.calibdir + '/' + CalibFinder([self.cframes[cam].meta]).data['FIBERFLAT'])

            self.flat_expid      = self.fiberflats[cam].header['EXPID']
        
            if self.aux:
                start_getaux       = time.perf_counter()

                # Used for NEA, RDNOISE calc. 
                self.fibermaps[cam]  = self.cframes[cam].fibermap

                self.ofibermapcols   = list(self.cframes[cam].fibermap.dtype.names)

                self.fibermaps[cam]['TILEFIBERID'] = 10000 * self.tileid + self.fibermaps[cam]['FIBER']
                
                # Used for NEA, RDNOISE calc. and model 2D extraction;  Types: ['psf', 'psfnight']
                self.psfs[cam]     = GaussHermitePSF(findfile('psf', night=self.night, expid=self.expid, camera=cam, specprod_dir=self.prod))

                # Used for RDNOISE calc. 
                self.preprocs[cam] = read_image(findfile('preproc', night=self.night, expid=self.expid, camera=cam, specprod_dir=self.prod))

                end_getaux         = time.perf_counter()

                self.auxtime      += (end_getaux - start_getaux) / 60.
                
        # End the clock.                                                                                                                                                                                     
        end_getdata    = time.perf_counter()

        if not shallow:
            print('Rank {}:  Retrieved pipeline data in {:.3f} mins.'.format(self.rank, (end_getdata - start_getdata) / 60.))

            if self.aux:
                print('Rank {}:  Retrieved AUX data in {:.3f} mins.'.format(self.rank, self.auxtime))
            
    def get_gfas(self, survey='SV1', thru_date='20210116', printit=False, cache=True):
        '''                                                                                                                                                                                                  
        Get A. Meisner's GFA exposures .fits file.                                                                                                                                                            
        Yields seeing FWHM, Transparency and fiberloss for a given spectroscopic expid.                                                                                                                       
        '''

        start_getgfa = time.perf_counter()

        cgfa_path    = self.gfa_dir + '/spectro_gfas_{}_thru_{}.fits'.format(survey, thru_date)
        
        if (not cache) | (not os.path.exists(cgfa_path)):
            gfa_path     = '/global/cfs/cdirs/desi/users/ameisner/GFA/conditions/offline_all_guide_ccds_{}-thru_{}.fits'.format(survey, thru_date)
            gfa          = Table.read(gfa_path)
      
            # Quality cuts.                                                                                                                                                                                        
            gfa          = gfa[gfa['SPECTRO_EXPID'] > -1]                                                                                                                                                          
            gfa          = gfa[gfa['N_SOURCES_FOR_PSF'] > 2]                                                                                                                                                       
            gfa          = gfa[gfa['CONTRAST'] > 2]                                                                                                                                                                
            gfa          = gfa[gfa['NPIX_BAD_TOTAL'] < 10]                                                                                                                                                         

            gfa.sort('SPECTRO_EXPID')

            by_spectro_expid                        = gfa.group_by('SPECTRO_EXPID')
                                                                                                                                                                                              
            self.gfa_median                         = by_spectro_expid.groups.aggregate(np.median)

            self.gfa_median                         = self.gfa_median['SPECTRO_EXPID', 'FWHM_ASEC', 'TRANSPARENCY', 'FIBER_FRACFLUX', 'SKY_MAG_AB', 'MJD', 'NIGHT']

            self.gfa_median.write(cgfa_path, format='fits', overwrite=True)

        else:
            self.gfa_median = Table.read(cgfa_path)
            
        self.gfa_median_exposure                = self.gfa_median[self.gfa_median['SPECTRO_EXPID'] == self.expid]
        self.fiberloss                          = self.gfa_median_exposure['FIBER_FRACFLUX'][0]

        if printit:
            print(self.gfa_median)

        end_getgfa = time.perf_counter()

        print('Rank {}:  Retrieved GFA info. in {:.3f} mins.'.format(self.rank, (end_getgfa - start_getgfa) / 60.))
            
    def calc_nea(self, psf_wave=None, write=False):
        '''
        Calculate the noise equivalent area for each fiber of a given camera from the local psf, at a representative OII wavelength.  
        Added to self.fibermaps[cam].
        
        Input:
            psf_wave:  wavelength at which to evaluate the PSF, e.g. 3727. * (1. + 1.1) 

        Result:
            populates instance.fibermaps with NEA and ANGSTROMPERPIXEL.

        Approx:
            if approx, nea= , angstromperpix= 

        '''

        start_calcnea = time.perf_counter()
        
        for cam in self.cameras:
            psf = self.psfs[cam]
            
            # Note:  psf.nspec, psf.npix_x, psf.npix_y
            if psf_wave is None:
                # Representative wavelength.
                z_elg    = 1.1
                psf_wave = 3727. * (1. + z_elg)    
            
            if (psf_wave < self.cframes[cam].wave.min()) | (psf_wave > self.cframes[cam].wave.max()):
                psf_wave = np.median(self.cframes[cam].wave)
               
            fiberids = self.fibermaps[cam]['FIBER']
                
            #  Fiber centroid position on CCD.
            #  https://github.com/desihub/specter/blob/f242a3d707c4cba549030af6df8cf5bb12e2b47c/py/specter/psf/psf.py#L467
            #  x,y = psf.xy(fiberids, self.psf_wave)
            
            #  https://github.com/desihub/specter/blob/f242a3d707c4cba549030af6df8cf5bb12e2b47c/py/specter/psf/psf.py#L300
            #  Range that boxes in fiber 'trace':  (xmin, xmax, ymin, ymax)
            #  ranges = psf.xyrange(fiberids, self.psf_wave)
            
            #  Note:  Expectation of ** 3.44 ** for PSF size in pixel units (spectro paper).
            #  Return Gaussian sigma of PSF spot in cross-dispersion direction in CCD pixel units.
            #  Gaussian PSF, radius R that maximizes S/N for a faint source in the sky-limited case is 1.7σ
            #  http://www.ucolick.org/~bolte/AY257/s_n.pdf
            #  2. * 1.7 * psf.xsigma(ispec=fiberid, wavelength=oii)
            
            #  Gaussian sigma of PSF spot in dispersion direction in CCD pixel units.
            #  Gaussian PSF, radius R that maximizes S/N for a faint source in the sky-limited case is 1.7σ
            #  http://www.ucolick.org/~bolte/AY257/s_n.pdf
            #  2. * 1.7 * psf.ysigma(ispec=fiberid, wavelength=oii)
           
            neas             = []
            angstrom_per_pix = []
    
            for ifiber, fiberid in enumerate(fiberids):
                psf_2d = psf.pix(ispec=ifiber, wavelength=psf_wave)
            
                # norm = np.sum(psf_2d)
             
                # http://articles.adsabs.harvard.edu/pdf/1983PASP...95..163K
                neas.append(1. / np.sum(psf_2d ** 2.))  # [pixel units].

                angstrom_per_pix.append(psf.angstroms_per_pixel(ifiber, psf_wave))

            self.fibermaps[cam]['NEA']              = np.array(neas)
            self.fibermaps[cam]['ANGSTROMPERPIXEL'] = np.array(angstrom_per_pix)
            
            if write:
                raise  NotImplementedError()

        # End the clock.                                                                                                                                                                                      
        end_calcnea = time.perf_counter()

        print('Rank {}:  Calculated NEA in {:.3f} mins.'.format(self.rank, (end_calcnea - start_calcnea) / 60.))
        
    def calc_readnoise(self, psf_wave=None):
        '''
        Calculate the readnoise for each fiber of a given camera from the patch matched to the local psf.
        Added to self.fibermaps[cam].
        
        Requires pre_procs prod. loading by aux=True in class initialization. 

        Input:
          psf_wave:  wavelength at which to evaluate the PSF, e.g. 3727. * (1. + 1.1).

        Result:
          populates instance.fibermap with ccdx, ccdy, rdnoise, quad, rdnoise_quad.

        Questions:
          can get readnoise from cframes if quadrant of each fiber known (ccdx, ccdy) without psf?
          
        '''
        
        self.ccdsizes          = {}   

        start_calcread         = time.perf_counter()
        
        for cam in self.cameras:
            self.ccdsizes[cam] = np.array(self.cframes[cam].meta['CCDSIZE'].split(',')).astype(np.int)
                                                    
            if psf_wave is None:
                #  Representative wavelength.
                z_elg      = 1.1
                psf_wave   = 3727. * (1. + z_elg)    
            
            if (psf_wave < self.cframes[cam].wave.min()) | (psf_wave > self.cframes[cam].wave.max()):
                psf_wave   = np.median(self.cframes[cam].wave)
               
            fiberids       = self.fibermaps[cam]['FIBER']
        
            rd_noises      = []
            
            ccd_x          = []
            ccd_y          = []
                
            quads          = [] 
            rd_quad_noises = []
        
            def quadrant(x, y):
                if (x < (self.ccdsizes[cam][0] / 2)):
                    if (y < (self.ccdsizes[cam][1] / 2)):
                        return  'A'
                    else:
                        return  'C'
                        
                else:
                    if (y < (self.ccdsizes[cam][1] / 2)):
                        return  'B'
                    else:
                        return  'D'

            psf = self.psfs[cam]
                    
            for ifiber, fiberid in enumerate(fiberids):
                #  TODO:  CCD X,Y not otherwise available?  E.g. in header.  
                x, y                     = psf.xy(ifiber, psf_wave)
        
                ccd_quad                 = quadrant(x, y)
    
                (xmin, xmax, ymin, ymax) = psf.xyrange(ifiber, psf_wave)

                # [electrons/pixel] (float).
                rd_cutout = self.preprocs[cam].readnoise[ymin:ymax, xmin:xmax]
                    
                rd_noises.append(np.median(rd_cutout))
        
                ccd_x.append(x)
                ccd_y.append(y)        

                quads.append(ccd_quad)

                rd_quad_noises.append(self.cframes[cam].meta['OBSRDN{}'.format(ccd_quad)])
                        
            self.fibermaps[cam]['CCDX']         = np.array(ccd_x)
            self.fibermaps[cam]['CCDY']         = np.array(ccd_y)
                    
            self.fibermaps[cam]['RDNOISE']      = np.array(rd_noises)

            # https://github.com/desihub/spectropaper/blob/master/figures/ccdlayout.png
            self.fibermaps[cam]['QUAD']         = np.array(quads)
            self.fibermaps[cam]['RDNOISE_QUAD'] = np.array(rd_quad_noises)

        end_calcread = time.perf_counter()

        print('Rank {}:  Calculated psf-local readnoise in {:.3f} mins.'.format(self.rank, (end_calcread - start_calcread) / 60.))

    def grab_ensemble(self, ensemble_type='template', tracer='BGS'):
        assert  np.isin(ensemble_type, ['template', 'deepfield'])

        print(self.ensemble_dir + '/{}-{}-ensemble-flux.fits'.format(ensemble_type, tracer.lower()))
        
        with open(self.ensemble_dir + '/{}-{}-ensemble-flux.fits'.format(ensemble_type, tracer.lower()), 'rb') as handle:
            self.ensemble_flux = pickle.load(handle)

        with open(self.ensemble_dir + '/{}-{}-ensemble-dflux.fits'.format(ensemble_type, tracer.lower()), 'rb') as handle:
            self.ensemble_dflux = pickle.load(handle)

        with open(self.ensemble_dir + '/{}-{}-ensemble-meta.fits'.format(ensemble_type, tracer.lower()), 'rb') as handle:
            self.ensemble_meta = pickle.load(handle)
            
        with open(self.ensemble_dir + '/{}-{}-ensemble-objmeta.fits'.format(ensemble_type, tracer.lower()), 'rb') as handle:
            self.ensemble_objmeta = pickle.load(handle)

        self.nmodel = len(self.ensemble_flux['b'])

        print('Successfully retrieved pre-written ensemble files at: {} (nmodel: {})'.format(self.ensemble_dir, self.nmodel))

    def calc_templatesnrs(self, tracer='ELG'):
        '''
        Calculate template SNRs for the ensemble. 
        
        Calls generation of templates, which reads
        cache if possible. 
        
        Result:
            [nfiber x ntemplate] array of tSNR values
            for each camera. 
        '''        

        start_templatesnr           = time.perf_counter()

        self.template_snrs[tracer]  = {}
        
        for cam in self.cameras:  
            band                    = cam[0]
            
            nfiber                  = len(self.fibermaps[cam]['FIBER'])
                
            self.template_snrs[tracer][cam]                   = {} 

            # Uses cframe IVAR. 
            self.template_snrs[tracer][cam]['TSNR']           = np.zeros((nfiber, self.nmodel[tracer])) 

            # Uses CCD-equation based on model, calculate npix, rdnoise, etc. 
            self.template_snrs[tracer][cam]['TSNR_MODELIVAR'] = np.zeros((nfiber, self.nmodel[tracer])) 
            
            for ifiber, fiberid in enumerate(self.fibermaps[cam]['FIBER']):
                for j, template_dflux in enumerate(self.ensemble_dflux[tracer][band]):
                    # Pipeline processed.
                    sky_flux            = self.skies[cam].flux[ifiber,:]
                    flux_calib          = self.fluxcalibs[cam].calib[ifiber,:]
                    flux_ivar           = self.cframes[cam].ivar[ifiber, :]
                    fiberflat           = self.fiberflats[cam].fiberflat[ifiber,:]

                    # 
                    readnoise           = self.cframes[cam].fibermap['RDNOISE'][ifiber]  
                    npix                = self.cframes[cam].fibermap['NEA'][ifiber]  
                    angstroms_per_pixel = self.cframes[cam].fibermap['ANGSTROMPERPIXEL'][ifiber]  

                    # TSNR with cframe IVAR
                    self.template_snrs[tracer][cam]['TSNR'][ifiber, j]           = dtemplateSNR(template_dflux, flux_ivar)

                    # TSNR with ccd-model IVAR (derived from rdnoise, sky etc.).
                    self.template_snrs[tracer][cam]['TSNR_MODELIVAR'][ifiber, j] = dtemplateSNR_modelivar(template_dflux, sky_flux=sky_flux, flux_calib=flux_calib,
                                                                                                          fiberflat=fiberflat, readnoise=readnoise, npix=npix,
                                                                                                          angstroms_per_pixel=angstroms_per_pixel)
        # 
        self.template_snrs[tracer]['brz'] = {}
            
        for cam in self.cameras:
            petal = cam[1]
            
            coadd = 'brz{}'.format(petal) 
            
            if coadd not in self.template_snrs[tracer].keys():
                self.template_snrs[tracer][coadd] = {}
            
            for tsnr_type in ['TSNR', 'TSNR_MODELIVAR']:
                if tsnr_type not in self.template_snrs[tracer][coadd].keys():
                    self.template_snrs[tracer][coadd][tsnr_type] = np.zeros_like(self.template_snrs[tracer][self.cameras[0]][tsnr_type]) 
              
                self.template_snrs[tracer][coadd][tsnr_type]    += self.template_snrs[tracer][cam][tsnr_type]

        end_templatesnr = time.perf_counter()
                
        print('Rank {}:  Solved for template snrs in {:.3f} mins.'.format(self.rank, (end_templatesnr - start_templatesnr) / 60.))
                        
    def write_radweights(self, output_dir='/global/scratch/mjwilson/', vadd_fibermap=False):
        '''
        Each camera, each fiber, a template SNR (redshift efficiency) as a function of redshift &
        target magnitude, per target class.  This could be for instance in the form of fits images in 3D
        (fiber x redshift x magnitude), with one HDU per target class, and one fits file per tile.
        '''

        start_writeweights = time.perf_counter()

        for petal in self.petals:            
          hdr             = fits.open(findfile('cframe', night=self.night, expid=self.expid, camera=self.cameras[0], specprod_dir=self.prod))[0].header

          hdr['EXTNAME']  = 'RADWEIGHT'

          primary         = fits.PrimaryHDU(header=hdr)

          hdu_list        = [primary] + [fits.ImageHDU(self.template_snrs[tracer]['brz{}'.format(petal)]['TSNR'], name=tracer) for tracer in self.ensemble_tracers]  
 

          all_hdus        = fits.HDUList(hdu_list)

          all_hdus.writeto(self.outdir + '/radweights-{}-{:08d}.fits'.format(petal, self.expid), overwrite=True)

        if vadd_fibermap:
          for cam in self.cameras:
              #  Derived fibermap info.
              derived_cols    = [x for x in list(self.cframes[cam].fibermap.dtype.names) if x not in self.ofibermapcols]
 
              self.cframes[cam].fibermap[derived_cols].write(self.outdir + '/vadd-fibermap-{}-{:08d}.fits'.format(cam, self.expid), overwrite=True)
 
        end_writeweights  = time.perf_counter()

        print('Rank {}:  Written rad. weights (and value added fibermap) to {} in {:.3f} mins.'.format(self.rank, self.outdir + '/radweights-?-{:08d}.fits'.format(self.expid), (end_writeweights - start_writeweights) / 60.))
          
    def compute(self, tracers=['BGS'], ensemble_type='deepfield'):
        #  Process only science exposures.
        if self.flavor == 'science':        
            if (not self.fail):
                self.calc_nea()
             
                self.calc_readnoise()
                
                for tracer in tracers:
                    self.grab_ensemble(ensemble_type, tracer=tracer)

                    self.calc_templatesnrs(tracer=tracer)
                    
                # tSNRs & derived fibermap info., e.g. NEA, RDNOISE, etc ... 
                self.write_radweights()
                
            else:
                print('Skipping non-science exposure: {:08d}'.format(self.expid))
