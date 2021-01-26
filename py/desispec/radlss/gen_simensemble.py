import time
import pickle
import desisim
import os.path           as path
import numpy             as np
import desisim.templates

from   astropy.convolution           import convolve, Box1DKernel
from   pathlib                       import Path


np.random.seed(seed=314)

# AR/DK DESI spectra wavelengths                                                                                                                                                                                                      
# TODO:  where are brz extraction wavelengths defined?  https://github.com/desihub/desispec/issues/1006.                                                                                                                               
wmin, wmax, wdelta = 3600, 9824, 0.8
wave               = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
cslice             = {"b": slice(0, 2751), "r": slice(2700, 5026), "z": slice(4900, 7781)}

class template_ensemble(object):
    '''                                                                                                                                                                                                                                   
    Generate an ensemble of templates to sample tSNR for a range of points in                                                                                                                                                             
    (z, m, OII, etc.) space.                                                                                                                                                                                                                                                                                                                                                                                                                                                       
    Uses cache on disk if possible, otherwise writes it.                                                                                                                                                                                                                                                                                                                                                                                                                     
    If conditioned, uses deepfield redshifts and (currently r) magnitudes to condition simulated templates.                                                                                                                               
    '''
    def __init__(self, tracer='ELG', nmodel=5, cached=True, sort=True, conditionals=None, tile=1, prod='/global/cfs/cdirs/desi/spectro/redux/blanc',\
                 out_dir='/global/cscratch1/sd/mjwilson/radlss/', vi_dir='/global/cfs/cdirs/desi/sv/vi/TruthTables/'):
        
        def tracer_maker(wave, tracer=tracer, nmodel=nmodel, redshifts=None, mags=None):
            if tracer == 'ELG':
                maker = desisim.templates.ELG(wave=wave)

            elif tracer == 'QSO':
                maker = desisim.templates.QSO(wave=wave)

            elif tracer == 'LRG':
                maker = desisim.templates.LRG(wave=wave)

            elif tracer == 'BGS':
                maker = desisim.templates.BGS(wave=wave)

            else:
                raise  ValueError('{} is not an available tracer.'.format(tracer))

            flux, wave, meta, objmeta = maker.make_templates(nmodel=nmodel, trans_filter='decam2014-r', redshift=redshifts, mag=mags, south=True)

            return  wave, flux, meta, objmeta

        start_genensemble  = time.perf_counter()
        
        self.ensemble_type = 'desisim'

        # Strip e.g. blanc from prod. path.                                                                                                                                                                                                   
        out_dir           += prod.split('/')[-1]
        self.ensemble_dir  = '{}/ensemble/'.format(out_dir)

        Path(self.ensemble_dir).mkdir(parents=True, exist_ok=True)
        
        # If already available, load.                                                                                                                                                                                                         
        # TODO:  Simplify to one fits with headers?                                                                                                                                                                                           
        if (conditionals is None) & cached:
            try:                
                with open(self.ensemble_dir + '/template-{}-ensemble-flux.fits'.format(tracer.lower()), 'rb') as handle:
                    self.ensemble_flux = pickle.load(handle)

                with open(self.ensemble_dir + '/template-{}-ensemble-dflux.fits'.format(tracer.lower()), 'rb') as handle:
                    self.ensemble_dflux = pickle.load(handle)

                with open(self.ensemble_dir + '/template-{}-ensemble-meta.fits'.format(tracer.lower()), 'rb') as handle:
                    self.ensemble_meta = pickle.load(handle)

                with open(self.ensemble_dir + '/template-{}-ensemble-objmeta.fits'.format(tracer.lower()), 'rb') as handle:
                    self.ensemble_objmeta = pickle.load(handle)

                self.nmodel = len(self.ensemble_flux['b'])

                if self.nmodel != nmodel:
                    # Pass to below.                                                                                                                                                                                                        
                    raise  ValueError('Retrieved ensemble had erroneous model numbers (nmodel: {} == {}?)'.format(nmodel, self.nmodel))

                print('Successfully retrieved pre-written ensemble files at: {} (nmodel: {} == {}?)'.format(self.ensemble_dir, nmodel, self.nmodel))

                end_genensemble = time.perf_counter()

                print('Template ensemble (nmodel: {}) in {:.3f} mins.'.format(self.nmodel, (end_genensemble - start_genensemble) / 60.))

                # Successfully retrieved.                                                                                                                                                                                                   
                return

            except:
                print('Failed to retrieve pre-written ensemble files.  Regenerating.')
            
        if conditionals is None:
            _, flux, meta, objmeta     = tracer_maker(wave, tracer=tracer, nmodel=nmodel)
            
        else:
            conditional_redshifts      = conditionals['REDSHIFT'].data
            conditional_mags           = conditionals['FLUX_R'].data

            print('Assuming {} conditional redshifts and magnitudes for ensemble (tiling factor: {:d}).'.format(len(conditional_redshifts), tile))

            if tile > 1:
                conditional_redshifts  = np.tile(conditional_redshifts, tile)
                conditional_mags       = np.tile(conditional_mags,      tile)

                conditional_redshifts += np.random.normal(loc=0.0, scale=0.02, size=len(conditional_redshifts))
                # conditional_mags    += np.random.normal(loc=0.0, scale=0.05, size=len(conditional_mags))

                conditional_mags       = conditional_mags[conditional_redshifts > 0.0]
                conditional_redshifts  = conditional_redshifts[conditional_redshifts > 0.0]

                nmodel                 = len(conditional_redshifts)
                
            # Check:  assumes DECAM-R as standard.
            _, flux, meta, objmeta     = tracer_maker(wave, tracer=tracer, nmodel=nmodel, redshifts=conditional_redshifts, mags=conditional_mags)
                
        self.nmodel                    = nmodel
        
        self.ensemble_flux             = {}
        self.ensemble_dflux            = {}
        self.ensemble_meta             = meta
        self.ensemble_objmeta          = objmeta
        
        # Generate template (d)fluxes for brz bands.                                                                                                                                                                                          
        for band in ['b', 'r', 'z']:
            band_wave                     = wave[cslice[band]]

            in_band                       = np.isin(wave, band_wave)

            self.ensemble_flux[band]      = flux[:, in_band]

            dflux                         = np.zeros_like(self.ensemble_flux[band])
        
            # Retain only spectral features < 100. Angstroms.                                                                                                                                                                                 
            # dlambda per pixel = 0.8; 100A / dlambda per pixel = 125.                                                                                                                                                                        
            for i, ff in enumerate(self.ensemble_flux[band]):
                sflux                     = convolve(ff, Box1DKernel(125), boundary='extend')
                dflux[i,:]                = ff - sflux

            self.ensemble_dflux[band]     = dflux

        if sort and (tracer == 'ELG'):
            # Sort tracers for SOM-style plots, by redshift.                                                                                                                                                                                 
            indx                          = np.argsort(self.ensemble_meta, order=['REDSHIFT', 'OIIFLUX', 'MAG'])

            self.ensemble_meta            = self.ensemble_meta[indx]
            self.ensemble_objmeta         = self.ensemble_objmeta[indx]
        
            for band in ['b', 'r', 'z']:
                self.ensemble_flux[band]  = self.ensemble_flux[band][indx]
                self.ensemble_dflux[band] = self.ensemble_dflux[band][indx]

        # Write flux & dflux.                                                                                                                                                                                                                
        with open(self.ensemble_dir + '/template-{}-ensemble-flux.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_flux, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.ensemble_dir + '/template-{}-ensemble-dflux.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_dflux, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.ensemble_dir + '/template-{}-ensemble-meta.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.ensemble_dir + '/template-{}-ensemble-objmeta.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_objmeta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        end_genensemble = time.perf_counter()
    
        print('Template ensemble (nmode; {}) in {:.3f} mins.'.format(self.nmodel, (end_genensemble - start_genensemble) / 60.))

    def stack_ensemble(self, vet=True):
        '''                                                                                                                                                                                                                                  
        Stack the ensemble to an average.                                                                                                                                                                                                     
        '''
        self.ensemble_dflux_stack = {}

        for band in ['b', 'r', 'z']:
            self.ensemble_dflux_stack[band] = np.sqrt(np.mean(self.ensemble_dflux[band]**2., axis=0).reshape(1, len(self.ensemble_dflux[band].T)))

        if vet:
            import pylab as pl
            

            for band in ['b', 'r', 'z']:
                pl.plot(wave[cslice[band]], self.ensemble_dflux_stack[band].T)

            pl.show()

if __name__ == '__main__':
    mpath = '/global/cscratch1/sd/mjwilson/radlss/blanc/ensemble/template-bgs-ensemble-meta.fits'
    
    with open(mpath, 'rb') as handle: 
        meta = pickle.load(handle)
        
    #
    rads = template_ensemble(tracer='BGS', nmodel=5000, cached=True, sort=False, conditionals=meta, tile=1)
    rads.stack_ensemble()
