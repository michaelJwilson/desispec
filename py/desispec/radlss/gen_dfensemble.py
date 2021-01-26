import time
import pickle
import numpy                         as     np
import pandas                        as     pd

from   os                            import path
from   astropy.table                 import vstack, join
from   desispec.io.spectra           import read_spectra
from   desitarget.sv1.sv1_targetmask import bgs_mask as sv1_bgs_mask
from   pathlib                       import Path
from   astropy.table                 import Table
from   astropy.convolution           import convolve, Box1DKernel
from   desiutil.dust                 import mwdust_transmission

# AR/DK DESI spectra wavelengths                                                                                                                                                                                                              
wmin, wmax, wdelta = 3600, 9824, 0.8
fullwave           = np.round(np.arange(wmin, wmax + wdelta, wdelta), 1)
cslice             = {"b": slice(0, 2751), "r": slice(2700, 5026), "z": slice(4900, 7781)}

def class_cuts(zbest, tracer='BGS'):    
    if tracer == 'BGS':
        isin = ((zbest['SV1_BGS_TARGET'] & sv1_bgs_mask['BGS_BRIGHT']) != 0)
        isin = isin | ((zbest['SV1_BGS_TARGET'] & sv1_bgs_mask['BGS_FAINT'])  != 0)

    else:
        raise ValueError('Invalid tracer {} passed to ensemble class_cuts.')

    return  isin
    
class df_ensemble(object):
    def __init__(self, tracer='BGS', cached=False, sort=True, survey='SV1', dX2_lim=100., zlo=0.0013, prod='/global/cfs/cdirs/desi/spectro/redux/blanc',\
                                                   out_dir='/global/cscratch1/sd/mjwilson/radlss/', vi_dir='/global/cfs/cdirs/desi/sv/vi/TruthTables/'):

        start_deepfield       = time.perf_counter()

        self.ensemble_type    = survey

        # https://desi.lbl.gov/trac/wiki/SurveyValidation/TruthTables.                                                                                                                                                                     
        # Tile 67230 and night 20200315 (incomplete - 950/2932 targets).                                                                                                                                                                    
        vi_truthtable         = vi_dir + '/truth_table_{}_v1.2.csv'.format(tracer)

        self.nmodel           = 0
        self.tracer           = tracer

        # Strip e.g. blanc from prod. path. 
        self.ensemble_dir     = '{}/ensemble/'.format(out_dir)

        Path(self.ensemble_dir).mkdir(parents=True, exist_ok=True)
        
        if cached:
            try:
                with open(self.ensemble_dir + '/deepfield-{}-ensemble-flux.fits'.format(tracer.lower()), 'rb') as handle:
                    self.ensemble_flux = pickle.load(handle)

                with open(self.ensemble_dir + '/deepfield-{}-ensemble-dflux.fits'.format(tracer.lower()), 'rb') as handle:
                    self.ensemble_dflux = pickle.load(handle)

                with open(self.ensemble_dir + '/deepfield-{}-ensemble-meta.fits'.format(tracer.lower()), 'rb') as handle:
                    self.ensemble_meta = pickle.load(handle)

                with open(self.ensemble_dir + '/deepfield-{}-ensemble-objmeta.fits'.format(tracer.lower()), 'rb') as handle:
                    self.ensemble_objmeta = pickle.load(handle)

                self.nmodel = len(self.ensemble_flux['b'])

                print('Successfully retrieved pre-written ensemble files at: {} (nmodel: {})'.format(self.ensemble_dir, self.nmodel))
                
                end_deepfield = time.perf_counter()

                print('Grabbed deepfield ensemble (nmodel: {}) in {:.3f} mins.'.format(self.nmodel, (end_deepfield - start_deepfield) / 60.))

                return

            except:
                print('Failed to retrieve pre-written ensemble files from {}.  Regenerating.'.format(self.ensemble_dir + '/deepfield-{}-ensemble-flux.fits'.format(tracer.lower())))

        # --  Regenerate --
        # By petal.
        self.df_ensemble_coadds        = {}
        self.df_ensemble_zbests        = {}

        # All.
        self.df_ensemble_allzbests     = Table()
        
        self.ensemble_flux             = {}
        self.ensemble_dflux            = {}
        
        self.ensemble_meta             = Table()
        self.ensemble_objmeta          = Table()

        for band in ['b', 'r', 'z']:
            self.ensemble_flux[band]   = None
            self.ensemble_dflux[band]  = None
        
        #  First, get the truth table.                                                                                                                                                                                                       
        df                             = pd.read_csv(vi_truthtable)
        names                          = [x.upper() for x in df.columns]
        
        truth                          = Table(df.to_numpy(), names=names)
        truth['TARGETID']              = np.array(truth['TARGETID']).astype(np.int64)
        truth['BEST QUALITY']          = np.array(truth['BEST QUALITY']).astype(np.int64)

        truth.sort('TARGETID')

        self._df_vitable               = truth

        cols     = ['TARGETID', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'EBV', 'PHOTSYS', 'Z', 'ZERR', 'DELTACHI2', 'NCOEFF', 'COEFF']

        df_tiles = {'BGS': ['80611', '80612', '80613', '80614', '80616', '80617', '80618', '80619']}

        for tileid in df_tiles[tracer]:
            for petal in np.arange(10).astype(np.str):
                # E.g.  /global/cfs/cdirs/desi/spectro/redux/blanc/tiles/80613/deep/                                                                                                                                                  
                deepfield_coadd_file                                = path.join(prod, 'tiles', tileid, 'deep', 'coadd-{}-{}-{}.fits'.format(petal, tileid, 'deep'))
                deepfield_zbest_file                                = path.join(prod, 'tiles', tileid, 'deep', 'zbest-{}-{}-{}.fits'.format(petal, tileid, 'deep'))
            
                self.df_ensemble_coadds[petal]                      = read_spectra(deepfield_coadd_file)
                self.df_ensemble_zbests[petal]                      = Table.read(deepfield_zbest_file, 'ZBEST')

                assert  np.all(self.df_ensemble_zbests[petal]['TARGETID'] == self.df_ensemble_coadds[petal].fibermap['TARGETID'])

                self.df_ensemble_zbests[petal]                      = join(self.df_ensemble_zbests[petal], self.df_ensemble_coadds[petal].fibermap, join_type='left', keys='TARGETID')
                self.df_ensemble_zbests[petal]                      = join(self.df_ensemble_zbests[petal], truth['TARGETID', 'BEST QUALITY'], join_type='left', keys='TARGETID')

                # Apply cuts to be in the df ensemble.  Currently, BGS specific. 
                self.df_ensemble_zbests[petal]['IN_ENSEMBLE']       = (self.df_ensemble_zbests[petal]['ZWARN'] == 0)  & ( self.df_ensemble_zbests[petal]['FIBERSTATUS'] == 0)
                self.df_ensemble_zbests[petal]['IN_ENSEMBLE']       =  self.df_ensemble_zbests[petal]['IN_ENSEMBLE']  & ( self.df_ensemble_zbests[petal]['DELTACHI2'] > dX2_lim)
                self.df_ensemble_zbests[petal]['IN_ENSEMBLE']       =  self.df_ensemble_zbests[petal]['IN_ENSEMBLE']  & ( self.df_ensemble_zbests[petal]['Z'] > zlo) # Faux-stellar cut.                                                   
                self.df_ensemble_zbests[petal]['IN_ENSEMBLE']       =  self.df_ensemble_zbests[petal]['IN_ENSEMBLE']  & class_cuts(self.df_ensemble_zbests[petal], tracer=tracer)
                
                # Stack 'meta' info. for the ensemble, as for the simulated targets. 
                self.ensemble_meta = vstack((self.ensemble_meta, self.df_ensemble_zbests[petal][cols][self.df_ensemble_zbests[petal]['IN_ENSEMBLE']]))

                for band in ['b', 'r', 'z']:
                    if self.ensemble_flux[band] is None:
                        self.ensemble_flux[band] = self.df_ensemble_coadds[petal].flux[band][self.df_ensemble_zbests[petal]['IN_ENSEMBLE'], :]

                    else:
                        self.ensemble_flux[band] = np.vstack((self.ensemble_flux[band], self.df_ensemble_coadds[petal].flux[band][self.df_ensemble_zbests[petal]['IN_ENSEMBLE'],:]))

                self.nmodel += np.count_nonzero(self.df_ensemble_zbests[petal]['IN_ENSEMBLE'])

            # Compress across petals:                                                                                                                                                                                              
            petals = list(self.df_ensemble_zbests.keys())
          
            for petal in petals:
                self.df_ensemble_allzbests = vstack((self.df_ensemble_allzbests, self.df_ensemble_zbests[petal][self.df_ensemble_zbests[petal]['IN_ENSEMBLE']]))
                
                del self.df_ensemble_zbests[petal]

        for band in ['b', 'r', 'z']:
            dflux = np.zeros_like(self.ensemble_flux[band])
                
            # Retain only spectral features < 100. Angstroms.                                                                                                                                                          
            # dlambda per pixel = 0.8; 100A / dlambda per pixel = 125.                                                                                                                                                                         
            for i, ff in enumerate(self.ensemble_flux[band]):
                sflux      = convolve(ff, Box1DKernel(125), boundary='extend')
                dflux[i,:] = ff - sflux

            self.ensemble_dflux[band] = dflux

        self.ensemble_objmeta['OIIFLUX']  = np.zeros(len(self.ensemble_meta))

        self.ensemble_meta['REDSHIFT']    = self.ensemble_meta['Z']

        del self.ensemble_meta['Z']

        if sort:
            # Sort tracers for SOM-style plots.                                                                                                                                                                             
            indx                          = np.argsort(self.ensemble_meta, order=['REDSHIFT'])
            
            self.ensemble_meta            = self.ensemble_meta[indx]
            self.ensemble_objmeta         = self.ensemble_objmeta[indx]

            for band in ['b', 'r', 'z']:
                self.ensemble_flux[band]  = self.ensemble_flux[band][indx]
                self.ensemble_dflux[band] = self.ensemble_dflux[band][indx]
    
        with open(self.ensemble_dir + '/deepfield-{}-ensemble-flux.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_flux, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.ensemble_dir + '/deepfield-{}-ensemble-dflux.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_dflux, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.ensemble_dir + '/deepfield-{}-ensemble-meta.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.ensemble_dir + '/deepfield-{}-ensemble-objmeta.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_objmeta, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        end_deepfield = time.perf_counter()
        
        print('Writing to {}.'.format(self.ensemble_dir + '/deepfield-{}-ensemble-flux.fits'.format(tracer.lower())))
        print('Created deepfield ensemble of {} galaxies in {:.3f} mins.'.format(self.nmodel, (end_deepfield - start_deepfield) / 60.))

    def stack_ensemble(self, vet=True):
        '''                                                                                                                                                                                                                              
        Stack the ensemble to an average.                                                                                                                                                                                                
        '''
        import pylab as pl

        self.ensemble_dflux_stack = {}
        
        for band in ['b', 'r', 'z']:
            self.ensemble_dflux_stack[band] = np.sqrt(np.mean(self.ensemble_dflux[band]**2., axis=0).reshape(1, len(self.ensemble_dflux[band].T)))

        with open(self.ensemble_dir + '/deepfield-{}-ensemble-dflux-stack.fits'.format(self.tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_dflux_stack, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        if vet:
            for band in ['b', 'r', 'z']:
                pl.plot(fullwave[cslice[band]], self.ensemble_dflux_stack[band].T)

            pl.show()
        

if __name__ == '__main__':
    rads = df_ensemble(tracer='BGS', cached=True, sort=False, survey='SV1', dX2_lim=100., zlo=0.0013, out_dir='/global/cscratch1/sd/mjwilson/radlss/blanc/', vi_dir='/global/cfs/cdirs/desi/sv/vi/TruthTables/')
    rads.stack_ensemble()
    
    print('Done.')
