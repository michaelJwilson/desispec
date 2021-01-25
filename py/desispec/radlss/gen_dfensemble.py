import pandas                        as     pd

from   astropy.table                 import vstack, join
from   desispec.io.spectra           import read_spectra
from   desitarget.sv1.sv1_targetmask import bgs_mask as sv1_bgs_mask


def gen_df_ensemble(self, tracer='ELG', cached=False, sort=True, survey='SV1', dX2_lim=100., zlo=0.0013):
    start_deepfield                = time.perf_counter()

    self.ensemble_type             = survey

    # https://desi.lbl.gov/trac/wiki/SurveyValidation/TruthTables.                                                                                                                                                                         
    # Tile 67230 and night 20200315 (incomplete - 950/2932 targets).                                                                                                                                                                       
    vi_truthtable                  = '/global/cfs/cdirs/desi/sv/vi/TruthTables/truth_table_{}_v1.2.csv'.format(tracer)

    self.nmodel[tracer]            =  0
    
    if cached and path.exists(self.ensemble_dir + '/deepfield-{}-ensemble-flux.fits'.format(tracer.lower())):
        try:
            with open(self.ensemble_dir + '/deepfield-{}-ensemble-flux.fits'.format(tracer.lower()), 'rb') as handle:
                self.ensemble_flux = pickle.load(handle)

            with open(self.ensemble_dir + '/deepfield-{}-ensemble-dflux.fits'.format(tracer.lower()), 'rb') as handle:
                self.ensemble_dflux = pickle.load(handle)

            with open(self.ensemble_dir + '/deepfield-{}-ensemble-meta.fits'.format(tracer.lower()), 'rb') as handle:
                self.ensemble_meta = pickle.load(handle)

            with open(self.ensemble_dir + '/deepfield-{}-ensemble-objmeta.fits'.format(tracer.lower()), 'rb') as handle:
                self.ensemble_objmeta = pickle.load(handle)

	    self.nmodel[tracer] = len(self.ensemble_flux[tracer][self.cameras[0][0]])
            self.ensemble_tracers.append(tracer)

	    print('Rank {}:  Successfully retrieved pre-written ensemble files at: {} (nmodel: {})'.format(self.rank, self.ensemble_dir, self.nmodel[tracer]))

	    end_deepfield = time.perf_counter()

	    print('Rank {}:  Grabbed deepfield ensemble (nmodel: {}) in {:.3f} mins.'.format(self.rank, self.nmodel[tracer], (end_deepfield - start_deepfield) / 60.))

	    return

        except:
            print('Rank {}:  Failed to retrieve pre-written ensemble files.  Regenerating.'.format(self.rank))

    self.df_nmodel                 = 0

    self.df_ensemble_coadds        = {}
    self.df_ensemble_zbests        = {}

    self.df_ensemble_allzbests     = Table()
        
    self.ensemble_flux[tracer]     = {}
    self.ensemble_dflux[tracer]    = {}
    
    self.ensemble_meta[tracer]     = Table()
    self.ensemble_objmeta[tracer]  = Table()
        
    keys = []

    for key in self.cframes.keys():
        if key[0] == 'b':
            keys.append(key)
            break

    for key in self.cframes.keys():
        if key[0] == 'r':
            keys.append(key)
            break

    for key in self.cframes.keys():
        if key[0] == 'z':
            keys.append(key)
            break

    for band in ['b', 'r', 'z']:
        self.ensemble_flux[tracer][band]  = None
	self.ensemble_dflux[tracer][band] = None

    #  First, get the truth table.                                                                                                                                                                                                         
    df                             = pd.read_csv(vi_truthtable)
    names                          = [x.upper() for x in df.columns]

    truth                          = Table(df.to_numpy(), names=names)
    truth['TARGETID']              = np.array(truth['TARGETID']).astype(np.int64)
    truth['BEST QUALITY']          = np.array(truth['BEST QUALITY']).astype(np.int64)

    truth.sort('TARGETID')

    self._df_vitable               = truth

    df_tiles = {'BGS': ['80613', '80614', '80611', '80612', '80617']}

    for tileid in df_tiles[tracer]:
        for petal in np.arange(10).astype(np.str):
            # E.g.  /global/cfs/cdirs/desi/spectro/redux/blanc/tiles/80613/deep/                                                                                                                                                            
            deepfield_coadd_file                                = os.path.join(self.prod, 'tiles', tileid, 'deep', 'coadd-{}-{}-{}.fits'.format(petal, tileid, 'deep'))
            deepfield_zbest_file                                = os.path.join(self.prod, 'tiles', tileid, 'deep', 'zbest-{}-{}-{}.fits'.format(petal, tileid, 'deep'))

            self.df_ensemble_coadds[petal]                      = read_spectra(deepfield_coadd_file)
            self.df_ensemble_zbests[petal]                      = Table.read(deepfield_zbest_file, 'ZBEST')

            # fibermaps                                         = Table.read(deepfield_zbest_file, 'FIBERMAP')                                                                                                                               
            assert  np.all(self.df_ensemble_zbests[petal]['TARGETID'] == self.df_ensemble_coadds[petal].fibermap['TARGETID'])

            self.df_ensemble_zbests[petal]                      = join(self.df_ensemble_zbests[petal], self.df_ensemble_coadds[petal].fibermap, join_type='left', keys='TARGETID')
            self.df_ensemble_zbests[petal]                      = join(self.df_ensemble_zbests[petal], truth['TARGETID', 'BEST QUALITY'], join_type='left', keys='TARGETID')

            self.df_ensemble_zbests[petal]['IN_ENSEMBLE']       = (self.df_ensemble_zbests[petal]['ZWARN'] == 0)  & ( self.df_ensemble_zbests[petal]['FIBERSTATUS'] == 0)
            self.df_ensemble_zbests[petal]['IN_ENSEMBLE']       =  self.df_ensemble_zbests[petal]['IN_ENSEMBLE']  & ( self.df_ensemble_zbests[petal]['DELTACHI2'] > dX2_lim)
            self.df_ensemble_zbests[petal]['IN_ENSEMBLE']       =  self.df_ensemble_zbests[petal]['IN_ENSEMBLE']  & ( self.df_ensemble_zbests[petal]['Z'] > zlo) # Faux-stellar cut.                                                        
            self.df_ensemble_zbests[petal]['IN_ENSEMBLE']       =  self.df_ensemble_zbests[petal]['IN_ENSEMBLE']  & ((self.df_ensemble_zbests[petal]['SV1_BGS_TARGET']  & sv1_bgs_mask['BGS_FAINT']) != 0)

            '''                                                                                                                                                                                                                               
            # TARGETIDs not in the truth table have masked elements in array.                                                                                                                                                                 
            unmasked                                            = ~self.df_ensemble_zbests[petal]['BEST QUALITY'].mask                                                                                                                       
                                                                                                                                                                                                    
            self.df_ensemble_zbests[petal]['IN_ENSEMBLE'][unmasked] =  self.df_ensemble_zbests[petal]['IN_ENSEMBLE'][unmasked] & (self.df_ensemble_zbests[petal]['BEST QUALITY'][unmasked] >= 2.5)                                      
            '''

            # Stack those making ensemble cut.                                                                                                                                                                                                
            cols = ['TARGETID', 'FLUX_G', 'FLUX_R', 'FLUX_Z', 'EBV', 'PHOTSYS', 'Z', 'ZERR', 'DELTACHI2', 'NCOEFF', 'COEFF']
            self.ensemble_meta[tracer] = vstack((self.ensemble_meta[tracer], self.df_ensemble_zbests[petal][cols][self.df_ensemble_zbests[petal]['IN_ENSEMBLE']]))

            for key in keys:
                band                                 = key[0]

                if self.ensemble_flux[tracer][band] is None:
                    self.ensemble_flux[tracer][band] = self.df_ensemble_coadds[petal].flux[band][self.df_ensemble_zbests[petal]['IN_ENSEMBLE'], :]

                else:
                    self.ensemble_flux[tracer][band] = np.vstack((self.ensemble_flux[tracer][band], self.df_ensemble_coadds[petal].flux[band][self.df_ensemble_zbests[petal]['IN_ENSEMBLE'],:]))

            self.nmodel[tracer] += np.count_nonzero(self.df_ensemble_zbests[petal]['IN_ENSEMBLE'])

          # Compress across petals:                                                                                                                                                                                                            
          petals = list(self.df_ensemble_zbests.keys())

          for petal in petals:
              self.df_ensemble_allzbests = vstack((self.df_ensemble_allzbests, self.df_ensemble_zbests[petal][self.df_ensemble_zbests[petal]['IN_ENSEMBLE']]))

              del self.df_ensemble_zbests[petal]

	for band in ['b', 'r', 'z']:
            dflux = np.zeros_like(self.ensemble_flux[tracer][band])

            # Retain only spectral features < 100. Angstroms.                                                                                                                                                                                  
            # dlambda per pixel = 0.8; 100A / dlambda per pixel = 125.                                                                                                                                                                         
            for i, ff in enumerate(self.ensemble_flux[tracer][band]):
                sflux      = convolve(ff, Box1DKernel(125), boundary='extend')
                dflux[i,:] = ff - sflux

            self.ensemble_dflux[tracer][band]     = dflux

	# TO DO: MW Transmission correction.                                                                                                                                                                                                 
        self.ensemble_meta[tracer]['MAG_G']       = 22.5 - 2.5 * np.log10(self.ensemble_meta[tracer]['FLUX_G'])
        self.ensemble_meta[tracer]['MAG_R']       = 22.5 - 2.5 * np.log10(self.ensemble_meta[tracer]['FLUX_R'])
        self.ensemble_meta[tracer]['MAG_Z']       = 22.5 - 2.5 * np.log10(self.ensemble_meta[tracer]['FLUX_Z'])

	self.ensemble_meta[tracer]['REDSHIFT']    = self.ensemble_meta[tracer]['Z']
        self.ensemble_objmeta[tracer]['OIIFLUX']  = np.zeros(len(self.ensemble_meta[tracer]))

	del self.ensemble_meta[tracer]['Z']

	if sort:
            # Sort tracers for SOM-style plots.                                                                                                                                                                                               
            indx                                  = np.argsort(self.ensemble_meta[tracer], order=['REDSHIFT'])

            self.ensemble_meta[tracer]            = self.ensemble_meta[tracer][indx]
            self.ensemble_objmeta[tracer]         = self.ensemble_objmeta[tracer][indx]

            for band in ['b', 'r', 'z']:
                self.ensemble_flux[tracer][band]  = self.ensemble_flux[tracer][band][indx]
                self.ensemble_dflux[tracer][band] = self.ensemble_dflux[tracer][band][indx]

        #
        with open(self.ensemble_dir + '/deepfield-{}-ensemble-flux.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_flux, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(self.ensemble_dir + '/deepfield-{}-ensemble-dflux.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_dflux, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(self.ensemble_dir + '/deepfield-{}-ensemble-meta.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(self.ensemble_dir + '/deepfield-{}-ensemble-objmeta.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_objmeta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # List of tracers computed.                                                                                                                                                                                                           
        if tracer not in self.ensemble_tracers:
            self.ensemble_tracers.append(tracer)

        end_deepfield = time.perf_counter()

        print('Rank {}:  Writing to {}.'.format(self.rank, self.ensemble_dir + '/deepfield-{}-ensemble-flux.fits'.format(tracer.lower())))
        print('Rank {}:  Created deepfield ensemble of {} galaxies in {:.3f} mins.'.format(self.rank, self.nmodel[tracer], (end_deepfield - start_deepfield) / 60.))
