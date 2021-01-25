def gen_template_ensemble(self, tracer='ELG', nmodel=500, cached=True, sort=True, conditioned=False):
    '''                                                                                                                                                                                                                                   
    Generate an ensemble of templates to sample tSNR for a range of points in                                                                                                                                                             
    (z, m, OII, etc.) space.                                                                                                                                                                                                              
                                                                                                                                                                                                                                              
    Uses cache on disk if possible, otherwise writes it.                                                                                                                                                                                                                                                                                                                                                                                                                     
    If conditioned, uses deepfield redshifts and (currently r) magnitudes to condition simulated templates.                                                                                                                               
    '''
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

    ##                                                                                                                                                                                                                                    
    start_genensemble  = time.perf_counter()
    
    self.ensemble_type = 'DESISIM'
    
    # If already available, load.                                                                                                                                                                                                         
    # TODO:  Simplify to one fits with headers?                                                                                                                                                                                           
    if cached and path.exists(self.ensemble_dir + '/template-{}-ensemble-flux.fits'.format(tracer.lower())):
        try:
            with open(self.ensemble_dir + '/template-{}-ensemble-flux.fits'.format(tracer.lower()), 'rb') as handle:
                self.ensemble_flux = pickle.load(handle)

            with open(self.ensemble_dir + '/template-{}-ensemble-dflux.fits'.format(tracer.lower()), 'rb') as handle:
                self.ensemble_dflux = pickle.load(handle)

            with open(self.ensemble_dir + '/template-{}-ensemble-meta.fits'.format(tracer.lower()), 'rb') as handle:
                self.ensemble_meta = pickle.load(handle)

            with open(self.ensemble_dir + '/template-{}-ensemble-objmeta.fits'.format(tracer.lower()), 'rb') as handle:
                self.ensemble_objmeta = pickle.load(handle)

            self.nmodel[tracer] = len(self.ensemble_flux[tracer][self.cameras[0][0]])

            if self.nmodel[tracer] != nmodel:
                # Pass to below.                                                                                                                                                                                                            
                raise ValueError('Retrieved ensemble had erroneous model numbers.')

            self.ensemble_tracers.append(tracer)

            print('Rank {}:  Successfully retrieved pre-written ensemble files at: {} (nmodel: {} == {}?)'.format(self.rank, self.ensemble_dir, nmodel, self.nmodel[tracer]))

            end_genensemble = time.perf_counter()

            print('Rank {}:  Template ensemble (nmodel: {}) in {:.3f} mins.'.format(self.rank, self.nmodel[tracer], (end_genensemble - start_genensemble) / 60.))

            # Successfully retrieved.                                                                                                                                                                                                        
            return

        except:
            print('Rank {}:  Failed to retrieve pre-written ensemble files.  Regenerating.'.format(self.rank))
            
    # Generate model dfluxes, limit to wave in each camera of expid.                                                                                                                                                                   
    keys                           = set([key for key in self.cframes.keys() if key[0] in ['b','r','z']])
    keys                           = np.array(list(keys))

    # TODO:  where are brz extraction wavelengths defined?  https://github.com/desihub/desispec/issues/1006.                                                                                                                             
    wave                           = self.cframes[keys[0]].wave

    if len(keys) > 1:
        for key in keys[1:]:
            wave                   = np.concatenate((wave, self.cframes[key].wave))

    wave                           = np.unique(wave) # sorted.                                                                                                                                                                            
    
    ##  ---------------------  Sim  ---------------------                                                                                                                                                                                 
    if not conditioned:
        wave, flux, meta, objmeta  = tracer_maker(wave=wave, tracer=tracer, nmodel=nmodel)

    else:
        # Condition simulated spectra on deepfield redshifts and magnitudes.                                                                                                                                                              
        from desiutil.dust import mwdust_transmission
            
        
        # Check we have deep field redshifts & fluxes.                                                                                                                                                                                    
        assert  (self.df_ensemble_zbests is not None)

        # Extinction correction for flux, band 'G', 'R', or 'Z'.                                                                                                                                                                          
        trans                      = mwdust_transmission(self.df_ensemble_allzbests['EBV'], 'R', self.df_ensemble_allzbests['PHOTSYS'])
            
        # Use deep field ensemble redshifts and magnitudes to condition desisim template call.                                                                                                                                            
        #                                                                                                                                                                                                                                 
        # TOD:  Defaults to r magnitude.  Vary by tracer?                                                                                                                                                                                 
        #                                                                                                                                                                                                                                 
        conditional_rmags          = 22.5 - 2.5 * np.log10(meta['FLUX_R'] / trans)
        conditional_zs             = self.df_ensemble_allzbests['Z']

        nmodel                     = len(conditional_zs)

        wave, flux, meta, objmeta  = tracer_maker(wave=wave, tracer=tracer, nmodel=nmodel, redshifts=conditional_zs, mags=conditional_rmags)

        print('Rank {}:  Assuming conditional redshifts and magnitudes for ensemble.'.format(self.rank))

	##  No extinction correction.                                                                                                                                                                                                        
        meta['MAG_G']                  = 22.5 - 2.5 * np.log10(meta['FLUX_G'])
        meta['MAG_R']                  = 22.5 - 2.5 * np.log10(meta['FLUX_R'])
        meta['MAG_Z']                  = 22.5 - 2.5 * np.log10(meta['FLUX_Z'])

	if tracer == 'ELG':
            meta['OIIFLUX']            = objmeta['OIIFLUX']

	self.nmodel[tracer]            = nmodel

	self.ensemble_flux[tracer]     = {}
        self.ensemble_dflux[tracer]    = {}
        self.ensemble_meta[tracer]     = meta
        self.ensemble_objmeta[tracer]  = objmeta

    # Generate template (d)fluxes for brz bands.                                                                                                                                                                                          
    for band in ['b', 'r', 'z']:
        band_key                          = [x[0] == band for x in keys]
        band_key                          = keys[band_key][0]
            
        band_wave                         = self.cframes[band_key].wave

        in_band                           = np.isin(wave, band_wave)

        self.ensemble_flux[tracer][band]  = flux[:, in_band]

        dflux                             = np.zeros_like(self.ensemble_flux[tracer][band])
        
        # Retain only spectral features < 100. Angstroms.                                                                                                                                                                                 
        # dlambda per pixel = 0.8; 100A / dlambda per pixel = 125.                                                                                                                                                                        
        for i, ff in enumerate(self.ensemble_flux[tracer][band]):
	    sflux                         = convolve(ff, Box1DKernel(125), boundary='extend')
            dflux[i,:]                    = ff - sflux

            self.ensemble_dflux[tracer][band] = dflux

        if sort and (tracer == 'ELG'):
            # Sort tracers for SOM-style plots, by redshift.                                                                                                                                                                                 
            indx                                  = np.argsort(self.ensemble_meta[tracer], order=['REDSHIFT', 'OIIFLUX', 'MAG'])

            self.ensemble_meta[tracer]            = self.ensemble_meta[tracer][indx]
            self.ensemble_objmeta[tracer]         = self.ensemble_objmeta[tracer][indx]

            for band in ['b', 'r', 'z']:
                self.ensemble_flux[tracer][band]  = self.ensemble_flux[tracer][band][indx]
		self.ensemble_dflux[tracer][band] = self.ensemble_dflux[tracer][band][indx]

	# Write flux & dflux.                                                                                                                                                                                                                
        with open(self.ensemble_dir + '/template-{}-ensemble-flux.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_flux, handle, protocol=pickle.HIGHEST_PROTOCOL)

	with open(self.ensemble_dir + '/template-{}-ensemble-dflux.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_dflux, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.ensemble_dir + '/template-{}-ensemble-meta.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_meta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        with open(self.ensemble_dir + '/template-{}-ensemble-objmeta.fits'.format(tracer.lower()), 'wb') as handle:
            pickle.dump(self.ensemble_objmeta, handle, protocol=pickle.HIGHEST_PROTOCOL)

        # List of tracers computed.                                                                                                                                                                                                         
        if tracer not in self.ensemble_tracers:
            self.ensemble_tracers.append(tracer)

        end_genensemble = time.perf_counter()

        print('Rank {}:  Template ensemble (nmode; {}) in {:.3f} mins.'.format(self.rank, self.nmodel[tracer], (end_genensemble - start_genensemble) / 60.))
    
