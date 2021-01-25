def stack_ensemble(self, vet=False):
    '''                                                                                                                                                                                                                                    
    Stack the ensemble to an average.                                                                                                                                                                                                      
    '''
    for tracer in self.ensemble_tracers:
	for band in ['b', 'r', 'z']:
            self.ensemble_dflux[tracer][band] = np.sqrt(np.mean(self.ensemble_dflux[tracer][band]**2., axis=0).reshape(1, len(self.ensemble_dflux[tracer][band].T)))
            
            self.ensemble_flux[tracer][band]  = None
            self.ensemble_meta[tracer]        = None
            
            self.nmodel[tracer] = 1

    if vet:
        petals = list(self.df_ensemble_coadds.keys())
            
        if len(petals) > 0:
            petal  = petals[0]

            for band in ['b', 'r', 'z']:
                pl.plot(self.df_ensemble_coadds[petal].wave[band], self.ensemble_dflux[tracer][band].T)
                
            pl.show()
