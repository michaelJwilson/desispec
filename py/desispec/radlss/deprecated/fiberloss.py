def calc_fiberlosses(self):
        '''                                                                                                                                                                                                                                   
        Calculate the fiberloss for each target of a given camera from the gfa psf and legacy morphtype.  Added to self.fibermaps[cam].                                                                                                       
                                                                                                                                                                                                                                           
        Input:                                                                                                                                                                                                                                 
          psf_wave:  wavelength at which to evaluate the PSF, e.g. 3727. * (1. + 1.1)                                                                                                                                                          
        '''

        from  specsim.fastfiberacceptance import FastFiberAcceptance
        from  desimodel.io                import load_desiparams, load_fiberpos, load_platescale, load_tiles, load_deviceloc


        start_calcfiberloss = time.perf_counter()

        self.ffa = FastFiberAcceptance(self.desimodel + '/data/throughput/galsim-fiber-acceptance.fits')

	#- Platescales in um/arcsec                                                                                                                                                                                                            
        ps       = load_platescale()

        for cam in self.cameras:
            x            = self.cframes[cam].fibermap['FIBER_X']
            y            = self.cframes[cam].fibermap['FIBER_Y']

            r            = np.sqrt(x**2 + y**2)

            # ps['radius'] in mm.                                                                                                                                                                                                              
            radial_scale = np.interp(r, ps['radius'], ps['radial_platescale'])
            az_scale     = np.interp(r, ps['radius'], ps['az_platescale'])

            plate_scale  = np.sqrt(radial_scale, az_scale)

	    psf          = self.gfa_median_exposure['FWHM_ASEC']

            if np.isnan(psf):
                psf      = 1.2

		print('Ill defined PSF set to {:.2f} arcsecond for {} of {}.'.format(psf, cam, self.expid))

            # Gaussian assumption: FWHM_ARCSECOND to STD.                                                                                                                                                                                      
            psf         /= 2.355

            # um                                                                                                                                                                                                                               
            psf_um       = psf * plate_scale

            # POINT, DISK or BULGE for point source, exponential profile or De Vaucouleurs profile.                                                                                                                                            
            morph_type   = np.array(self.cframes[cam].fibermap['MORPHTYPE'], copy=True)

            self.cframes[cam].fibermap['FIBERLOSS'] = np.ones(len(self.cframes[cam].fibermap))

            psf_um       = psf_um * np.ones(len(self.cframes[cam].fibermap))
            fiber_offset = np.zeros_like(psf_um)

            for t, o in zip(['DEV', 'EXP', 'PSF', 'REX'], ['BULGE', 'DISK', 'POINT', 'DISK']):
                is_type  = (morph_type == t)

		# Optional:  half light radii.                                                                                                                                                                                                 
                self.cframes[cam].fibermap['FIBERLOSS'][is_type] = self.ffa.rms(o, psf_um[is_type], fiber_offset[is_type])

	end_calcfiberloss = time.perf_counter()

	print('Rank {}:  Calculated per-fiber fiberloss in {:.3f} mins.'.format(self.rank, (end_calcfiberloss - start_calcfiberloss) / 60.))
