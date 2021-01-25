    def reconstruct_rr_cframes(self):
        '''                                                                                                                                                                                                                                    
        Reconstruct best-fit redrock template from data ZBEST.                                                                                                                                                                                 
        Use this to calculate sourceless IVAR & resulting TSNR.                                                                                                                                                                                
        '''

        start_rrcframe  = time.perf_counter()

        # Retrieve redrock PCA templates.                                                                                                                                                                                                      
        if not bool(self.templates):
            for filename in redrock.templates.find_templates():
                t       = redrock.templates.Template(filename)
                self.templates[(t.template_type, t.sub_type)] = t

            print('Updated rr templates.')

        templates       = self.templates

        self.rr_cframes = {}
        self.zbests     = {}
        self.zbests_fib = {}

        for cam in self.cameras:
            petal       = cam[1]

            # E.g. /global/scratch/mjwilson/desi-data.dm.noao.edu/desi/spectro/redux/andes/tiles/67230/20200314.                                                                                                                               
            self.zbest_path        = os.path.join(self.prod, 'tiles', str(self.tileid), str(self.night), 'zbest-{}-{}-{}.fits'.format(petal, self.tileid, self.night))

            self.zbests[petal]     = zbest = Table.read(self.zbest_path, 'ZBEST')
            self.zbests_fib[petal] = Table.read(self.data_zbest_file, 'FIBERMAP')

            # Limit to a single exposure.                                                                                                                                                                                                      
            self.zbests_fib[petal] = self.zbests_fib[petal][self.zbests_fib[petal]['EXPID'] == self.expid]

            #                                                                                                                                                                                                                                  
            self.zbests_fib[petal].sort('TARGETID')

            # Sorting by zbest-style TARGETID to fibermap-style FIBER.                                                                                                                                                                         
            indx       = np.argsort(self.zbests_fib[petal]['FIBER'].data)

            # Sorted by TARGETID.                                                                                                                                                                                                              
            rr_z       = zbest['Z']

            spectype   = [x.strip() for x in zbest['SPECTYPE']]
            subtype    = [x.strip() for x in zbest['SUBTYPE']]

            fulltype   = list(zip(spectype, subtype))

            ncoeff     = [templates[ft].flux.shape[0] for ft in fulltype]

            coeff      = [x[0:y] for (x,y) in zip(zbest['COEFF'], ncoeff)]

            # Each list does not have the same length.                                                                                                                                                                                         
            tfluxs     = [templates[ft].flux.T.dot(cf).tolist()     for (ft, cf) in zip(fulltype, coeff)]
            twaves     = [(templates[ft].wave * (1. + rz)).tolist() for (ft, rz) in zip(fulltype, rr_z)]

            # Sort by FIBER rather than TARGETID.                                                                                                                                                                                              
            tfluxs     = [tfluxs[ind] for ind in indx]
            twaves     = [twaves[ind] for ind in indx]

            # FIBER order.                                                                                                                                                                                                                     
            Rs         = [Resolution(x) for x in self.cframes[cam].resolution_data]
            txfluxs    = [R.dot(resample_flux(self.cframes[cam].wave, twave, tflux)) for (R, twave, tflux) in zip(Rs, twaves, tfluxs)]
            txfluxs    =  np.array(txfluxs)

            # txflux  *= self.fluxcalibs[cam].calib     # [e/A].                                                       
            # txflux  /= self.fiberflats[cam].fiberflat # [e/A]. (Instrumental).                                                                                                                                                               

            # Estimated sourceless IVAR [ergs/s/A] ...                                                                                                                                                                                         
            # Note:  assumed Poisson in Flux.                                                                                                                                                                                                  

            # Sourceless-IVAR from sky subtracted spectrum.                                                                                                                                                                                    
            sless_ivar    = np.zeros_like(self.cframes[cam].ivar)

            # Sourceless-IVAR from redrock template fit to spectrum.                                                                                                                                                                           
            rrsless_ivar  = np.zeros_like(self.cframes[cam].ivar)

            # **  Sky-subtracted ** realized flux.                                                                                                                                                                                             
            sflux         = self.cframes[cam].flux[self.cframes[cam].mask == 0] / self.fluxcalibs[cam].calib[self.cframes[cam].mask == 0] / self.fiberflats[cam].fiberflat[self.cframes[cam].mask == 0]
            sless_ivar[self.cframes[cam].mask == 0] = 1. / ((1. / self.cframes[cam].ivar[self.cframes[cam].mask == 0]) - sflux)

            # Best-fit redrock template to the flux.                                                                                                                                                                                           
            sflux         = txfluxs[self.cframes[cam].mask == 0] / self.fluxcalibs[cam].calib[self.cframes[cam].mask == 0] / self.fiberflats[cam].fiberflat[self.cframes[cam].mask == 0]
            rrsless_ivar[self.cframes[cam].mask == 0] = 1. / ((1. / self.cframes[cam].ivar[self.cframes[cam].mask == 0]) - sflux)

            # https://github.com/desihub/desispec/blob/6c9810df3929b8518d49bad14356d35137d25a89/py/desispec/frame.py#L41                                                                                                                       
            self.rr_cframes[cam] = Frame(self.cframes[cam].wave, txfluxs, rrsless_ivar, mask=self.cframes[cam].mask,\
                                         resolution_data=self.cframes[cam].resolution_data, fibermap=self.cframes[cam].fibermap,\
                                         meta=self.cframes[cam].meta)

            # Raw IVAR.                                                                                                                                                                                                                        
            rr_tsnrs             = []

            # Sourceless IVAR - using sky subtracted spectrum.                                                                                                                                                                                 
            rr_tsnrs_sless       = []

            # Sourceless IVAR - using redrock best-fit subtracted spectrum.                                                                                                                                                                    
            rr_tsnrs_rrsless     = []

            # Sourceless IVAR - using ccdmodel, e.g. sky_flux, flux_calib, fiberflat, readnoise, npix, angstroms per pixel.                                                                                                                    
            rr_tsnrs_modelivar   = []

            for j, template_flux in enumerate(self.rr_cframes[cam].flux):
                # Best-fit redrock template & cframe ivar  [Calibrated flux].                                                                                                                                                                  
                rr_tsnrs.append(templateSNR(template_flux, flux_ivar=self.cframes[cam].ivar[j,:]))

		# Best-fit redrock template & sourceless cframe ivar (cframe subtracted) [Calibrated flux].                                                                                                                                    
                rr_tsnrs_sless.append(templateSNR(template_flux, flux_ivar=sless_ivar[j,:]))

		# Best-fit redrock template & sourceless cframe ivar (rr best-fit cframe subtracted) [Calibrated flux].                                                                                                                        
                rr_tsnrs_rrsless.append(templateSNR(template_flux, flux_ivar=rrsless_ivar[j,:]))

		# Model IVAR:                                                                                                                                                                                                                  
                sky_flux            = self.skies[cam].flux[j,:]
                flux_calib          = self.fluxcalibs[cam].calib[j,:]
                fiberflat           = self.fiberflats[cam].fiberflat[j,:]
                readnoise           = self.cframes[cam].fibermap['RDNOISE'][j]
                npix                = self.cframes[cam].fibermap['NEA'][j]
                angstroms_per_pixel = self.cframes[cam].fibermap['ANGSTROMPERPIXEL'][j]

		rr_tsnrs_modelivar.append(templateSNR(template_flux, sky_flux=sky_flux, flux_calib=flux_calib, fiberflat=fiberflat, readnoise=readnoise, npix=npix,\
                                                      angstroms_per_pixel=angstroms_per_pixel, fiberloss=None, flux_ivar=None))

            self.cframes[cam].fibermap['TSNR']           = np.array(rr_tsnrs)
            self.cframes[cam].fibermap['TSNR_SLESS']     = np.array(rr_tsnrs_sless)
            self.cframes[cam].fibermap['TSNR_RRSLESS']   = np.array(rr_tsnrs_rrsless)
            self.cframes[cam].fibermap['TSNR_MODELIVAR'] = np.array(rr_tsnrs_modelivar)

            # Reduce over cameras, for each petal.                                                                                                                                                                                     
            self.tsnrs                  = {}
            self.tsnr_types             = ['TSNR', 'TSNR_SLESS', 'TSNR_RRSLESS', 'TSNR_MODELIVAR']

            for tsnr_type in self.tsnr_types:
                self.tsnrs[tsnr_type]   = {}

	    for tsnr_type in self.tsnr_types:
                for cam in self.cameras:
                    petal  = cam[1]

                    if petal not in self.tsnrs[tsnr_type].keys():
                        self.tsnrs[tsnr_type][petal] = np.zeros(len(self.cframes[cam].fibermap))

                    self.tsnrs[tsnr_type][petal] += self.cframes[cam].fibermap[tsnr_type]

            end_rrcframe = time.perf_counter()

            print('Rank {}:  Calculated redrock cframe in {:.3f} mins.'.format(self.rank, (end_rrcframe - start_rrcframe) / 60.))
