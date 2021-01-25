def qa_plots(self, plots_dir=None):
    '''                                                                                                                                                                                                                                   
    Generate QA plots under self.outdir(/QA/*).                                                                                                                                                                                            
    '''

    start_plotqa  = time.perf_counter()
    
    if plots_dir is None:
        plots_dir = self.qadir

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for cam in self.cameras:
        # ---------------  Skies  ---------------                                                                                                                                                                                          
        pl.clf()
        pl.plot(self.skies[cam].wave, self.skies[cam].flux[0, :].T)
        pl.xlabel('Wavelength [Angstroms]')
        pl.ylabel('Electrons per Angstrom')
        
        Path(plots_dir + '/skies/').mkdir(parents=True, exist_ok=True)

        pl.savefig(plots_dir + '/skies/sky-{}.pdf'.format(cam))

        # ---------------  Flux calibration  ---------------                                                                                                                                                                               
        fluxcalib          = self.fluxcalibs[cam].calib[0,:].T
        lossless_fluxcalib = fluxcalib / self.fiberloss

        wave               = self.cframes[cam].wave
            
        pl.clf()
        pl.plot(wave, fluxcalib,          label='Flux calib.')
        pl.plot(wave, lossless_fluxcalib, label='Lossless flux calib.')

        pl.title('FIBERLOSS of {:.2f}'.format(self.fiberloss))

        pl.legend(loc=0, frameon=False)

        Path(plots_dir + '/fluxcalib/').mkdir(parents=True, exist_ok=True)

        pl.savefig(plots_dir + '/fluxcalib/fluxcalib-{}.pdf'.format(cam))

        # ---------------  Fiber flats  ---------------                                                                                                                                                                                    
        xs = self.fibermaps[cam]['FIBERASSIGN_X']
        ys = self.fibermaps[cam]['FIBERASSIGN_Y']

        # Center on zero.                                                                                                                                                                                                                  
        fs = self.fiberflats[cam].fiberflat[:,0]

        pl.clf()
        pl.scatter(xs, ys, c=fs, vmin=0.0, vmax=2.0, s=2)

        Path(plots_dir + '/fiberflats/').mkdir(parents=True, exist_ok=True)

        pl.savefig(plots_dir + '/fiberflats/fiberflat-{}.pdf'.format(cam))
        
        # ---------------  Read noise ---------------                                                                                                                                                                                      
        # NOTE: PSF_WAVE == OII may only test, e.g. two quadrants.                                                                                                                                                                         
        pl.clf()
            
        fig, ax = plt.subplots(1, 1, figsize=(5., 5.))

        for shape, cam in zip(['s', 'o', '*'], self.cameras):
	    for color, quad in zip(['r', 'b', 'c', 'm'], ['A', 'B', 'C', 'D']):
                in_quad = self.fibermaps[cam]['QUAD'] == quad
                ax.plot(self.fibermaps[cam]['RDNOISE'][in_quad], self.fibermaps[cam]['RDNOISE_QUAD'][in_quad], marker='.', lw=0.0, c=color, markersize=4)

        xs = np.arange(10)
        
        ax.plot(xs, xs, alpha=0.4, c='k')

        ax.set_title('{} EXPID: {:d}'.format(self.flavor.upper(), self.expid))

        ax.set_xlim(2.0, 5.0)
        ax.set_ylim(2.0, 5.0)

        ax.set_xlabel('PSF median readnoise')
        ax.set_ylabel('Quadrant readnoise')

        Path(plots_dir + '/readnoise/').mkdir(parents=True, exist_ok=True)

        pl.savefig(plots_dir + '/readnoise/readnoise.pdf'.format(cam))

        plt.close(fig)

        # --------------- Template meta  ---------------                                                                                                                                                                                   
        Path(plots_dir + '/ensemble/').mkdir(parents=True, exist_ok=True)

        fig, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].hist(self.ensemble_meta['ELG']['REDSHIFT'], bins=50)
        axes[0].set_xlim(-0.05, 2.5)
        axes[0].set_xlabel('redshift')
        
        plt.close(fig)


        # --------------- Redrock:  Data  ---------------                                                                                                                                                                                  
        Path(plots_dir + '/redrock/').mkdir(parents=True, exist_ok=True)

        pl.clf()

        fig, axes = plt.subplots(1, len(self.tsnr_types), figsize=(20, 7.5))
        
        for i, (color, tsnr_type) in enumerate(zip(colors, self.tsnr_types)):
            xs      = []
	    ys      = []

            for petal in self.petals:
                xs   += self.tsnrs[tsnr_type][petal].tolist()
                ys   += self.zbests[petal]['DELTACHI2'].tolist()

                axes[i].loglog(self.tsnrs[tsnr_type][petal], self.zbests[petal]['DELTACHI2'], marker='.', lw=0.0, c=color, alpha=0.4)

		xs      = np.array(xs)
		ys      = np.array(ys)

                slope, intercept, r_value, p_value, std_err = stats.linregress(np.log10(xs[xs > 0.0]), np.log10(ys[xs > 0.0]))

		xs      = np.logspace(0., 4., 50)

		axes[i].loglog(xs, 10. ** (slope * np.log10(xs) + intercept), label='${:.3f} \cdot$ {} + {:.3f}'.format(slope, tsnr_type.upper(), intercept), c='k')

            for ax in axes:
                ax.set_xlim(1., 1.e4)
                ax.set_ylim(1., 1.e4)

		ax.set_xlabel('TEMPLATE SNR')

		ax.set_ylabel('$\Delta \chi^2$')

		ax.legend(loc=2, frameon=False)

		ax.set_title('EXPID: {:08d}'.format(self.expid))

            pl.savefig(plots_dir + '/redrock/redrock-data-tsnr-{}.pdf'.format(cam))

            plt.close(fig)

        Path(plots_dir + '/TSNRs/').mkdir(parents=True, exist_ok=True)

        for tracer in ['ELG']: # self.tracers                                                                                                                                                                                              
	    for petal in self.petals:
                pl.clf()

                fig, axes = plt.subplots(1, 4, figsize=(25,5))

                axes[0].imshow(np.log10(self.template_snrs[tracer]['brz{}'.format(petal)]['TSNR']), aspect='auto')

                axes[1].imshow(np.broadcast_to(self.ensemble_meta[tracer]['REDSHIFT'],   (500, self.nmodel[tracer])), aspect='auto')
                axes[2].imshow(np.broadcast_to(self.ensemble_meta[tracer]['MAG_G'],      (500, self.nmodel[tracer])), aspect='auto')
                axes[3].imshow(np.broadcast_to(self.ensemble_objmeta[tracer]['OIIFLUX'], (500, self.nmodel[tracer])), aspect='auto')

                axes[0].set_title('$\log_{10}$TSNR')
                axes[1].set_title('REDSHIFT')
                axes[2].set_title(r'$g_{\rm AB}$')
                axes[3].set_title('OIIFLUX')

                for ax in axes:
                    ax.set_xlabel('TEMPLATE #')
                    ax.set_ylabel('FIBER')

                fig.suptitle('Petal {} of EXPID: {:08d}'.format(petal, self.expid))

                pl.savefig(plots_dir + '/TSNRs/tsnr-{}-ensemble-meta-{}.pdf'.format(tracer, petal))
                
                plt.close(fig)

        end_plotqa  = time.perf_counter()
        
        print('Rank {}:  Created QA plots ({}) in {:.3f} mins.'.format(self.rank, plots_dir, (end_plotqa - start_plotqa) / 60.))
