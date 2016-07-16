""" Class to organize QA for a full DESI production run
"""

from __future__ import print_function, absolute_import, division, unicode_literals

import numpy as np
import os, glob
import pdb

from desispec.io import get_exposures
from desispec.io import get_files
from desispec.io import read_frame
from desispec.io import meta

from desispec.log import get_logger

log=get_logger()


class QA_Prod(object):
    def __init__(self, specprod_dir, in_data=None):
        """ Class to organize and execute QA for a DESI production

        Args:
            expid: int -- Exposure ID
            night: str -- YYYYMMDD
            flavor: str
              exposure type (e.g. flat, arc, science)
            specprod_dir(str): Path containing the exposures/ directory to use. If the value
                is None, then the value of :func:`specprod_root` is used instead.
            in_data: dict, optional -- Input data
              Mainly for reading from disk

        Notes:

        Attributes:
            All input args become object attributes.
        """
        self.specprod_dir = specprod_dir

    def remake_frame_qa(self, remake_plots=False):
        """ Work through the Production and remake QA for all frames

        Parameters:
            remake_plots: bool, optional
              Remake the plots too?
        Returns:

        """
        # imports
        from desispec.io.qa import load_qa_frame, write_qa_frame
        from desispec.io.fiberflat import read_fiberflat
        from desispec.io.sky import read_sky
        from desispec.io.fluxcalibration import read_flux_calibration
        from desispec.qa import qa_plots
        from desispec.io.fluxcalibration import read_stdstar_models

        # Loop on nights
        path_nights = glob.glob(self.specprod_dir+'/exposures/*')
        nights = [ipathn[ipathn.rfind('/')+1:] for ipathn in path_nights]
        for night in nights:
            for exposure in get_exposures(night, specprod_dir = self.specprod_dir):
                # Object only??
                frames_dict = get_files(filetype = str('frame'), night = night,
                        expid = exposure, specprod_dir = self.specprod_dir)
                for camera,frame_fil in frames_dict.items():
                    # Load frame
                    frame = read_frame(frame_fil)
                    if frame.meta['FLAVOR'] in ['flat','arc']:
                        qatype = 'qa_calib'
                    else:
                        qatype = 'qa_data'
                    qafile = meta.findfile(qatype, night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                    # Load
                    qaframe = load_qa_frame(qafile, frame, flavor=frame.meta['FLAVOR'])
                    # Flat QA
                    if frame.meta['FLAVOR'] in ['flat']:
                        fiberflat_fil = meta.findfile('fiberflat', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                        fiberflat = read_fiberflat(fiberflat_fil)
                        qaframe.run_qa('FIBERFLAT', (frame, fiberflat))
                        if remake_plots:
                            # Do it
                            qafig = meta.findfile('qa_flat_fig', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                            qa_plots.frame_fiberflat(qafig, qaframe, frame, frame.fibermap, fiberflat)
                    # SkySub QA
                    if qatype == 'qa_data':
                        sky_fil = meta.findfile('sky', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                        skymodel = read_sky(sky_fil)
                        qaframe.run_qa('SKYSUB', (frame, skymodel))
                        if remake_plots:
                            qafig = meta.findfile('qa_sky_fig', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                            qa_plots.frame_skyres(qafig, frame, skymodel, qaframe)
                    # FluxCalib QA
                    if qatype == 'qa_data':
                        # Standard stars
                        stdstar_fil = meta.findfile('stdstars', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                        model_tuple=read_stdstar_models(stdstar_fil)
                        flux_fil = meta.findfile('calib', night=night, camera=camera, expid=exposure, specprod_dir=self.specprod_dir)
                        fluxcalib = read_flux_calibration(flux_fil)
                        qaframe.run_qa('FLUXCALIB', (frame, fluxcalib, model_tuple))#, indiv_stars))
                    # Write
                    write_qa_frame(qafile, qaframe)

    def slurp(self, remake=False, remove=True):
        """ Slurp all the individual QA files into one master QA file
        Args:
            remake: bool, optional
              Regenerate the individual QA files (at the frame level first)
            remove: bool, optional
              Remove

        Returns:

        """

    def __repr__(self):
        """ Print formatting
        """
        return ('{:s}: specprod_dir={:s}'.format(self.__class__.__name__, self.specprod_dir))
