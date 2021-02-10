import numba
import epix

# Fixed detector dimensions of XENONnT:
# See also: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenonnt:analysis:coordinate_system
xenonnt_sensitive_volume_radius = 66.4  # cm
xenonnt_z_gate_mesh = 0.  # bottom of the gate electrode
xenonnt_z_top_pmts = 7.3936  # cm
xenonnt_z_cathode = -148.6515  # cm ... top of the cathode electrode
xenonnt_z_bottom_pmts = -154.6555  # cm ... top surface of the bottom PMT window
xenonnt_z_lxe = 0.416 # cm ... liquid-gas interface

# Fixed detector dimensions of XENON1T:
# See also: https://xe1t-wiki.lngs.infn.it/doku.php?id=xenon:xenon1t:analysis:coordinate_system
xenon1t_sensitive_volume_radius = 47.925  # cm
xenon1t_z_gate_mesh = 0.  # bottom of the gate electrode
xenon1t_z_top_pmts = 7.38  # cm
xenon1t_z_cathode = -96.87  # cm ... top of the cathode electrode
xenon1t_z_bottom_pmts = -103.09  # cm ... top surface of the bottom PMT window
xenon1t_z_lxe = 0.27 # cm ... liquid-gas interface


def xenonnt():
    """
    Default XENONnT TPC with different sensitive volumes.

    The structure for each volume is as follows:
        key: VolumeName
        items: keyword arguments for the Sensitive Volume class

    :return: A list of volumes for the default nT detector.
    """
    # TODO: Can we do this differently? Ideas are welcome....
    # Outer edges of the volume of interest. This is needed for a
    # preselection of interactions inside the desired volumes before
    # clustering.
    outer_cylinder = {'max_z': xenonnt_z_top_pmts,
                      'min_z': xenonnt_z_bottom_pmts,
                      'max_r': xenonnt_sensitive_volume_radius
                      }

    volumes = {'TPC': {'volume_id': 1,
                       'roi': _make_roi_cylinder(xenonnt_z_cathode,
                                                 xenonnt_z_gate_mesh,
                                                 xenonnt_sensitive_volume_radius),
                       'electric_field': 200,
                       'create_S2': True,
                       'xe_density': 2.862,
                       'efield_outside_map': 200,
                       'to_be_stored': True,
                       },
               'BelowCathode': {'volume_id': 2,
                                'roi': _make_roi_cylinder(xenonnt_z_bottom_pmts,
                                                          xenonnt_z_cathode,
                                                          xenonnt_sensitive_volume_radius),
                                'electric_field': 200,
                                'create_S2': False,
                                'xe_density': 2.862,
                                'efield_outside_map': 200,
                                'to_be_stored': True,
                                },
               'GasPhase': {'volume_id': 3,
                            'roi': _make_roi_cylinder(xenonnt_z_lxe,
                                                      xenonnt_z_top_pmts,
                                                      xenonnt_sensitive_volume_radius),
                            'electric_field': 200,
                            'create_S2': False,
                            'xe_density': 0.0177,
                            'efield_outside_map': 200,
                            'to_be_stored': False,
                            },
               }
    return volumes, outer_cylinder


def xenon1t():
    """
    Default XENON1T TPC with different sensitive volumes.

    The structure for each volume is as follows:
        key: VolumeName
        items: keyword arguments for the Sensitive Volume class

    :return: A list of volumes for the default 1T detector.
    """
    # TODO: Can we do this differently? Ideas are welcome....
    # Outer edges of the volume of interest. This is needed for a
    # preselection of interactions inside the desired volumes before
    # clustering.
    outer_cylinder = {'max_z': xenon1t_z_top_pmts,
                      'min_z': xenon1t_z_bottom_pmts,
                      'max_r': xenon1t_sensitive_volume_radius
                      }

    volumes = {'TPC': {'volume_id': 1,
                       'roi': _make_roi_cylinder(xenon1t_z_cathode,
                                                 xenon1t_z_gate_mesh,
                                                 xenon1t_sensitive_volume_radius),
                       'electric_field': 82,
                       'create_S2': True,
                       'xe_density': 2.862,
                       'efield_outside_map': 82,
                       'to_be_stored': True,
                       },
               'BelowCathode': {'volume_id': 2,
                                'roi': _make_roi_cylinder(xenon1t_z_bottom_pmts,
                                                          xenon1t_z_cathode,
                                                          xenon1t_sensitive_volume_radius),
                                'electric_field': 82,
                                'create_S2': False,
                                'xe_density': 2.862,
                                'efield_outside_map': 82,
                                'to_be_stored': True,
                                },
               'GasPhase': {'volume_id': 3,
                            'roi': _make_roi_cylinder(xenon1t_z_lxe,
                                                      xenon1t_z_top_pmts,
                                                      xenon1t_sensitive_volume_radius),
                            'electric_field': 82,
                            'create_S2': False,
                            'xe_density': 0.0177,
                            'efield_outside_map': 82,
                            'to_be_stored': False,
                            },
               }
    return volumes, outer_cylinder


def _make_roi_cylinder(z_min, z_max, r_max):
    """
    Function to generate different rois, this is needed since
    we would like to have numba.njitted functions later.
    """

    # see: https://numba.pydata.org/numba-doc/dev/user/faq.html
    @numba.njit()
    def roi(x, y, z):
        return epix.in_cylinder(x, y, z, z_min, z_max, r_max)

    return roi
