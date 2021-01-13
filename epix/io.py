import numpy as np
import uproot4
import awkward1 as ak

import os
import warnings
import wfsim

from .common import awkward_to_flat_numpy, sensitive_volume_radius, z_top_pmts, z_bottom_pmts, offset_range


def loader(directory, file_name, cut_outside_tpc=True, kwargs_uproot_ararys={}):
    """
    Function which loads geant4 interactions from a root file via
    uproot4.

    Beside loading the a simple data selection is performed. Units are
    already converted into strax conform values. mm -> cm and s -> ns.

    Args:
        directory (str): Directory in which the data is stored.
        file (str): File name

    Kwargs:
        cut_outside_tpc (bool): If true only interactions inside the TPC
            are loaded. False all interactions in any sensetive volume
            are loaded.
        kwargs_uproot_ararys: Keyword arguments passed to .arrays of
            uproot4.

    Returns:
        awkward1.records: Interactions (eventids, parameters, types).

    Note:
        We process eventids and the rest of the data in two different
        arrays due to different array structures. Also the type strings
        are split off since they suck. All arrays are finally merged.
    """
    root_dir = uproot4.open(os.path.join(directory, file_name))
    ttree = root_dir['events']

    # Columns to be read from the root_file:
    column_names = ["x", "y", "z", "t", "ed",
                    "type", "trackid",
                    "parenttype", "parentid",
                    "creaproc", "edproc"]

    # Conversions and parameters to be computed:
    alias = {'x': 'xp/10',  # converting "geant4" mm to "straxen" cm
             'y': 'yp/10',
             'z': 'zp/10',
             'r': 'sqrt(x**2 + y**2)',
             't': 'time*10**9'
             }

    if cut_outside_tpc:
        cut_string = (f'(r < {sensitive_volume_radius})'
                      f' & ((zp >= {z_bottom_pmts * 10}) & (zp < {z_top_pmts * 10}))')
    else:
        cut_string = None

    # Radin in data convert mm to cm and perform a first cut if specified:
    interactions = ttree.arrays(column_names,
                                cut_string,
                                aliases=alias,
                                **kwargs_uproot_ararys)
    eventids = ttree.arrays('eventid', **kwargs_uproot_ararys)
    eventids = ak.broadcast_arrays(eventids['eventid'], interactions['x'])[0]
    interactions['evtid'] = eventids

    if np.any(interactions['ed'] < 0):
        warnings.warn('At least one of the energy deposits is negative!')

    # Removing all zero energy depsoits
    m = interactions['ed'] > 0
    interactions = interactions[m]

    # Removing all events with no interactions:
    m = ak.num(interactions['ed']) > 0
    interactions = interactions[m]

    return interactions


# ----------------------
# Outputing wfsim instructions:
# ----------------------
int_dtype = wfsim.instruction_dtype


def awkward_to_really_awkward(interactions):
    ninteractions = np.sum(ak.num(interactions['ed']))
    res = np.zeros(2 * ninteractions, dtype=int_dtype)
    res['recoil'] = 'er' #default

    # TODO: Currently not supported rows with only electrons or photons due to
    # this super odd shape
    for i in range(2):
        res['event_number'][i::2] = offset_range(ak.to_numpy(ak.num(interactions['evtid'])))
        res['type'][i::2] = i + 1
        res['x'][i::2] = awkward_to_flat_numpy(interactions['x'])
        res['y'][i::2] = awkward_to_flat_numpy(interactions['y'])
        res['z'][i::2] = awkward_to_flat_numpy(interactions['z'])
        res['time'][i::2] = awkward_to_flat_numpy(interactions['t'])
        res['g4id'][i::2] = awkward_to_flat_numpy(interactions['evtid'])
        res['e_dep'][i::2] = awkward_to_flat_numpy(interactions['ed'])
        if i:
            res['amp'][i::2] = awkward_to_flat_numpy(interactions['electrons'])
        else:
            res['amp'][i::2] = awkward_to_flat_numpy(interactions['photons'])

        res['recoil'][i::2][awkward_to_flat_numpy(interactions['nestid'] == 0)] = 'nr'
        res['recoil'][i::2][awkward_to_flat_numpy(interactions['nestid'] == 6)] = 'alpha'

    #TODO: Add a function which generates a new event if interactions are too far apart
    return res
