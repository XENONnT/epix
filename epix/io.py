import numpy as np
import uproot
import awkward as ak

import os
import warnings
import configparser

from .common import awkward_to_flat_numpy, offset_range

SUPPORTED_OPTION = {'to_be_stored': 'getboolean',
                    'electric_field': ('getfloat', 'get'),
                    'create_S2': 'getboolean',
                    'xe_density': 'getfloat',
                    'electirc_field_outside_map': 'getfloat',
                    }


def load_config(config_file_path):
    """
    Loads config file and returns dictionary.

    :param config_file_path:
    :return: dict
    """
    config = configparser.ConfigParser()
    config.read(config_file_path)
    sections = config.sections()

    settings = {}
    for s in sections:
        options = {}
        c = config[s]
        for key in c.keys():
            if key not in SUPPORTED_OPTION:
                warnings.warn(f'Option "{key}" of section {s} is not supported'
                              ' and will be ignored.')
                continue
            # Get correct get method to convert string input:
            if key == 'electric_field':
                # Electric field is a bit more complicated can be
                # either a float or string:
                try:
                    getter = getattr(c, SUPPORTED_OPTION[key][0])
                    options[key] = getter(key)
                except ValueError:
                    getter = getattr(c, SUPPORTED_OPTION[key][1])
                    options[key] = getter(key)
            else:
                try:
                    getter = getattr(c, SUPPORTED_OPTION[key])
                    options[key] = getter(key)
                except Exception as e:
                    raise ValueError(f'Cannot load "{key}" from section "{s}" in config file.') from e

        settings[s] = options
    return settings


def loader(directory, file_name, arg_debug=False, outer_cylinder=None, kwargs_uproot_arrays={}):
    """
    Function which loads geant4 interactions from a root file via
    uproot4.

    Beside loading the a simple data selection is performed. Units are
    already converted into strax conform values. mm -> cm and s -> ns.

    Args:
        directory (str): Directory in which the data is stored.
        file (str): File name

    Kwargs:
        arg_debug: If true, print out loading information.
        outer_cylinder: If specified will cut all events outside of the
            given cylinder.
        kwargs_uproot_arrays: Keyword arguments passed to .arrays of
            uproot4.

    Returns:
        awkward1.records: Interactions (eventids, parameters, types).

    Note:
        We process eventids and the rest of the data in two different
        arrays due to different array structures. Also the type strings
        are split off since they suck. All arrays are finally merged.
    """
    root_dir = uproot.open(os.path.join(directory, file_name))
    
    if root_dir.classname_of('events') == 'TTree':
        ttree = root_dir['events']
    elif root_dir.classname_of('events/events') == 'TTree':
        ttree = root_dir['events/events']
    else:
        ttrees = []
        for k, v in root_dir.classnames().items():
            if v == 'TTree':
                ttrees.append(k)
        raise ValueError(f'Cannot find ttree object of "{file_name}".' 
                         'I tried to search in events and events/events.' 
                         f'Found a ttree in {ttrees}?')
    if arg_debug:
        print(f'Total entries in input file = {ttree.num_entries}')
        if kwargs_uproot_arrays['entry_stop']!=None:
            print(f'... from which {kwargs_uproot_arrays["entry_stop"]} are read')

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

    if outer_cylinder:
        cut_string = (f'(r < {outer_cylinder["max_r"]})'
                      f' & ((zp >= {outer_cylinder["min_z"] * 10}) & (zp < {outer_cylinder["max_z"] * 10}))')
    else:
        cut_string = None

    # Radin in data convert mm to cm and perform a first cut if specified:
    interactions = ttree.arrays(column_names,
                                cut_string,
                                aliases=alias,
                                **kwargs_uproot_arrays)
    eventids = ttree.arrays('eventid', **kwargs_uproot_arrays)
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
int_dtype = [(('Waveform simulator event number.', 'event_number'), np.int32),
             (('Quanta type (S1 photons or S2 electrons)', 'type'), np.int8),
             (('Time of the interaction [ns]', 'time'), np.int64),
             (('X position of the cluster[cm]', 'x'), np.float32),
             (('Y position of the cluster[cm]', 'y'), np.float32),
             (('Z position of the cluster[cm]', 'z'), np.float32),
             (('Number of quanta', 'amp'), np.int32),
             (('Recoil type of interaction.', 'recoil'), '<U5'),
             (('Energy deposit of interaction', 'e_dep'), np.float32),
             (('Eventid like in geant4 output rootfile', 'g4id'), np.int32)
             ]


def awkward_to_wfsim_row_style(interactions):
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
