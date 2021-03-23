import numpy as np
import uproot
import awkward as ak

import os
import warnings
import wfsim
import configparser

from .common import awkward_to_flat_numpy, offset_range

SUPPORTED_OPTION = {'to_be_stored': 'getboolean',
                    'electric_field': ('getfloat', 'get'),
                    'create_S2': 'getboolean',
                    'xe_density': 'getfloat',
                    'efield_outside_map': 'getfloat',
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
    if not len(sections):
        raise ValueError(f'Cannot load sections from config file "{config_file_path}".' 
                         'Have you specified a wrong file?')
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


def loader(directory,
           file_name,
           arg_debug=False,
           outer_cylinder=None,
           kwargs_uproot_arrays={},
           cut_by_eventid=False,
           ):
    """
    Function which loads geant4 interactions from a root file via
    uproot4.

    Besides loading, a simple data selection is performed. Units are
    already converted into strax conform values. mm -> cm and s -> ns.

    Args:
        directory (str): Directory in which the data is stored.
        file_name (str): File name
        arg_debug (bool): If true, print out loading information.
        outer_cylinder (dict): If specified will cut all events outside of the
            given cylinder.
        kwargs_uproot_arrays (dict): Keyword arguments passed to .arrays of
            uproot4.
        cut_by_eventid (bool): If true event start/stop are applied to
            eventids, instead of rows.

    Returns:
        awkward1.records: Interactions (eventids, parameters, types).
        integer: Number of events simulated.
    """
    ttree, n_simulated_events = _get_ttree(directory, file_name)

    if arg_debug:
        print(f'Total entries in input file = {ttree.num_entries}')
        cutby_string='output file entry'
        if cut_by_eventid:
            cutby_string='g4 eventid'

        if kwargs_uproot_arrays['entry_start'] is not None:
            print(f'Starting to read from {cutby_string} {kwargs_uproot_arrays["entry_start"]}')
        if kwargs_uproot_arrays['entry_stop'] is not None:
            print(f'Ending read in at {cutby_string} {kwargs_uproot_arrays["entry_stop"]}')

    # If user specified entry start/stop we have to update number of
    # events for source rate computation:
    if kwargs_uproot_arrays['entry_start'] is not None:
        start = kwargs_uproot_arrays['entry_start']
    else:
        start = 0

    if kwargs_uproot_arrays['entry_stop'] is not None:
        stop = kwargs_uproot_arrays['entry_stop']
    else:
        stop = n_simulated_events
    n_simulated_events = stop - start

    if cut_by_eventid:
        # Start/stop refers to eventid so drop start drop from kwargs
        # dict if specified, otherwise we cut again on rows.
        kwargs_uproot_arrays.pop('entry_start', None)
        kwargs_uproot_arrays.pop('entry_stop', None)

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

    # Read in data, convert mm to cm and perform a first cut if specified:
    interactions = ttree.arrays(column_names,
                                cut_string,
                                aliases=alias,
                                **kwargs_uproot_arrays)
    eventids = ttree.arrays('eventid', **kwargs_uproot_arrays)
    eventids = ak.broadcast_arrays(eventids['eventid'], interactions['x'])[0]
    interactions['evtid'] = eventids

    if np.any(interactions['ed'] < 0):
        warnings.warn('At least one of the energy deposits is negative!')

    # Removing all events with zero energy deposit
    m = interactions['ed'] > 0
    if cut_by_eventid:
        # ufunc does not work here...
        m2 = (interactions['evtid'] >= start) & (interactions['evtid'] < stop)
        m = m & m2
    interactions = interactions[m]

    # Removing all events with no interactions:
    m = ak.num(interactions['ed']) > 0
    interactions = interactions[m]

    return interactions, n_simulated_events


def _get_ttree(directory, file_name):
    """
    Function which searches for the correct ttree in MC root file.

    :param directory: Directory where file is
    :param file_name: Name of the file
    :return: root ttree and number of simulated events
    """
    root_dir = uproot.open(os.path.join(directory, file_name))

    # Searching for TTree according to old/new MC file structure:
    if root_dir.classname_of('events') == 'TTree':
        ttree = root_dir['events']
        n_simulated_events = root_dir['nEVENTS'].members['fVal']
    elif root_dir.classname_of('events/events') == 'TTree':
        ttree = root_dir['events/events']
        n_simulated_events = root_dir['events/nbevents'].members['fVal']
    else:
        ttrees = []
        for k, v in root_dir.classnames().items():
            if v == 'TTree':
                ttrees.append(k)
        raise ValueError(f'Cannot find ttree object of "{file_name}".'
                         'I tried to search in events and events/events.'
                         f'Found a ttree in {ttrees}?')
    return ttree, n_simulated_events


# ----------------------
# Outputing wfsim instructions:
# ----------------------
int_dtype = wfsim.instruction_dtype


def awkward_to_wfsim_row_style(interactions):
    """
    Converts awkward array instructions into instructions required by
    WFSim.

    :param interactions: awkward.Array containing GEANT4 simulation
        information.
    :return: Structured numpy.array. Each row represents either a S1 or
        S2
    """
    ninteractions = np.sum(ak.num(interactions['ed']))
    res = np.zeros(2 * ninteractions, dtype=int_dtype)

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
        res['vol_id'][i::2] = awkward_to_flat_numpy(interactions['vol_id'])
        res['e_dep'][i::2] = awkward_to_flat_numpy(interactions['ed'])
        
        recoil = awkward_to_flat_numpy(interactions['nestid'])
        res['recoil'][i::2] = np.where(np.isin(recoil, [0,6,7,8,11]), recoil, 8)

        if i:
            res['amp'][i::2] = awkward_to_flat_numpy(interactions['electrons'])
        else:
            res['amp'][i::2] = awkward_to_flat_numpy(interactions['photons'])

    # Remove entries with no quanta
    res = res[res['amp'] > 0]
    return res
