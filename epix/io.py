import numpy as np
import uproot
import awkward as ak
import pandas as pd

import os
import warnings
import wfsim
import configparser

from .common import awkward_to_flat_numpy, offset_range, reshape_awkward

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


class file_loader():
    """
    Class which contains functions to load geant4 interactions from
    a root file via uproot4 or interactions from a csv file via pandas.
    
    Besides loading, a simple data selection is performed. Units are
    already converted into strax conform values. mm -> cm and s -> ns.
    Args:
        directory (str): Directory in which the data is stored.
        file_name (str): File name
        arg_debug (bool): If true, print out loading information.
        outer_cylinder (dict): If specified will cut all events outside of the
            given cylinder.
        kwargs (dict): Keyword arguments passed to .arrays of
            uproot4.
        cut_by_eventid (bool): If true event start/stop are applied to
            eventids, instead of rows.
    Returns:
        awkward1.records: Interactions (eventids, parameters, types).
        integer: Number of events simulated.
    """

    def __init__(self,
                directory,
                file_name, 
                arg_debug=False,
                outer_cylinder=None,
                kwargs={},
                cut_by_eventid=False,
                cut_nr_only=False,
                ):

        self.directory = directory
        self.file_name = file_name
        self.arg_debug = arg_debug
        self.outer_cylinder = outer_cylinder
        self.kwargs = kwargs
        self.cut_by_eventid = cut_by_eventid
        self.cut_nr_only = cut_nr_only

        self.file = os.path.join(self.directory, self.file_name)

        self.column_names = ["x", "y", "z",
                             "t", "ed",
                             "type", "trackid",
                             "parenttype", "parentid",
                             "creaproc", "edproc"]

        #Prepare cut for root and csv case
        if self.outer_cylinder:
            self.cut_string = (f'(r < {self.outer_cylinder["max_r"]})'
                               f' & ((zp >= {self.outer_cylinder["min_z"] * 10}) & (zp < {self.outer_cylinder["max_z"] * 10}))')            
        else:
            self.cut_string = None

    def load_file(self):
        """ 
        Function which reads a root or csv file and removes 
        interactions and events without energy depositions. 

        Returns:
            interactions: awkward array
            n_simulated_events: Total number of simulated events in the 
                                opened file. (This includes events removed by cuts)
        """

        if self.file.endswith(".root"):
            interactions, n_simulated_events, start, stop = self._load_root_file()
        elif self.file.endswith(".csv"):
            interactions, n_simulated_events, start, stop = self._load_csv_file()
        else:
            raise ValueError(f'Cannot load events from file "{self.file}": .root or .cvs file needed.')

        if np.any(interactions['ed'] < 0):
            warnings.warn('At least one of the energy deposits is negative!')

        # Removing all events with zero energy deposit
        m = interactions['ed'] > 0
        if self.cut_by_eventid:
            # ufunc does not work here...
            m2 = (interactions['evtid'] >= start) & (interactions['evtid'] < stop)
            m = m & m2
        interactions = interactions[m]

        if self.cut_nr_only:
            m = ((interactions['type'] == "neutron")&(interactions['edproc'] == "hadElastic")) | (interactions['edproc'] == "ionIoni")
            e_dep_er = ak.sum(interactions[~m]['ed'], axis=1)
            e_dep_nr = ak.sum(interactions[m]['ed'], axis=1)
            interactions = interactions[(e_dep_er<10) & (e_dep_nr>0)]

        # Removing all events with no interactions:
        m = ak.num(interactions['ed']) > 0
        interactions = interactions[m]

        return interactions, n_simulated_events

    def _load_root_file(self):
        """
        Function which reads a root file using uproot,
        performs a simple cut and builds an awkward array.

        Returns:
            interactions: awkward array
            n_simulated_events: Total number of simulated events
            start: Index of the first loaded interaction
            stop: Index of the last loaded interaction
        """

        ttree, n_simulated_events = self._get_ttree()

        if self.arg_debug:
            print(f'Total entries in input file = {ttree.num_entries}')
            cutby_string='output file entry'
            if self.cut_by_eventid:
                cutby_string='g4 eventid'

            if self.kwargs['entry_start'] is not None:
                print(f'Starting to read from {cutby_string} {self.kwargs["entry_start"]}')
            if self.kwargs['entry_stop'] is not None:
                print(f'Ending read in at {cutby_string} {self.kwargs["entry_stop"]}')

        # If user specified entry start/stop we have to update number of
        # events for source rate computation:
        if self.kwargs['entry_start'] is not None:
            start = self.kwargs['entry_start']
        else:
            start = 0

        if self.kwargs['entry_stop'] is not None:
            stop = self.kwargs['entry_stop']
        else:
            stop = n_simulated_events
        n_simulated_events = stop - start

        if self.cut_by_eventid:
            # Start/stop refers to eventid so drop start drop from kwargs
            # dict if specified, otherwise we cut again on rows.
            self.kwargs.pop('entry_start', None)
            self.kwargs.pop('entry_stop', None)

        # Conversions and parameters to be computed:
        alias = {'x': 'xp/10',  # converting "geant4" mm to "straxen" cm
                 'y': 'yp/10',
                 'z': 'zp/10',
                 'r': 'sqrt(x**2 + y**2)',
                 't': 'time*10**9'
                }

        # Read in data, convert mm to cm and perform a first cut if specified:
        interactions = ttree.arrays(self.column_names,
                                    self.cut_string,
                                    aliases=alias,
                                    **self.kwargs)
        eventids = ttree.arrays('eventid', **self.kwargs)
        eventids = ak.broadcast_arrays(eventids['eventid'], interactions['x'])[0]
        interactions['evtid'] = eventids

        xyz_pri = ttree.arrays(['x_pri', 'y_pri', 'z_pri'],
                              aliases={'x_pri': 'xp_pri/10',
                                       'y_pri': 'yp_pri/10',
                                       'z_pri': 'zp_pri/10'
                                      },
                              **self.kwargs)

        interactions['x_pri'] = ak.broadcast_arrays(xyz_pri['x_pri'], interactions['x'])[0]
        interactions['y_pri'] = ak.broadcast_arrays(xyz_pri['y_pri'], interactions['x'])[0]
        interactions['z_pri'] = ak.broadcast_arrays(xyz_pri['z_pri'], interactions['x'])[0]

        return interactions, n_simulated_events, start, stop

    def _load_csv_file(self):
        """ 
        Function which reads a csv file using pandas, 
        performs a simple cut and builds an awkward array.

        Returns:
            interactions: awkward array
            n_simulated_events: Total number of simulated events
            start: Index of the first loaded interaction
            stop: Index of the last loaded interaction
        """

        print("Load instructions from a csv file!")
        
        instr_df =  pd.read_csv(self.file)

        #unit conversion similar to root case
        instr_df["x"] = instr_df["xp"]/10 
        instr_df["y"] = instr_df["yp"]/10 
        instr_df["z"] = instr_df["zp"]/10
        instr_df["x_pri"] = instr_df["xp_pri"]/10
        instr_df["y_pri"] = instr_df["yp_pri"]/10
        instr_df["z_pri"] = instr_df["zp_pri"]/10
        instr_df["r"] = np.sqrt(instr_df["x"]**2 + instr_df["y"]**2)
        instr_df["t"] = instr_df["time"]*10**9

        #Check if all needed columns are in place:
        if not set(self.column_names).issubset(instr_df.columns):
            warnings.warn("Not all needed columns provided!")

        n_simulated_events = len(np.unique(instr_df.eventid))

        if self.outer_cylinder:
            instr_df = instr_df.query(self.cut_string)

        interactions = self._awkwardify_df(instr_df)

        #Use always all events in the csv file
        start = 0
        stop = n_simulated_events

        return interactions, n_simulated_events, start, stop 

    def _get_ttree(self):
        """
        Function which searches for the correct ttree in MC root file.

        :param directory: Directory where file is
        :param file_name: Name of the file
        :return: root ttree and number of simulated events
        """
        root_dir = uproot.open(self.file)

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

    def _awkwardify_df(self, df):
        """
        Function which builds an jagged awkward array from pandas dataframe.

        Args:
            df: Pandas Dataframe

        Returns:
            ak.Array(dictionary): awkward array

        """

        _, evt_offsets = np.unique(df["eventid"], return_counts = True)
    
        dictionary = {"x": reshape_awkward(df["x"].values , evt_offsets),
                      "y": reshape_awkward(df["y"].values , evt_offsets),
                      "z": reshape_awkward(df["z"].values , evt_offsets),
                      "x_pri": reshape_awkward(df["x_pri"].values, evt_offsets),
                      "y_pri": reshape_awkward(df["y_pri"].values, evt_offsets),
                      "z_pri": reshape_awkward(df["z_pri"].values, evt_offsets),
                      "r": reshape_awkward(df["r"].values , evt_offsets),
                      "t": reshape_awkward(df["t"].values , evt_offsets),
                      "ed": reshape_awkward(df["ed"].values , evt_offsets),
                      "type":reshape_awkward(np.array(df["type"], dtype=str) , evt_offsets),
                      "trackid": reshape_awkward(df["trackid"].values , evt_offsets),
                      "parenttype": reshape_awkward(np.array(df["parenttype"], dtype=str) , evt_offsets),
                      "parentid": reshape_awkward(df["parentid"].values , evt_offsets),
                      "creaproc": reshape_awkward(np.array(df["creaproc"], dtype=str) , evt_offsets),
                      "edproc": reshape_awkward(np.array(df["edproc"], dtype=str) , evt_offsets),
                      "evtid": reshape_awkward(df["eventid"].values , evt_offsets),
                    }

        return ak.Array(dictionary)

# ----------------------
# Outputing wfsim instructions:
# ----------------------
def awkward_to_wfsim_row_style(interactions):
    """
    Converts awkward array instructions into instructions required by
    WFSim.

    :param interactions: awkward.Array containing GEANT4 simulation
        information.
    :return: Structured numpy.array. Each row represents either a S1 or
        S2
    """
    if len(interactions) == 0:
        return np.empty(0, dtype=wfsim.instruction_dtype)

    ninteractions = np.sum(ak.num(interactions['ed']))
    res = np.zeros(2 * ninteractions, dtype=wfsim.instruction_dtype)

    # TODO: Currently not supported rows with only electrons or photons due to
    # this super odd shape
    for i in range(2):
        res['event_number'][i::2] = offset_range(ak.to_numpy(ak.num(interactions['evtid'])))
        res['type'][i::2] = i + 1
        res['x'][i::2] = awkward_to_flat_numpy(interactions['x'])
        res['y'][i::2] = awkward_to_flat_numpy(interactions['y'])
        res['z'][i::2] = awkward_to_flat_numpy(interactions['z'])
        res['x_pri'][i::2] = awkward_to_flat_numpy(interactions['x_pri'])
        res['y_pri'][i::2] = awkward_to_flat_numpy(interactions['y_pri'])
        res['z_pri'][i::2] = awkward_to_flat_numpy(interactions['z_pri'])
        res['time'][i::2] = awkward_to_flat_numpy(interactions['t'])
        res['g4id'][i::2] = awkward_to_flat_numpy(interactions['evtid'])
        res['vol_id'][i::2] = awkward_to_flat_numpy(interactions['vol_id'])
        res['e_dep'][i::2] = awkward_to_flat_numpy(interactions['ed'])
        if 'local_field' in res.dtype.names:
            res['local_field'][i::2] = awkward_to_flat_numpy(interactions['e_field'])

        recoil = awkward_to_flat_numpy(interactions['nestid'])
        res['recoil'][i::2] = np.where(np.isin(recoil, [0, 6, 7, 8, 11]), recoil, 8)

        if i:
            res['amp'][i::2] = awkward_to_flat_numpy(interactions['electrons'])
        else:
            res['amp'][i::2] = awkward_to_flat_numpy(interactions['photons'])
            if 'n_excitons' in res.dtype.names:
                res['n_excitons'][i::2] = awkward_to_flat_numpy(interactions['excitons'])
    # Remove entries with no quanta
    res = res[res['amp'] > 0]
    return res
