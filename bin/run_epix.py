import logging  #TODO: Logging is missing

import os
import argparse
import time

import awkward1 as ak
import numpy as np
import pandas as pd

# TODO: Fix epix import, making system epix aware
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import epix
from .my_volumes import my_volumes


parser = argparse.ArgumentParser(description="Electron and Photon Instructions generator for XENON wfsim")
parser.add_argument('--InputFile', dest='InputFile',
                    action='store', required=True,
                    help='Input Geant4 ROOT file')
parser.add_argument('--EntryStop', dest='EntryStop', type=int,
                    action='store',
                    help='Number of entries to read from first. Defaulted to all')
parser.add_argument('--MicroSeparation', dest='MicroSeparation', type=float,
                    action='store', default=0.05,
                    help='Spatial resolution for DBSCAN micro-clustering [mm]')
parser.add_argument('--MicroSeparationTime', dest='MicroSeparationTime', type=float,
                    action='store', default=10,
                    help='Time resolution for DBSCAN micro-clustering [ns]')
parser.add_argument('--TagClusterBy', dest='TagClusterBy', type=str,
                    action='store', default='time',
                    help=('Classification of the type of particle of a cluster, '
                          'based on most energetic contributor ("energy") or first '
                          'depositing particle ("time")'),
                    choices={'time', 'energy'})
parser.add_argument('--Efield', dest='Efield',
                    action='store', default=200,
                    help=('Drift field map as text file ("r z E", with '
                          'length in cm and field in V/cm) or as a constant (in V/cm; '
                          'recommended only for testing)'))
parser.add_argument('--MaxDelay', dest='MaxDelay', type=float,
                    action='store', default=1e7, #ns
                    help='Maximal time delay to first interaction which will be stored [ns]')
parser.add_argument('--Timing', dest='Timing', type=bool,
                    action='store', default=False,
                    help='If true will print out the time needed.')
parser.add_argument('--OutputPath', dest='OutputPath',
                   action='store', default="",
                   help=('Optional output path. If not specified the result will be saved'
                         'in the same dir as the input file.'))

args = parser.parse_args(sys.argv[1:])


def main(args):
    if is_number(args.Efield):
        args.Efield = float(args.Efield)

    print("epix configuration: ", args)
    tnow = 0
    starttime = 0
    if args.Timing:
        # TODO: also add memory information see starxer and change this to debug
        # Getting time information:
        starttime = time.time()
        tnow = starttime

    # Getting file path and split it into directory and file name:
    p = args.InputFile
    p = p.split('/')
    if p[0] == "":
        p[0] = "/"
    path = os.path.join(*p[:-1])
    file_name = p[-1]
    print(f'Reading in root file {file_name}')

    # Loading data:
    inter = epix.loader(path, file_name, kwargs_uproot_ararys={'entry_stop': args.EntryStop})
    if args.Timing:
        tnow = monitor_time(tnow, 'load data.')

    # Cluster finding and clustering:
    print((f'Finding clusters of interactions with a dr = {args.MicroSeparation} cm'
           f' and dt = {args.MicroSeparationTime} ns'))
    inter = epix.find_cluster(inter, args.MicroSeparation, args.MicroSeparationTime)

    if args.Timing:
        tnow = monitor_time(tnow, 'cluster finding.')

    result = epix.cluster(inter, args.TagClusterBy == 'energy')
    if args.Timing:
        tnow = monitor_time(tnow, 'cluster merging.')

    # Add eventid again:
    result['evtid'] = ak.broadcast_arrays(inter['evtid'][:, 0], result['ed'])[0]
    
    # Sort detector volumes and keep interactions in selected ones:
    print('Removing clusters not in volumes:', *[x.name for x in my_volumes])
    print(f'Number of clusters before: {np.sum(ak.num(result["ed"]))}')

    res_det = epix.in_sensitive_volume(result, my_volumes)
    # Adding new fields to result:
    for field in res_det.fields:
        result[field] = res_det[field]
    m = result['ids'] > 0  # All volumes have an idea larger zero
    result = result[m]

    # Removing now empty events as a result of the selection above:
    m = ak.num(result['ed']) > 0
    result = result[m]

    print(f'Number of clusters after: {np.sum(ak.num(result["ed"]))}')
    print('Assigning electric field to clusters')

    if is_number(args.Efield):
        # TODO: Efield handling is super odd, we should move this to detector volumes
        #  We can make this argeument optional and keep definition of my_volumes in case
        #  no new field is specified.
        efields = np.ones(np.sum(ak.num(result)), np.float32)*args.Efield
    else:
        e_field_handler = epix.MyElectricFieldHandler(args.Efield)
        efields = e_field_handler.get_field(epix.awkward_to_flat_numpy(result.x),
                                            epix.awkward_to_flat_numpy(result.y),
                                            epix.awkward_to_flat_numpy(result.z),
                                            outside_map=200  # V/cm
                                            )

    result['e_field'] = epix.reshape_awkward(efields, ak.num(result))

    # Sort in time and set first cluster to t=0, then chop all delayed
    # events which are too far away from the rest.
    # (This is a requirement of WFSim)
    result = result[ak.argsort(result['t'])]
    result['t'] = result['t'] - result['t'][:, 0]
    result = result[result['t'] <= args.MaxDelay]
    # Secondly truly separate events by time (1.1 times the max time),
    # with first event starting at max time (needed for wfsim)
    dt = np.arange(1, len(result['t'])+1) + np.arange(1, len(result['t'])+1) / 10
    dt *= args.MaxDelay
    result['t'] = (result['t'][:, :] + result['t'][:, 0] + dt)

    print('Generating photons and electrons for events')
    # Generate quanta:
    photons, electrons = epix.quanta_from_NEST(epix.awkward_to_flat_numpy(result['ed']),
                                               epix.awkward_to_flat_numpy(result['nestid']),
                                               epix.awkward_to_flat_numpy(result['e_field']),
                                               epix.awkward_to_flat_numpy(result['A']),
                                               epix.awkward_to_flat_numpy(result['Z']),
                                               epix.awkward_to_flat_numpy(result['ids']),
                                               density=epix.awkward_to_flat_numpy(result['xe_density']))
    result['photons'] = epix.reshape_awkward(photons, ak.num(result['ed']))
    result['electrons'] = epix.reshape_awkward(electrons, ak.num(result['ed']))
    if args.Timing:
        _ = monitor_time(tnow, 'get quanta.')

    # Reshape instructions:
    instructions = epix.awkward_to_really_awkward(result)

    # Remove entries with no quanta, or
    # which were outside of the energy range supported by NEST
    instructions = instructions[instructions['amp'] > 0]
    ins_df = pd.DataFrame(instructions)
    
    if args.OutputPath:
        if not os.path.isdir(args.OutputPath):
            os.makedirs(args.OutputPath)
        if not args.OutputPath.endswith("/"):
            args.OutputPath += "/"
        output_path_and_name = args.OutputPath + file_name[:-5] + "_wfim_instructions.csv"
    else:
        output_path_and_name = args.InputFile[:-5] + "_wfim_instructions.csv"
    ins_df.to_csv(output_path_and_name, index=False)
    
    print('Done')
    print('Instructions saved to ', output_path_and_name)
    if args.Timing:
        _ = monitor_time(starttime, 'run epix.')


def is_number(x):
    try:
        float(x)
        return True
    except ValueError:
        return False


def monitor_time(prev_time, task):
    t = time.time()
    print(f'It took {(t - prev_time):.4f} sec to {task}')
    return t


if __name__ == "__main__":
    main(args)
