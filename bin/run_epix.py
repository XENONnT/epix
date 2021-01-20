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

def pars_args():
    parser = argparse.ArgumentParser(description="Electron and Photon Instructions generator for XENON wfsim")
    parser.add_argument('--InputFile', dest='InputFile',
                        action='store', required=True,
                        help='Input Geant4 ROOT file')
    parser.add_argument('--Detector', dest='detector', type=str,
                        action='store', default='xenonnt_detector',
                        help='Detector which should be used. Has to be defined in epix.detectors.')
    parser.add_argument('--Config', dest='config', type=str,
                        action='store', default='',
                        help='Config file to overwrite default detector settings.')
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
    parser.add_argument('--MaxDelay', dest='MaxDelay', type=float,
                        action='store', default=1e7, #ns
                        help='Maximal time delay to first interaction which will be stored [ns]')
    parser.add_argument('--OutputPath', dest='OutputPath',
                       action='store', default="",
                       help=('Optional output path. If not specified the result will be saved'
                             'in the same dir as the input file.'))
    parser.add_argument('--Debug', dest='debug',
                       action='store_true', default="",
                       help=('If specifed additional information is printed to the consol.')
                        )

    args = parser.parse_args(sys.argv[1:])
    return args


def main(return_df=True):
    #TODO: remove setup from main for strax
    args, path, file, detector, outer_cylinder = setup()

    if args.debug:
        print("epix configuration: ", args)
        # TODO: also add memory information see starxer and change this to debug
        # Getting time information:
        starttime = time.time()
        tnow = starttime

    # Loading data:
    inter = epix.loader(path, file,
                        outer_cylinder=outer_cylinder,
                        kwargs_uproot_ararys={'entry_stop': args.EntryStop}
                        )

    if args.debug:
        tnow = monitor_time(tnow, 'load data.')
        print((f'Finding clusters of interactions with a dr = {args.MicroSeparation} cm'
               f' and dt = {args.MicroSeparationTime} ns'))

    # Cluster finding and clustering:
    inter = epix.find_cluster(inter, args.MicroSeparation, args.MicroSeparationTime)

    if args.debug:
        tnow = monitor_time(tnow, 'cluster finding.')

    result = epix.cluster(inter, args.TagClusterBy == 'energy')

    if args.debug:
        tnow = monitor_time(tnow, 'cluster merging.')

    # Add eventid again:
    result['evtid'] = ak.broadcast_arrays(inter['evtid'][:, 0], result['ed'])[0]
    
    # Sort detector volumes and keep interactions in selected ones:
    if args.debug:
        print('Removing clusters not in volumes:', *[v.name for v in detector])
        print(f'Number of clusters before: {np.sum(ak.num(result["ed"]))}')

    # Returns all interactions which are inside in one of the volumes,
    # Checks for volume overlap, assigns xe density and create S2 to
    # interactions. EField comes later since interpolated maps cannot be
    # called inside numba functions.
    res_det = epix.in_sensitive_volume(result, detector)
    # Adding new fields to result:
    for field in res_det.fields:
        result[field] = res_det[field]
    m = result['vol_ids'] > 0  # All volumes have an id larger zero
    result = result[m]

    # Removing now empty events as a result of the selection above:
    m = ak.num(result['ed']) > 0
    result = result[m]

    if args.debug:
        print(f'Number of clusters after: {np.sum(ak.num(result["ed"]))}')
        print('Assigning electric field to clusters')

    # Add electric field to array:
    efields = np.ones(np.sum(ak.num(result)), np.float32)
    result['e_field'] = epix.reshape_awkward(efields, ak.num(result))

    # Loop over volume and assign values:
    for volume in detector:
        if isinstance(volume.electric_field, float):
            m = result['vol_ids'] == volume.volume_id
            result['e_field'][m] = volume.electric_field
        #else:
        #    e_field_handler = epix.MyElectricFieldHandler(args.Efield)
        #    efields = e_field_handler.get_field(epix.awkward_to_flat_numpy(result.x),
        #                                        epix.awkward_to_flat_numpy(result.y),
        #                                        epix.awkward_to_flat_numpy(result.z),
        #                                        outside_map=200  # V/cm
        #                                        )



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

    if args.debug:
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
    if args.debug:
        _ = monitor_time(tnow, 'get quanta.')

    # Reshape instructions:
    instructions = epix.awkward_to_really_awkward(result)

    # Remove entries with no quanta
    instructions = instructions[instructions['amp'] > 0]
    ins_df = pd.DataFrame(instructions)

    if return_df:
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


def setup():
    """
    Function which sets-up configurations. (for strax conversion)

    :return:
    """
    args = pars_args()
    # Getting file path and split it into directory and file name:
    p = args.InputFile
    p = p.split('/')
    if p[0] == "":
        p[0] = "/"
    path = os.path.join(*p[:-1])
    file_name = p[-1]

    if args.debug:
        print(f'Reading in root file {file_name}')

    # Init detector volume according to settings and get outer_cylinder
    # for data reduction of non-relevant interactions.
    detector = epix.init_detector(args.detector, args.config)
    outer_cylinder = getattr(epix.detectors, args.detector)
    _, outer_cylinder = outer_cylinder()
    return args, path, file_name, detector, outer_cylinder

if __name__ == "__main__":
    main()

