#!/usr/bin/env python3

import logging
import sys
import os
import argparse
import time
import awkward1 as ak
import numpy as np
import pandas as pd
import epix

def isNumber(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

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
parser.add_argument('--EventRate', dest='EventRate',
                    action='store', default=-1, 
                    help='Event rate for event separation. Use -1 for clean simulations'
                         'or give a rate >0 to space events randomly.'
                         'or give a csv file containing time[s] and rate[Hz]')
parser.add_argument('--Timing', dest='Timing', type=bool,
                    action='store', default=False,
                    help='If true will print out the time needed.')
parser.add_argument('--OutputPath', dest='OutputPath',
                   action='store', default="",
                   help=('Optional output path. If not specified the result will be saved'
                        'in the same dir as the input file.'))

args = parser.parse_args(sys.argv[1:])

if isNumber(args.Efield):
    args.Efield = float(args.Efield)

print("epix configuration: ", args)

def main(args):
    if args.Timing:
        starttime = time.time()
        tnow = starttime
    # Loading the data:
    p = args.InputFile
    p = p.split('/')
    if p[0] == "":
        p[0] = "/"
    path = os.path.join(*p[:-1])
    file_name = p[-1]
    print(f'Reading in root file {file_name}')
    inter = epix.loader(path, file_name, kwargs_uproot_ararys={'entry_stop': args.EntryStop})
    if args.Timing:
        t = time.time()
        print(f'It took {round(t - tnow, 5)} sec to load the data.')
        tnow = t

    print((f'Finding clusters of interactions with a dr = {args.MicroSeparation} mm'
           f' and dt = {args.MicroSeparationTime} ns'))
    inter = epix.find_cluster(inter, args.MicroSeparation/10, args.MicroSeparationTime)
    if args.Timing:
        t = time.time()
        print(f'It took {round(t - tnow, 5)} sec to find clusters.')
        tnow = t
        
    result = epix.cluster(inter, args.TagClusterBy=='energy')
    if args.Timing:
        t = time.time()
        print(f'It took {round(t - tnow, 5)} sec to cluster events.')
        tnow = t

    # Add eventid again:
    result['evtid'] = ak.broadcast_arrays(inter['evtid'][:, 0], result['ed'])[0]
    
    # Sort detector volumes and keep interactions in selected ones:
    sensitive_volumes = [epix.tpc, epix.below_cathode] #TODO: add options
    print('Removing clusters not in volumes:', *[x.name for x in sensitive_volumes])
    print(f'Number of clusters before: {np.sum(ak.num(result["x"]))}')
    result['ids'] = epix.in_sensitive_volume(result, sensitive_volumes)
    m = (result['ids'] == sensitive_volumes[0].volume_id) | (result['ids'] == sensitive_volumes[1].volume_id)
    # TODO: The idea is to code this properly and depending on len(sensitive_volumes)
    result = result[m]
    # Removing now empty events as a result of the selection above:
    m = ak.num(result['ed']) > 0
    result = result[m]
    print(f'Number of clusters after: {np.sum(ak.num(result["x"]))}')

    print('Assigning electric field to clusters')
    if isNumber(args.Efield):
        efields = np.ones(np.sum(ak.num(result)), np.float32)*args.Efield
    else:
        E_field_handler = epix.MyElectricFieldHandler(args.Efield)
        efields = E_field_handler.get_field(epix.awkward_to_flat_numpy(result.x),
                                            epix.awkward_to_flat_numpy(result.y),
                                            epix.awkward_to_flat_numpy(result.z))
        # TODO: Move this into GetField:
        efields[efields == np.nan] = 200
    result['e_field'] = epix.reshape_awkward(efields, ak.num(result))

    # Sort in time and set first cluster to t=0, then chop all delayed
    # events which are too far away from the rest.
    # (This is a requirement of WFSim)
    result = result[ak.argsort(result['t'])]
    result['t'] = result['t'] - result['t'][:, 0]
    result = result[result['t'] <= args.MaxDelay]

    #Separate event in time 
    number_of_events = len(result["t"])
    if isNumber(args.EventRate):
        if args.EventRate == -1:
            dt = epix.clean_separation(number_of_events, args.MaxDelay)
            print("Clean Event Separation")
        else:
            dt = epix.times_from_fixed_rate(args.EventRate, number_of_events)
            print("Fixed Event Rate")
    else:
        Rate_df = pd.read_csv(args.EventRate)
        dt = epix.times_from_variable_rate(Rate_df.Rate.values, Rate_df.Time.values, number_of_events)
        print("Variable Event Rate")
    result['t'] = result['t'][:, :] + dt


    print('Generating photons and electrons for events')
    # Generate quanta:
    # TODO: May crash for to large energy deposits?
    # TODO: Support different volumes...
    photons, electrons = epix.quanta_from_NEST(epix.awkward_to_flat_numpy(result['ed']),
                                               epix.awkward_to_flat_numpy(result['nestid']),
                                               epix.awkward_to_flat_numpy(result['e_field']),
                                               epix.awkward_to_flat_numpy(result['A']),
                                               epix.awkward_to_flat_numpy(result['Z']),
                                               epix.awkward_to_flat_numpy(result['ids']))
    result['photons'] = epix.reshape_awkward(photons, ak.num(result['ed']))
    result['electrons'] = epix.reshape_awkward(electrons, ak.num(result['ed']))
    if args.Timing:
        t = time.time()
        print(f'It took {round(t - tnow, 5)} sec to get quanta.')
        tnow = t

    # Reshape instructions:
    #TODO: Change me....
    instructions = epix.awkward_to_really_awkward(result)
    # Remove entries with no quanta
    instructions = instructions[instructions['amp']>0]
    ins_df = pd.DataFrame(instructions)
    
    if args.OutputPath:
        os.makedirs(args.OutputPath, exist_ok=True)
        if not args.OutputPath.endswith("/"):
            args.OutputPath+="/"
        output_path_and_name=args.OutputPath + file_name[:-5] + "_wfsim_instructions.csv"
    else:
        output_path_and_name=args.InputFile[:-5] + "_wfsim_instructions.csv"
    ins_df.to_csv(output_path_and_name, index=False)
    
    print('Done')
    print('Instructions saved to ', output_path_and_name)
    if args.Timing:
        t = time.time()
        print(f'It took {round(t - starttime, 5)} sec to process file.')

if __name__ == "__main__":
    main(args)
