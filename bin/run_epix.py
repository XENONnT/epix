import logging
import sys
import os
sys.path.append(os.path.abspath(os.path.join('..', 'epix')))
import argparse
import time

import awkward1 as ak

#TODO: change the importing...
from epix.ElectricFieldHandler import MyElectricFieldHandler  #TODO call me in SensetiveVolume
from epix.clustering import *
from epix.common import *
from epix.detector_volumes import *
from epix.io import *
from epix.quanta_generation import *

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
    inter = loader(path, file_name, kwargs_uproot_ararys={'entry_stop': args.EntryStop})
    if args.Timing:
        t = time.time()
        print(f'It took {round(t - tnow, 5)} sec to load the data.')
        tnow = t

    print((f'Finding clusters of interactions with a dr = {args.MicroSeparation} mm'
           f' and dt = {args.MicroSeparationTime} ns'))
    inter = find_cluster(inter, args.MicroSeparation/10, args.MicroSeparationTime)
    if args.Timing:
        t = time.time()
        print(f'It took {round(t - tnow, 5)} sec to find clusters.')
        tnow = t
        
    result = cluster(inter, args.TagClusterBy=='energy')
    if args.Timing:
        t = time.time()
        print(f'It took {round(t - tnow, 5)} sec to cluster events.')
        tnow = t

    # Add eventid again:
    result['evtid'] = ak.broadcast_arrays(inter['evtid'][:, 0], result['ed'])[0]
    
    # Sort detector volumes and keep interactions in selected ones:
    sensitive_volumes = [tpc, below_cathode] #TODO: add options
    print('Removing clusters not in volumes:', *[x.name for x in sensitive_volumes])
    print(f'Number of clusters before: {np.sum(ak.num(result["x"]))}')
    result['ids'] = in_sensitive_volume(result, sensitive_volumes)
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
        E_field_handler = MyElectricFieldHandler(args.Efield)
        efields = E_field_handler.get_field(awkward_to_flat_numpy(result.x),
                                            awkward_to_flat_numpy(result.y),
                                            awkward_to_flat_numpy(result.z))
        # TODO: Move this into GetField:
        efields[efields == np.nan] = 200
    result['e_field'] = reshape_awkward(efields, ak.num(result))

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
    # TODO: May crash for to large energy deposits?
    # TODO: Support different volumes...
    photons, electrons = quanta_from_NEST(awkward_to_flat_numpy(result['ed']),
                                          awkward_to_flat_numpy(result['nestid']),
                                          awkward_to_flat_numpy(result['e_field']),
                                          awkward_to_flat_numpy(result['A']),
                                          awkward_to_flat_numpy(result['Z']),
                                          awkward_to_flat_numpy(result['ids']))
    result['photons'] = reshape_awkward(photons, ak.num(result['ed']))
    result['electrons'] = reshape_awkward(electrons, ak.num(result['ed']))
    if args.Timing:
        t = time.time()
        print(f'It took {round(t - tnow, 5)} sec to get quanta.')
        tnow = t

    # Reshape instructions:
    #TODO: Change me....
    instructions = awkward_to_really_awkward(result)
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
