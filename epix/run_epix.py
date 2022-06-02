import os
import time
import awkward as ak
import numpy as np
import pandas as pd
import warnings
import epix

from .common import ak_num, calc_dt, apply_time_offset


def main(args, return_df=False, return_wfsim_instructions=False, strax=False):
    """Call this function from the run_epix script"""

    if args['debug']:
        print("epix configuration: ", args)
        # TODO: also add memory information (see straxer) and change this to debug
        # Getting time information:
        starttime = time.time()
        tnow = starttime

    # Loading data:
    epix_file_loader = epix.file_loader(args['path'],
                                        args['file_name'],
                                        args['debug'],
                                        outer_cylinder=args['outer_cylinder'],
                                        kwargs={'entry_start': args['entry_start'],
                                                'entry_stop': args['entry_stop']},
                                        cut_by_eventid=args.get('cut_by_eventid', False),
                                        cut_nr_only=args.get('nr_only', False),
                                        )
    inter, n_simulated_events = epix_file_loader.load_file()

    if args['debug']:
        tnow = monitor_time(tnow, 'load data.')
        print(f"Finding clusters of interactions with a dr = {args['micro_separation']} mm"
               f" and dt = {args['micro_separation_time']} ns")

    # Cluster finding and clustering (convert micro_separation mm -> cm):
    inter = epix.find_cluster(inter, args['micro_separation']/10, args['micro_separation_time'])

    if args['debug']:
        tnow = monitor_time(tnow, 'find clusters.')

    result = epix.cluster(inter, args['tag_cluster_by'] == 'energy')

    if args['debug']:
        tnow = monitor_time(tnow, 'merge clusters.')

    # Add eventid again:
    result['evtid'] = ak.broadcast_arrays(inter['evtid'][:, 0], result['ed'])[0]

    # Add x_pri, y_pri, z_pri again:
    result['x_pri'] = ak.broadcast_arrays(inter['x_pri'][:, 0], result['ed'])[0]
    result['y_pri'] = ak.broadcast_arrays(inter['y_pri'][:, 0], result['ed'])[0]
    result['z_pri'] = ak.broadcast_arrays(inter['z_pri'][:, 0], result['ed'])[0]

    # Sort detector volumes and keep interactions in selected ones:
    if args['debug']:
        print('Removing clusters not in volumes:', *[v.name for v in args['detector_config']])
        print(f'Number of clusters before: {np.sum(ak_num(result["ed"]))}')

    # Returns all interactions which are inside in one of the volumes,
    # Checks for volume overlap, assigns Xe density and create_S2 to
    # interactions. EField comes later since interpolated maps cannot be
    # called inside numba functions.
    res_det = epix.in_sensitive_volume(result, args['detector_config'])

    # Adding new fields to result:
    for field in res_det.fields:
        result[field] = res_det[field]
    m = result['vol_id'] > 0  # All volumes have an id larger zero
    result = result[m]

    # Removing now empty events as a result of the selection above:
    m = ak_num(result['ed']) > 0
    result = result[m]

    if args['debug']:
        print(f'Number of clusters after: {np.sum(ak_num(result["ed"]))}')
        print('Assigning electric field to clusters')

    # Add electric field to array:
    efields = np.zeros(np.sum(ak_num(result)), np.float32)
    # Loop over volume and assign values:
    for volume in args['detector_config']:
        if isinstance(volume.electric_field, (float, int)):
            ids = epix.awkward_to_flat_numpy(result['vol_id'])
            m = ids == volume.volume_id
            efields[m] = volume.electric_field
        else:
            efields = volume.electric_field(epix.awkward_to_flat_numpy(result.x),
                                            epix.awkward_to_flat_numpy(result.y),
                                            epix.awkward_to_flat_numpy(result.z)
                                            )

    result['e_field'] = epix.reshape_awkward(efields, ak_num(result))

    # Sort entries (in an event) by in time, then chop all delayed
    # events which are too far away from the rest.
    # (This is a requirement of WFSim)
    result = result[ak.argsort(result['t'])]
    dt = calc_dt(result)
    result = result[dt <= args['max_delay']]

    if args['debug']:
        print('Generating photons and electrons for events')
    # Generate quanta:
    if len(result) > 0:
        photons, electrons, excitons = epix.quanta_from_NEST(epix.awkward_to_flat_numpy(result['ed']),
                                                             epix.awkward_to_flat_numpy(result['nestid']),
                                                             epix.awkward_to_flat_numpy(result['e_field']),
                                                             epix.awkward_to_flat_numpy(result['A']),
                                                             epix.awkward_to_flat_numpy(result['Z']),
                                                             epix.awkward_to_flat_numpy(result['create_S2']),
                                                             density=epix.awkward_to_flat_numpy(result['xe_density']))
        result['photons'] = epix.reshape_awkward(photons, ak_num(result['ed']))
        result['electrons'] = epix.reshape_awkward(electrons, ak_num(result['ed']))
        result['excitons'] = epix.reshape_awkward(excitons, ak_num(result['ed']))
    else:
        result['photons'] = np.empty(0)
        result['electrons'] = np.empty(0)
        result['excitons'] = np.empty(0)

    if args['debug']:
        _ = monitor_time(tnow, 'get quanta.')

    # Separate events in time
    number_of_events = len(result["t"])
    if args['source_rate'] == -1:
        # Only needed for a clean separation:
        result['t'] = calc_dt(result)

        dt = epix.times_for_clean_separation(number_of_events, args['max_delay'])
        if args['debug']:
            print('Clean event separation')
    elif args['source_rate'] == 0:
        # In case no delay should be applied we just add zeros
        dt = np.zeros(number_of_events)
    else:
        # Rate offset computed based on the specified rate and job_id.
        # Assumes all jobs were started with the same number of events.
        offset = (args['job_number']*n_simulated_events)/args['source_rate']
        dt = epix.times_from_fixed_rate(args['source_rate'],
                                        number_of_events,
                                        n_simulated_events,
                                        offset
                                        )
        if args['debug']:
            print(f"Fixed event rate of {args['source_rate']} Hz")

    result['t'] = apply_time_offset(result, dt)

    # Reshape instructions:
    if args['debug'] & (len(result) == 0):
        warnings.warn('No interactions left, return empty DataFrame.')
    instructions = epix.awkward_to_wfsim_row_style(result)
    if args['source_rate'] != 0:
        # Only sort by time again if source rates were applied, otherwise
        # things are already sorted within the events and should stay this way.
        instructions = np.sort(instructions, order='time')

    ins_df = pd.DataFrame(instructions)

    if return_df:
        if args['output_path'] and not os.path.isdir(args['output_path']):
            os.makedirs(args['output_path'])

        output_path_and_name = os.path.join(args['output_path'], args['file_name'][:-5] + "_wfsim_instructions.csv")
        if os.path.isfile(output_path_and_name):
            warnings.warn("Output file already exists - Overwriting")
        ins_df.to_csv(output_path_and_name, index=False)

        print('Done')
        print('Instructions saved to ', output_path_and_name)
        if args['debug']:
            _ = monitor_time(starttime, 'run epix.')

    if return_wfsim_instructions:
        return instructions


def monitor_time(prev_time, task):
    t = time.time()
    print(f'It took {(t - prev_time):.4f} sec to {task}')
    return t


def setup(args):
    """
    Function which sets-up configurations. (for strax conversion)
    Is returning the dict like this nessecairy?
    :return:
    """
    # Getting file path and split it into directory and file name:
    if not ('path' in args and 'file_name' in args):
        p = args['input_file']
        if '/' in p:
            p = p.split('/')
        else:
            p = ["", p]
        if p[0] == "":
            p[0] = "/"
        args['path'] = os.path.join(*p[:-1])
        args['file_name'] = p[-1]

    # Init detector volume according to settings and get outer_cylinder
    # for data reduction of non-relevant interactions.
    args['detector_config'] = epix.init_detector(args['detector'].lower(), args['detector_config_override'])
    outer_cylinder = getattr(epix.detectors, args['detector'].lower())
    _, args['outer_cylinder'] = outer_cylinder()
    return args
