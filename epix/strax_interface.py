import strax
import time
import os
import epix
import wfsim
import numpy as np
import pandas as pd
import awkward1 as ak

def isNumber(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

@strax.takes_config(
strax.Option('InputFile',
                    help='Input Geant4 ROOT file',track=True),
strax.Option('EntryStop', type=int,default=None,
                    help='Number of entries to read from first. Defaulted to all'),
strax.Option('NumberOfChunks', type=int,default=5,
                    help='Number of chunks to be made from the data'),
strax.Option('EventsPerChunk', type=int,default=200,
                    help='Number of events put in a chunk'),
strax.Option('ChunkTimeSeperation',type=int,default=int(1e10),
                    help='Time seperation between chunks'),
strax.Option('MicroSeparation', type=float, default=0.05,track=True,
                    help='Spatial resolution for DBSCAN micro-clustering [mm]'),
strax.Option('MicroSeparationTime',type=float, default=10., track=True,
                    help='Time resolution for DBSCAN micro-clustering [ns]'),
strax.Option('TagClusterBy',  type=str,default='time',track=True,
                    help=('Classification of the type of particle of a cluster, '
                          'based on most energetic contributor ("energy") or first '
                          'depositing particle ("time")')),
strax.Option('Efield', default=200,track=True,
                    help=('Drift field map as text file ("r z E", with '
                          'length in cm and field in V/cm) or as a constant (in V/cm; '
                          'recommended only for testing)')),
strax.Option('MaxDelay', type=float,track=True,default=1e7, #ns
                    help='Maximal time delay to first interaction which will be stored [ns]'),
strax.Option('Timing', type=bool,default=False,
                    help='If true will print out the time needed.'),
strax.Option('OutputPath',  default="",
                    help=('Optional output path. If not specified the result will be saved'
                        'in the same dir as the input file.')))
class EpixInstructions(strax.Plugin):
    '''Strax plugin providing instructions for wfsim. Needs a GEANT4 mc file as input and
     will return the instructions for wfsim in strax format. The code in compute is pretty much
     a copy of the run_epix script'''
    depends_on = tuple()
    provides = 'epix_instructions'
    data_kind  = 'wfsim_instructions'
    dtype = wfsim.strax_interface.instruction_dtype
    parallel = False
    rechunk_on_save=False
    finished=False
    last_endtime=0
    __version__ = '0.0.2'

    def ReadAndProcessData(self):
        # This is slightly ugly. We'll do all the processing in this function.
        # This guy will be run for the first chunk and then
        # the compute method will only be used for the chunking of the result
        c = self.config
        self.events_per_chunk = c['NumberOfChunks']*c['EventsPerChunk']
        assert c['TagClusterBy'] in ('time','energy'), 'TagClusterBy must be either time or energy'

        if c['Timing']:
            starttime = time.time()
            tnow = starttime
        # Loading the data:
        p = c['InputFile']
        p = p.split('/')
        if p[0] == "":
            p[0] = "/"
        path = os.path.join(*p[:-1])
        file_name = p[-1]

        print(f'Reading in root file {file_name}')
        inter = epix.loader(path, file_name, kwargs_uproot_ararys={'entry_stop': c['EntryStop']})
        if c['Timing']:
            t = time.time()
            print(f'It took {round(t - tnow, 5)} sec to load the data.')
            tnow = t

        print((f"Finding clusters of interactions with a dr = {c['MicroSeparation']} mm"
               f" and dt = {c['MicroSeparationTime']} ns"))
        inter = epix.find_cluster(inter, c['MicroSeparation'] / 10, c['MicroSeparationTime'])
        if c['Timing']:
            t = time.time()
            print(f'It took {round(t - tnow, 5)} sec to find clusters.')
            tnow = t

        result = epix.cluster(inter, c['TagClusterBy'] == 'energy')
        if c['Timing']:
            t = time.time()
            print(f'It took {round(t - tnow, 5)} sec to cluster events.')
            tnow = t

        # Add eventid again:
        result['evtid'] = ak.broadcast_arrays(inter['evtid'][:, 0], result['ed'])[0]

        # Sort detector volumes and keep interactions in selected ones:
        sensitive_volumes = [epix.tpc, epix.below_cathode]  # TODO: add options
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
        if isNumber(c['Efield']):
            efields = np.ones(np.sum(ak.num(result)), np.float32) * c['Efield']
        else:
            E_field_handler = epix.MyElectricFieldHandler(c['Efield'])
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
        result = result[result['t'] <= c['MaxDelay']]
        # Secondly truly separate events by time (1.1 times the max time),
        # with first event starting at max time (needed for wfsim)
        dt = np.arange(1, len(result['t']) + 1) + np.arange(1, len(result['t']) + 1) / 10
        dt *= c['MaxDelay']
        result['t'] = (result['t'][:, :] + result['t'][:, 0] + dt)

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
        if c['Timing']:
            t = time.time()
            print(f'It took {round(t - tnow, 5)} sec to get quanta.')
            tnow = t

        # Reshape instructions:
        # TODO: Change me....
        instructions = epix.awkward_to_really_awkward(result)
        # Remove entries with no quanta
        self.instructions = instructions[instructions['amp'] > 0]

        print('Done')
        if c['Timing']:
            t = time.time()
            print(f'It took {round(t - starttime, 5)} sec to process file.')


    def compute(self,chunk_i):
        if chunk_i==0:
            self.ReadAndProcessData()

        #See if there is still a full chunk not jet read out. If not read out what is left and stop
        if len(self.instructions)<(chunk_i+1)*self.events_per_chunk:
            instructions = self.instructions[chunk_i*self.events_per_chunk:]
            self.finished = True
        else:
            instructions = self.instructions[chunk_i*self.events_per_chunk:(chunk_i+1)*self.events_per_chunk]

        endtime = self.last_endtime
        self.last_endtime = instructions['endtime'][-1]
        #Make some artifical time spacing
        return self.chunk(
                        start=int(endtime),
                        end=int(instructions['endtime'][-1]),
                        data=instructions,  
                        data_type=self.data_kind)

    def is_ready(self, chunk_i):
        """Return whether the chunk chunk_i is ready for reading.
        Returns True by default; override if you make an online input plugin.
        """
        ready=True
        if self.finished:
            ready=False
        return ready

    def source_finished(self):
        """Return whether all chunks the plugin wants to read have been written.
        Only called for online input plugins.
        """
        return self.finished

