import strax
import time
import os
import epix
import wfsim
import numpy as np
import pandas as pd
import awkward1 as ak

@strax.takes_config(
strax.Option('input_file',
             help='Input Geant4 ROOT file', track=True),
strax.Option('entry_stop', type=int, default=None,
             help='Number of entries to read from first. Defaulted to all'),
strax.Option('events_per_chunk', type=int, default=200,
             help='Number of events put in a chunk'),
strax.Option('chunk_time_seperation', type=int, default=int(1e10),
             help='Time seperation between chunks'),
strax.Option('micro_separation', type=float, default=0.05, track=True,
             help='Spatial resolution for DBSCAN micro-clustering [mm]'),
strax.Option('micro_separation_time', type=float, default=10., track=True,
             help='Time resolution for DBSCAN micro-clustering [ns]'),
strax.Option('tag_cluster_by', type=str, default='time', track=True,
             help=('Classification of the type of particle of a cluster, '
                   'based on most energetic contributor ("energy") or first '
                   'depositing particle ("time")')),
strax.Option('max_delay', type=float, track=True, default=1e7, #ns
             help='Maximal time delay to first interaction which will be stored [ns]'),
strax.Option('timing', type=bool, default=False,
             help='If true will print out the time needed.'),
strax.Option('output_path', default="",
             help=('Optional output path. If not specified the result will be saved'
                   'in the same dir as the input file.')),
strax.Option('detector', default='XENONnT',
             help=('Detector which should be used. Has to be defined in epix.detectors.')),
strax.Option('epix_config_override', type=str, default='',
             help='Config file to overwrite default detector settings.'),
strax.Option('event_rate_epix', type=float, default=-1,
             help='Event rate for event separation. Use -1 for clean simulations'
                  'or give a rate >0 to space events randomly.'),
strax.Option('debug', default=False,
             help=('If specifed additional information is printed to the console.')))
class EpixInstructions(strax.Plugin):
    '''strax plugin providing instructions for wfsim. Needs a Geant4 mc file as input and
     will return the instructions for wfsim in strax format. Queries the epix.run_epix script'''
    depends_on = tuple()
    provides = 'epix_instructions'
    data_kind  = 'wfsim_instructions'
    dtype = wfsim.strax_interface.instruction_dtype
    parallel = False
    rechunk_on_save=False
    finished=False
    last_endtime=0
    __version__ = '0.0.2'

    def compute(self,chunk_i):
        if chunk_i==0:
            self.config = epix.run_epix.setup(self.config)
            self.instructions = epix.run_epix.main(args=self.config,
                                                   return_wfsim_instructions=True)

        # See if there is still a full chunk not yet read out. If not, read out what is left and stop
        if len(self.instructions)<(chunk_i+1)*self.config['events_per_chunk']:
            instructions = self.instructions[chunk_i*self.config['events_per_chunk']:]
            self.finished = True
        else:
            instructions = self.instructions[chunk_i*self.events_per_chunk:(chunk_i+1)*self.events_per_chunk]

        endtime = self.last_endtime
        self.last_endtime = instructions['endtime'][-1]
        # Make some artifical time spacing
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

