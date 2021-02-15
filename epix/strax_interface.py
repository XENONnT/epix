import strax
import epix
import wfsim

def isNumber(x):
    try:
        float(x)
        return True
    except ValueError:
        return False

@strax.takes_config(
strax.Option('input_file',
                    help='Input Geant4 ROOT file',track=True),
strax.Option('entry_stop', type=int,default=None,track=False,
                    help='Number of entries to read from first. Defaulted to all'),
strax.Option('events_per_chunk', type=int,default=200,track=False,
                    help='Number of events put in a chunk'),
strax.Option('chunk_time_seperation',type=int,default=int(1e10),track=False,
                    help='Time seperation between chunks'),
strax.Option('micro_separation', type=float, default=0.05,track=True,
                    help='Spatial resolution for DBSCAN micro-clustering [mm]'),
strax.Option('micro_separation_time',type=float, default=10., track=True,
                    help='Time resolution for DBSCAN micro-clustering [ns]'),
strax.Option('tag_cluster_by',  type=str,default='time',track=True,
                    help=('Classification of the type of particle of a cluster, '
                          'based on most energetic contributor ("energy") or first '
                          'depositing particle ("time")')),
strax.Option('max_delay', type=float,track=True,default=1e7, #ns
                    help='Maximal time delay to first interaction which will be stored [ns]'),
strax.Option('detector',default='XENONnT',
                    help=('Detector which should be used. Has to be defined in epix.detectors.')),
strax.Option('epix_config_override', type=str,
                        default='',
                        help='Config file to overwrite default detector settings.'),
strax.Option('source_rate', type=float,default=-1,track=False,
                    help='Event rate for event separation. Use -1 for clean simulations'
                         'or give a rate >0 to space events randomly.'),
strax.Option('debug', default=False,track=False,
                    help=('If specifed additional information is printed to the consol.')))
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

    def compute(self,chunk_i):
        if chunk_i==0:
            self.config = epix.run_epix.setup(self.config)
            self.instructions = epix.run_epix.main(args=self.config,
                                                   return_wfsim_instructions=True)


        #See if there is still a full chunk not jet read out. If not read out what is left and stop
        if len(self.instructions)<(chunk_i+1)*self.config['events_per_chunk']:
            instructions = self.instructions[chunk_i*self.config['events_per_chunk']:]
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

