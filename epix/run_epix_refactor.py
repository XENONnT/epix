## epix structure proposal
import strax
import numpy as np

@strax.takes_config(
    strax.Option('path', default=".", track=False, infer_type=False,
                 help="Path to search for data"),
    strax.Option('file_name', track=False, infer_type=False,
                 help="File to open"),
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
    strax.Option('outer_cylinder', default=False, track=False, infer_type=False,
                 help="Dont know"),
    strax.Option('entry_start', default=0, track=False, infer_type=False,
                 help="First event to be read"),
    strax.Option('entry_stop', default=10, track=False, infer_type=False,
                 help="How many entries from the ROOT file you want to process"),
    strax.Option('cut_by_eventid', default=False, track=False, infer_type=False,
                 help="If selected, the next two arguments act on the G4 event id, and not the entry number (default)"),
    strax.Option('nr_only', default=False, track=False, infer_type=False,
                 help="Add if you want to filter only nuclear recoil events (maximum ER energy deposit 10 keV)"), 
)
class input_plugin(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = tuple()
    provides = "geant4_interactions"
    
    source_done = False
    
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('t', np.float64),
             ('ed', np.float32),
             ('type', "<U10"),
             ('trackid', np.int32),
             ('parenttype', "<U10"),
             ('parentid', np.int32),
             ('creaproc', "<U10"),
             ('edproc', "<U10"),
             ('evtid', np.int32),
             ('structure', np.int32),
            ]
    
    dtype = dtype + strax.time_fields

    def setup(self):

        self.epix_file_loader = epix.file_loader(self.path,
                                                 self.file_name,
                                                 self.debug,
                                                 outer_cylinder=self.outer_cylinder,
                                                 kwargs={'entry_start': self.entry_start,
                                                         'entry_stop': self.entry_stop},
                                                 cut_by_eventid=self.cut_by_eventid,
                                                 #cut_nr_only=self.nr_only,
                                                )
    
    def full_array_to_numpy(self, array):
    
        len_output = len(epix.awkward_to_flat_numpy(array["x"]))
        array_structure = np.array(epix.ak_num(array["x"]))
        array_structure = np.pad(array_structure, [0, len_output-len(array_structure)],constant_values = -1)

        numpy_data = np.zeros(len_output, dtype=self.dtype)

        for field in array.fields:
            numpy_data[field] = epix.awkward_to_flat_numpy(array[field])
        numpy_data["structure"] = array_structure
        
        return numpy_data
    

    def compute(self):

        inter, n_simulated_events = self.epix_file_loader.load_file()
        
        inter_reshaped = self.full_array_to_numpy(inter)
        
        inter_reshaped["time"] = (inter_reshaped["evtid"]+1) *1e9
        inter_reshaped["endtime"] = inter_reshaped["time"] +1e7
        
        return self.chunk(start=inter_reshaped['time'][0],
                          end=inter_reshaped['endtime'][-1],
                          data=inter_reshaped,
                          data_type='geant4_interactions')

    
    def is_ready(self, chunk):
        #For now, just one single large chunk. 
        return True if chunk == 0 else False

    def source_finished(self):
        return True


@strax.takes_config(
    strax.Option('micro_separation', default=0.005, track=False, infer_type=False,
                 help="DBSCAN clustering distance (mm)"),
    strax.Option('micro_separation_time', default = 10, track=False, infer_type=False,
                 help="Clustering time (ns)"),
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
    strax.Option('tag_cluster_by', default=False, track=False, infer_type=False,
                 help="decide if you tag the cluster (particle type, energy depositing process)\
                       according to first interaction in it (time) or most energetic (energy)"),
)
class clustering(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = "geant4_interactions"
    provides = "clustered_interactions"
    
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('t', np.float64),
             ('ed', np.float64),
             ('nestid', np.int64),
             ('A', np.int64),
             ('Z', np.int64),
             ('structure', np.int64),
            ]
    
    dtype = dtype + strax.time_fields
    
    def full_array_to_numpy(self, array):
    
        len_output = len(epix.awkward_to_flat_numpy(array["x"]))
        array_structure = np.array(epix.ak_num(array["x"]))
        
        fake_event_time = np.repeat(np.arange(len(result["t"])), array_structure)
        
        array_structure = np.pad(array_structure, [0, len_output-len(array_structure)],constant_values = -1)

        numpy_data = np.zeros(len_output, dtype=self.dtype)

        for field in array.fields:
            numpy_data[field] = epix.awkward_to_flat_numpy(array[field])
        numpy_data["structure"] = array_structure
        numpy_data["time"] = (fake_event_time+1)*1e9
        numpy_data["endtime"] = numpy_data["time"] +1e7
        
        return numpy_data

    def compute(self, geant4_interactions):
        
        inter = ak.from_numpy(np.empty(1, dtype=geant4_interactions.dtype))
        structure = geant4_interactions["structure"][geant4_interactions["structure"]>=0]
        
        for field in inter.fields:
            inter[field] =  epix.reshape_awkward(geant4_interactions[field], structure)
        
        inter = epix.find_cluster(inter, self.micro_separation/10, self.micro_separation_time)
        result = epix.cluster(inter, self.tag_cluster_by == 'energy')
        
        result = self.full_array_to_numpy(result)
        
        return result

class electic_field(strax.plugin):
    depends_on = "clustered_interactions"
    provides = "electic_field_values"
    dtype = "electic_field_data_that_can_be_added_to_clustered_interactions"

    def compute():
        """Again, use the epix functions"""
        pass

class yields(strax.plugin):
    depends_on = ["clustered_interactions", "electric_field_values"]
    provides = "quanta"
    dtype = "quanta_data_that_can_be_added_to_clustered_interactions"

    def compute():
        "Again, epix functions are already there"
        pass

class time_separation(strax.plugin):
    depends_on = ["clustered_interactions"]:
    provides = "interaction_times"
    dtype = "interaction_times_that_can_be_added_to_clustered_interactions"

    def compute():
        "Again, epix functions are already there"
        pass

class output_plugin(strax.plugin):
    depends_on = ["clustered_interactions", "quanta", "interaction_times"]
    returns = "wfsim_instructions"
    dtype = "wfsim_instructions_dtype"

    def compute():
        "this one will be new, but just merging the data of the connected plugins"
        pass