## epix structure proposal
import strax
import numpy as np
import epix
import awkward as ak

@strax.takes_config(
    strax.Option('path', default=".", track=False, infer_type=False,
                 help="Path to search for data"),
    strax.Option('file_name', track=False, infer_type=False,
                 help="File to open"),
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
    strax.Option('entry_start', default=0, track=False, infer_type=False,
                 help="First event to be read"),
    strax.Option('entry_stop', default=10, track=False, infer_type=False,
                 help="How many entries from the ROOT file you want to process"),
    strax.Option('cut_by_eventid', default=False, track=False, infer_type=False,
                 help="If selected, the next two arguments act on the G4 event id, and not the entry number (default)"),
    strax.Option('nr_only', default=False, track=False, infer_type=False,
                 help="Add if you want to filter only nuclear recoil events (maximum ER energy deposit 10 keV)"),
    strax.Option('Detector', default="XENONnT", track=False, infer_type=False,
                 help="Detector to be used. Has to be defined in epix.detectors"),
    strax.Option('DetectorConfigOverride', default=None, track=False, infer_type=False,
                 help="Config file to overwrite default epix.detectors settings; see examples in the configs folder"),
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
             ('trackid', np.int64),
             ('parenttype', "<U10"),
             ('parentid', np.int64),
             ('creaproc', "<U10"),
             ('edproc', "<U10"),
             ('evtid', np.int64),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('structure', np.int64),
            ]
    
    dtype = dtype + strax.time_fields

    def setup(self):
        
        #Do the volume cuts here #Maybe we can move these lines somewhere else?
        self.detector_config = epix.init_detector(self.Detector.lower(), self.DetectorConfigOverride)
        outer_cylinder = getattr(epix.detectors, self.Detector.lower())
        outer_cylinder = outer_cylinder()

        self.epix_file_loader = epix.file_loader(self.path,
                                                 self.file_name,
                                                 self.debug,
                                                 outer_cylinder=None, #This is not running 
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
        # For this plugin we'll smash everything into 1 chunk, should be oke
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
)
class clustering(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("geant4_interactions",)
    
    provides = "cluster_index"
    
    dtype = [('cluster_ids', np.int64),
            ]
    
    dtype = dtype + strax.time_fields
    

    def compute(self, geant4_interactions):
        
        inter = ak.from_numpy(np.empty(1, dtype=geant4_interactions.dtype))
        structure = geant4_interactions["structure"][geant4_interactions["structure"]>=0]
        
        for field in inter.fields:
            inter[field] = epix.reshape_awkward(geant4_interactions[field], structure)
        
        #We can optimize the find_cluster function for the refactor! No need to return more than cluster_ids , no need to bring it into awkward again
        inter = epix.find_cluster(inter, self.micro_separation/10, self.micro_separation_time)
        cluster_ids = inter['cluster_ids']
        
        len_output = len(epix.awkward_to_flat_numpy(cluster_ids))
        numpy_data = np.zeros(len_output, dtype=self.dtype)
        numpy_data["cluster_ids"] = epix.awkward_to_flat_numpy(cluster_ids)
        
        numpy_data["time"] = geant4_interactions["time"]
        numpy_data["endtime"] = geant4_interactions["endtime"]
        
        return numpy_data
        


@strax.takes_config(
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
    strax.Option('tag_cluster_by', default=False, track=False, infer_type=False,
                 help="decide if you tag the cluster (particle type, energy depositing process)\
                       according to first interaction in it (time) or most energetic (energy)"),
)
@strax.takes_config(
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
    strax.Option('tag_cluster_by', default=False, track=False, infer_type=False,
                 help="decide if you tag the cluster (particle type, energy depositing process)\
                       according to first interaction in it (time) or most energetic (energy)"),
    strax.Option('Detector', default="XENONnT", track=False, infer_type=False,
                 help="Detector to be used. Has to be defined in epix.detectors"),
    strax.Option('DetectorConfigOverride', default=None, track=False, infer_type=False,
                 help="Config file to overwrite default epix.detectors settings; see examples in the configs folder"),
)
class cluster_merging(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("geant4_interactions", "cluster_index")
    
    provides = "clustered_interactions"
    
    dtype = [('x', np.float32),
             ('y', np.float32),
             ('z', np.float32),
             ('t', np.float64),
             ('ed', np.float64),
             ('nestid', np.int64),
             ('A', np.int64),
             ('Z', np.int64),
             ('evtid', np.int64),
             ('x_pri', np.float32),
             ('y_pri', np.float32),
             ('z_pri', np.float32),
             ('xe_density', np.float32),
             ('vol_id', np.int64),
             ('create_S2', np.bool8),
             ('structure', np.int64),
            ]
    
    dtype = dtype + strax.time_fields
    
    def setup(self):
        #Do the volume cuts here #Maybe we can move these lines somewhere else?
        self.detector_config = epix.init_detector(self.Detector.lower(), self.DetectorConfigOverride)
        
    def full_array_to_numpy(self, array):
    
        len_output = len(epix.awkward_to_flat_numpy(array["x"]))
        array_structure = np.array(epix.ak_num(array["x"]))
        array_structure = np.pad(array_structure, [0, len_output-len(array_structure)],constant_values = -1)

        numpy_data = np.zeros(len_output, dtype=self.dtype)

        for field in array.fields:
            numpy_data[field] = epix.awkward_to_flat_numpy(array[field])
        numpy_data["structure"] = array_structure
        
        return numpy_data


    def compute(self, geant4_interactions):
        
        inter = ak.from_numpy(np.empty(1, dtype=geant4_interactions.dtype))
        structure = geant4_interactions["structure"][geant4_interactions["structure"]>=0]
        
        for field in inter.fields:
            inter[field] = epix.reshape_awkward(geant4_interactions[field], structure)

        result = epix.cluster(inter, self.tag_cluster_by == 'energy')
        
        result['evtid'] = ak.broadcast_arrays(inter['evtid'][:, 0], result['ed'])[0]
        # Add x_pri, y_pri, z_pri again:
        result['x_pri'] = ak.broadcast_arrays(inter['x_pri'][:, 0], result['ed'])[0]
        result['y_pri'] = ak.broadcast_arrays(inter['y_pri'][:, 0], result['ed'])[0]
        result['z_pri'] = ak.broadcast_arrays(inter['z_pri'][:, 0], result['ed'])[0]
        
        
        res_det = epix.in_sensitive_volume(result, self.detector_config)
        for field in res_det.fields:
            result[field] = res_det[field]
        m = result['vol_id'] > 0  # All volumes have an id larger zero
        result = result[m]
        
        # Removing now empty events as a result of the selection above:
        m = epix.ak_num(result['ed']) > 0
        result = result[m]
        
        result = self.full_array_to_numpy(result)
        
        result["time"] = (result["evtid"]+1) *1e9
        result["endtime"] = result["time"] +1e7
        
        return result
        

@strax.takes_config(
    strax.Option('debug', default=False, track=False, infer_type=False,
                 help="Show debug informations"),
    strax.Option('Detector', default="XENONnT", track=False, infer_type=False,
                 help="Detector to be used. Has to be defined in epix.detectors"),
    strax.Option('DetectorConfigOverride', default=None, track=False, infer_type=False,
                 help="Config file to overwrite default epix.detectors settings; see examples in the configs folder"),
)
class electic_field(strax.Plugin):
    
    __version__ = "0.0.0"
    
    depends_on = ("clustered_interactions",)
    provides = "electic_field_values"
    
    dtype = [('e_field', np.int64),
            ]
    
    dtype = dtype + strax.time_fields
    
    def setup(self):
        #Do the volume cuts here #Maybe we can move these lines somewhere else?
        self.detector_config = epix.init_detector(self.Detector.lower(), self.DetectorConfigOverride)
 
    #Why is geant4_interactions given from straxen? It should be called clustered_interactions or??
    def compute(self, geant4_interactions):
        
        result = np.zeros(len(geant4_interactions), dtype=self.dtype)
        result["time"] = geant4_interactions["time"]
        result["endtime"] = geant4_interactions["endtime"]

        efields = result["e_field"]
        
        for volume in self.detector_config:
            if isinstance(volume.electric_field, (float, int)):
                ids = geant4_interactions['vol_id']
                m = ids == volume.volume_id
                efields[m] = volume.electric_field
            else:
                efields = volume.electric_field(geant4_interactions.x,
                                                geant4_interactions.y,
                                                geant4_interactions.z
                                                )

        result["e_field"] = efields
        
        return result
        
        

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