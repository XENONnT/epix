## epix structure proposal 

class input_plugin(strax.plugin):
    depends_on = None
    provides = "geant4_interactions"
    dtype = "some_list_of_geant4_interactions"

    def compute():
        """Use the epix io functions here"""
        pass


class clustering(strax.plugin):
    depends_on = "geant4_interactions"
    provides = "clustered_interactions"
    dtype = "some_list_of_clustered_interactions"

    def compute():
        """Use the epix clustering functions here"""
        pass

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