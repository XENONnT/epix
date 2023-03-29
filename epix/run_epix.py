import strax

from input import input_plugin
from clustering import clustering
from cluster_merging import cluster_merging
from electric_field import electic_field
from nest_quanta_generation import nest_yields
from bbf_quanta_generation import bbf_yields
from output import output_plugin



st = strax.Context(register = [input_plugin,
                               clustering,
                               cluster_merging,
                               electic_field,
                               nest_yields,
                               #bbf_yields,
                               output_plugin],
                   storage = [strax.DataDirectory('./epix_data')]
                  )

st.set_config({"path": "/project2/lgrandi/xenonnt/simulations/testing",
               "file_name": "pmt_neutrons_100.root",
               "ChunkSize": 50
              })

st.make("00000", "geant4_interactions")
st.make("00000", "cluster_index")
st.make("00000", "clustered_interactions")
st.make("00000", "electic_field_values")
st.make("00000", "quanta")
st.make("00000", "wfsim_instructions")

wfsim_instructions = st.get_df("00000",[ "wfsim_instructions"])