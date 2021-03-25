import 'helpers.py'
import 'fast_simulator.py'

st = straxen.contexts.xenonnt_simulation()
run_id = '1'
st.register(StraxSimulator)
st.set_config=(dict(nchunk=1, event_rate=5, chunk_size=500,))

resource_path='/home/pkavrigin/g4-analysis/private_nt_aux_files/sim_files/'
nt_config=get_resource(resource_path+'fax_config_nt_low_field.json', fmt='json')

raw_data_dir='/home/pkavrigin/g4-output/2021-03-16_AmBe_100events/'
raw_data_filename='AmBe_TUT_100events.root'

epix_args={'path':raw_data_dir,
           'file_name':raw_data_filename,
           'debug':False,
           'entry_start':0,
           'entry_stop':100,
           'cut_by_eventid':False,
           'micro_separation_time':10.0,
           'micro_separation':0.005,
           'tag_cluster_by':'time',
           'max_delay':1e-7,
           'source_rate':-1}

configuration_files={'s1_relative_lce_map':pax_file('XENON1T_s1_xyz_lce_true_kr83m_SR1_pax-680_fdc-3d_v0.json'),
                     's2_xy_correction_map':pax_file('XENON1T_s2_xy_ly_SR1_v2.2.json'),
                     'photon_area_distribution':resource_path+'XENONnT_spe_distributions_20210305.csv',
                     's1_pattern_map':resource_path+'XENONnT_s1_xyz_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl',
                     's2_pattern_map':resource_path+'XENONnT_s2_xy_patterns_LCE_corrected_qes_MCva43fa9b_wires.pkl'}

st.config.update(dict(fax_config=nt_config,
                      g4_file=raw_data_dir+raw_data_filename,
                      epix_config=epix_args,
                      xy_resolution=5,
                      z_resolution=1,
                      configuration_files=configuration_files))

sim=StraxSimulator()
sim.config=st.config
sim.setup()
sim_output=sim.compute()

sim_output_df=pd.DataFrame(sim_output)
display(sim_output_df)