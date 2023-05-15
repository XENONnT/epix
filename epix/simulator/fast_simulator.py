import pickle
import time

import strax
import straxen
import wfsim.load_resource
import epix
import numpy as np
import pandas as pd
import os
from copy import deepcopy
from immutabledict import immutabledict
from .GenerateEvents import GenerateEvents
from .GenerateNveto import NVetoUtils
from .helpers import Helpers
import warnings

# 2023-02-19: configuration_files:
#   'nv_pmt_qe':'nveto_pmt_qe.json',
#   'photon_area_distribution':'XENONnT_SR0_spe_distributions_20210713_no_noise_scaled.csv',
#   's1_pattern_map': 'XENONnT_s1_xyz_patterns_LCE_MCvf051911_wires.pkl',
#   's2_pattern_map': 'XENONnT_s2_xy_patterns_GXe_LCE_corrected_qes_MCv4.3.0_wires.pkl',
#   's2_separation_bdt': 's2_separation_decision_tree_fast_sim.p'
#   (FROM: /dali/lgrandi/jgrigat/s2_separation/s2_separation_decision_tree_fast_sim.p)

def monitor_time(prev_time, task):
    t = time.time()
    print(f'It took {(t - prev_time):.4f} sec to {task}')
    return t


class Simulator:
    """Simulator class for epix to go from  epix instructions to fully processed data"""

    def __init__(self, instructions_epix, config, resource):
        self.config = config
        self.ge = GenerateEvents()
        self.resource = resource
        self.simulation_functions = sorted(
            # get the functions of GenerateEvents in order to run through them all
            [getattr(self.ge, field) for field in dir(self.ge)
             if hasattr(getattr(self.ge, field), "order")
             ],
            # sort them by their order
            key=(lambda field: field.order)
        )
        self.instructions_epix = instructions_epix

    def cluster_events(self, ):
        """Events have more than 1 s1/s2. Here we throw away all of them except the largest 2
         Take the position to be that of the main s2
         And strax wants some start and endtime, so we make something up
        
        First do the macro clustering. Clustered instructions will be flagged with amp=-1,
        so we can safely throw those out"""
        start_time = time.time()
        self.nn_weights = pickle.load(open('/home/jgrigat/epix/epix/simulator/nn_weights.p', 'rb+'))
        Helpers.macro_cluster_events(self.nn_weights, self.instructions_epix, self.config)
        self.instructions_epix = self.instructions_epix[self.instructions_epix['amp'] != -1]

        event_numbers = np.unique(self.instructions_epix['event_number'])
        ins_size = len(event_numbers)
        instructions = np.zeros(ins_size, dtype=StraxSimulator.dtype['events_tpc'])

        for ix in range(ins_size):
            i = instructions[ix]
            inst = self.instructions_epix[self.instructions_epix['event_number'] == event_numbers[ix]]
            inst_s1 = inst[inst['type'] == 1]
            inst_s2 = inst[inst['type'] == 2]

            s1 = np.argsort(inst_s1['amp'])
            s2 = np.argsort(inst_s2['amp'])

            if len(s1) < 1 or len(s2) < 1:
                continue

            i['s1_area'] = np.sum(inst_s1['amp'])

            i['s2_area'] = inst_s2[s2[-1]]['amp']
            i['e_dep'] = inst_s2[s2[-1]]['e_dep']

            if len(s2) > 1:
                i['alt_s2_area'] = inst_s2[s2[-2]]['amp']
                i['alt_e_dep'] = inst_s2[s2[-2]]['e_dep']
                i['alt_s2_x_true'] = inst_s2[s2[-2]]['x']
                i['alt_s2_y_true'] = inst_s2[s2[-2]]['y']
                i['alt_s2_z_true'] = inst_s2[s2[-2]]['z']

            i['x_true'] = inst_s2[s2[-1]]['x']
            i['y_true'] = inst_s2[s2[-1]]['y']
            i['z_true'] = inst_s2[s2[-1]]['z']

            i['x_pri'] = inst_s2[s2[-1]]['x_pri']
            i['y_pri'] = inst_s2[s2[-1]]['y_pri']
            i['z_pri'] = inst_s2[s2[-1]]['z_pri']

            i['g4id'] = inst_s2[s2[-1]]['g4id']

            # Strax wants begin and endtimes
            i['time'] = ix * 1000
            i['endtime'] = ix * 1000 + 1

        instructions = instructions[~((instructions['time'] == 0) & (instructions['endtime'] == 0))]
        self.instructions = instructions
        print(f'It took {(time.time() - start_time):.4f} sec to macro cluster events')

    def simulate(self, ):
        for func in self.simulation_functions:
            start = time.time()
            func(i=self.instructions,
                 config=self.config,
                 resource=self.resource)
            monitor_time(start, func.__name__)

    def run_simulator(self, ):
        self.cluster_events()
        # TODO this is for debug purposes - not super nice. We should have a fastsim_truth data type
        if isinstance(self.config['epix_config'].get('save_epix', None), str):
            file_name = self.config['epix_config']['file_name'].split('/')[-1][:-4] + '_instruction_after_macro_clustering'
            epix_path = os.path.join(self.config['epix_config']['save_epix'] ,file_name)
            print('Saving epix instruction: ', epix_path)
            np.save(epix_path, self.instructions)
        self.simulate()
        return self.instructions


# We should think a bit more to detector_config_override
# and tell fast_sim to look into epix_args
# also because one entry of self.config is epix_config
@strax.takes_config(
    strax.Option('detector', default='XENONnT', help='Detector model'),
    strax.Option('detector_config_override', default='', help='For the electric field, otherwise 200 V/cm'),
    strax.Option('g4_file', help='G4 file to simulate'),
    strax.Option('epix_config', default=dict(), help='Configuration file for epix', ),
    strax.Option('configuration_files', default=dict(), help='Files required for simulating'),
    strax.Option('fax_config', help='Fax configuration to load'),
    strax.Option('fax_config_overrides', help='Fax configuration to override', default=None),
    strax.Option('xy_resolution', default=0, help='xy position resolution (cm)'),
    strax.Option('z_resolution', default=0, help='xy position resolution (cm)'),
    strax.Option('nv_spe_resolution', default=0.40, help='nVeto SPE resolution'),
    strax.Option('nv_spe_res_threshold', default=0.50, help='nVeto SPE acceptance threshold'),
    strax.Option('nv_max_time_ns', default=1e7, help='nVeto maximum time for the acceptance of PMT hits in event'),
    strax.Option('s2_clustering_algorithm', default='bdt', help='Macroclustering algorithm for S2, [ nsort | naive | bdt ]'),
)
class StraxSimulator(strax.Plugin):
    provides = ('events_tpc', 'events_nveto')
    depends_on = ()
    data_kind = immutabledict(zip(provides, provides))
    dtype = dict(events_tpc=[('time', np.int64),
                             ('endtime', np.int64),
                             ('cs1', np.float),
                             ('cs2', np.float),
                             ('alt_cs2', np.float),
                             ('drift_time', np.float),
                             ('s1_area', np.float),
                             ('s2_area', np.float),
                             ('alt_s2_area', np.float),
                             ('alt_s2_interaction_drift_time', np.float),
                             ('s2_x', np.float),
                             ('s2_y', np.float),
                             ('alt_s2_x', np.float),
                             ('alt_s2_y', np.float),
                             ('x', np.float),
                             ('alt_s2_x_fdc', np.float),
                             ('y', np.float),
                             ('alt_s2_y_fdc', np.float),
                             ('r', np.float),
                             ('alt_s2_r_fdc', np.float),
                             ('z', np.float),
                             ('z_dv_corr', np.float),
                             ('alt_s2_z', np.float),
                             ('alt_s2_z_dv_corr', np.float),
                             ('r_naive', np.float),
                             ('alt_s2_r_naive', np.float),
                             ('z_naive', np.float),
                             ('alt_s2_z_naive', np.float),
                             ('r_field_distortion_correction', np.float),
                             ('alt_s2_r_field_distortion_correction', np.float),
                             ('z_field_distortion_correction', np.float32),
                             ('alt_s2_z_field_distortion_correction', np.float32),
                             ('alt_s2_theta', np.float32),
                             ('theta', np.float32),

                             # Truth values
                             ('x_true', np.float),
                             ('y_true', np.float),
                             ('z_true', np.float),
                             ('r_true', np.float),
                             ('theta_true', np.float),
                             ('alt_s2_x_true', np.float),
                             ('alt_s2_y_true', np.float),
                             ('alt_s2_z_true', np.float),
                             ('alt_s2_r_true', np.float),
                             ('alt_s2_theta_true', np.float),
                             ('e_dep', np.float),
                             ('alt_e_dep', np.float),
                             ('x_pri', np.float),
                             ('y_pri', np.float),
                             ('z_pri', np.float),
                             ('g4id', np.int)],
                 events_nveto=[('time', np.float),
                               ('endtime', np.float),
                               ('event_id', np.int),
                               ('channel', np.int), ])

    def load_resources(self):
        """First load the config through wfsim, then we add some things we'd like to have"""
        self.resource = wfsim.load_resource.load_config(self.config)
        # add mean photon area distribution with avg values.
        self.resource.mean_photon_area_distribution = epix.Helpers.average_spe_distribution(self.resource.photon_area_distribution)
        self.resource.fdc_map = wfsim.load_resource.make_map(self.config['field_distortion_correction_map'], fmt='json.gz')
        # TODO check if NV simulation is working without this stuff
        #if 'nv_pmt_qe' in self.config['configuration_files'].keys():
        #            self.resource.nv_pmt_qe = straxen.get_resource(
        #                self.config['configuration_files']['nv_pmt_qe'], fmt='json')
        #else:
        #    warnings.warn('The configuration_files should not exist!'
        #                  'Everything should come by one config!'
        #                  'Since nv_pmt_qe config is missing in configuration_files, '
        #                  'the default one nveto_pmt_qe.json will be used')
        #    self.resource.nv_pmt_qe = straxen.get_resource('nveto_pmt_qe.json', fmt='json')

    def setup(self):
        print('Setup')
        overrides = self.config['fax_config_overrides']
        self.config.update(straxen.get_resource(self.config['fax_config'], fmt='json'))
        if overrides is not None:
            self.config.update(overrides)
        # TODO: this should not be needed here. Where is this info normally taken from?
        if 'gains' not in self.config.keys():
            self.config['gains'] =  [1] * 494
        if 'n_top_pmts' not in self.config.keys():
            self.config['n_top_pmts'] =  253
        if 'n_tpc_pmts' not in self.config.keys():
            self.config['n_tpc_pmts'] =  494
        print('Loading resources')
        self.load_resources()

    def get_nveto_data(self, ):
        file_loader = epix.io.file_loader(directory=self.config['epix_config']['path'],
                                          file_name=self.config['epix_config']['file_name'])
        file_tree, _ = file_loader._get_ttree()

        if 'pmthitID' in file_tree.keys():
            nv_hits = NVetoUtils.get_nv_hits(ttree=file_tree,
                                             pmt_nv_json_dict=self.resource.nv_pmt_qe,
                                             nveto_dtype=self.dtype['events_nveto'],
                                             SPE_Resolution=self.config['nv_spe_resolution'],
                                             SPE_ResThreshold=self.config['nv_spe_res_threshold'],
                                             max_time_ns=self.config['nv_max_time_ns'],
                                             batch_size=10000)
            return nv_hits

    def get_epix_instructions(self, ):
        epix_config = deepcopy(self.config['epix_config'])
        fn = epix_config.get('file_name', '')
        if fn.endswith('.csv'):
            print('Loading epix instructions from csv-file')
            epix_ins = np.load(epix_config['file_name'])
        elif fn.endswith('.root'):
            print('Generating epix instructions from root-file')

            detector = epix.init_detector(self.config['detector'].lower(), self.config['detector_config_override'])
            epix_config['detector_config'] = detector

            outer_cylinder = getattr(epix.detectors, 'xenonnt')
            _, outer_cylinder = outer_cylinder()
            epix_config['outer_cylinder'] = outer_cylinder
            epix_ins = epix.run_epix.main(epix_config, return_wfsim_instructions=True)

        # TODO I don't know if I like this.
        save_epix = epix_config.get('save_epix', False)
        if isinstance(save_epix, str):
            # assume save epix as path to store
            file_name = self.config['epix_config']['file_name'].split('/')[-1][:-4] + '_instruction'
            epix_path = os.path.join(self.config['epix_config']['save_epix'], file_name)
            print('Saving epix instruction: ', epix_path)
            np.save(epix_path, epix_ins)
        elif save_epix:
            # if save epix True store in normal path
            file_name = self.config['epix_config']['file_name'].split('/')[-1][:-4] + '_instruction'
            epix_path = os.path.join(self.config['epix_config']['path'], file_name)
            print(f'Saving epix instruction: {epix_path}')
        return epix_ins

    def compute(self):
        print('Compute')
        #simulated_data_nveto = self.get_nveto_data()
        simulated_data_nveto = None
        self.epix_instructions = self.get_epix_instructions()
        self.Simulator = Simulator(instructions_epix=self.epix_instructions,
                                   config=self.config,
                                   resource=self.resource)
        simulated_data = self.Simulator.run_simulator()

        simulated_data_chunk = self.chunk(
            start=simulated_data['time'][0],
            end=simulated_data['endtime'][-1],
            data=simulated_data,
            data_type='events_tpc')

        # write empty chunk if nveto data isn't there	        return {'events_tpc':simulated_data_chunk}
        if simulated_data_nveto is None or len(simulated_data_nveto['time']) < 1:
            simulated_data_nveto_chunk = self.chunk(
                start=0,
                end=1,
                data=simulated_data_nveto,
                data_type='events_nveto')
        else:
            simulated_data_nveto_chunk = self.chunk(
                start=np.floor(simulated_data_nveto['time'][0]).astype(np.int64),
                end=np.ceil(np.max(simulated_data_nveto['endtime'])).astype(np.int64),
                data=simulated_data_nveto,
                data_type='events_nveto')

        return {'events_tpc': simulated_data_chunk,
                'events_nveto': simulated_data_nveto_chunk}
    
    def is_ready(self, chunk):
        # For this plugin we'll smash everything into 1 chunk, should be oke
        return True if chunk == 0 else False

    def source_finished(self):
        return True
