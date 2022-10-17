import strax
import straxen
from straxen import InterpolatingMap, get_resource
from wfsim.load_resource import load_config
import epix
import numpy as np
import pandas as pd
from copy import deepcopy
from immutabledict import immutabledict
from .GenerateEvents import GenerateEvents
from .GenerateNveto import NVetoUtils
from .helpers import Helpers
import warnings


class Simulator():
    '''Simulator class for epix to go from  epix instructions to fully processed data'''

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
        
        First do the macro clustering. Clustered instructions will be flagged with amp=-1
        so we can safely through those out"""

        # Helpers.macro_cluster_events(self.instructions_epix)
        # self.instructions_epix = self.instructions_epix[self.instructions_epix['amp'] != -1]

        Helpers.macro_cluster_events(self.instructions_epix)
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
            
            i['s1_area'] = np.sum(inst_s1['amp']) # sum over all s1 10/12/2022

            # why we consider only the larger s1 ?
            #i['s1_area'] = inst_s1[s1[-1]]['amp']
            #if len(s1) > 1:
                # I do not think it is correct
            #    i['alt_s1_area'] = inst_s1[s1[-2]]['amp']

            i['s2_area'] = np.sum(inst_s2[s2[-1]]['amp'])
            if len(s2) > 1:
                i['alt_s2_area'] = inst_s2[s2[-2]]['amp']
                i['alt_s2_x'] = inst_s2[s2[-2]]['x']
                i['alt_s2_y'] = inst_s2[s2[-2]]['y']
                i['alt_s2_z'] = inst_s2[s2[-2]]['z']

            i['x'] = inst_s2[s2[-1]]['x']
            i['y'] = inst_s2[s2[-1]]['y']
            i['z'] = inst_s2[s2[-1]]['z']

            # Strax wants begin and endtimes
            i['time'] = ix * 1000
            i['endtime'] = ix * 1000 + 1

        instructions = instructions[~((instructions['time'] == 0) & (instructions['endtime'] == 0))]
        self.instructions = instructions

    def simulate(self, ):
        for func in self.simulation_functions:
            func(instructions=self.instructions,
                 config=self.config,
                 resource=self.resource)

    def run_simulator(self, ):
        self.cluster_events()
        self.simulate()

        return self.instructions


# We should think a bit more to detector_config_override
# and tell fast_sim to look into epix_args
# also beacaus one entry of self.config is epix_config
@strax.takes_config(
    strax.Option('detector', default='XENONnT', help='Detector model'),
    strax.Option('g4_file', help='G4 file to simulate'),
    strax.Option('epix_config', default=dict(), help='Configuration file for epix', ),
    strax.Option('configuration_files', default=dict(), help='Files required for simulating'),
    strax.Option('fax_config', help='Fax configuration to load'),
    strax.Option('fax_config_overrides', help='Fax configuration to override', default=None),
    strax.Option('xy_resolution', default=5, help='xy position resolution (cm)'),
    strax.Option('z_resolution', default=1, help='xy position resolution (cm)'),
    strax.Option('nv_spe_resolution', default=0.40, help='nVeto SPE resolution'),
    strax.Option('nv_spe_res_threshold', default=0.50, help='nVeto SPE acceptance threshold'),
    strax.Option('nv_max_time_ns', default=1e7, help='nVeto maximum time for the acceptance of PMT hits in event'),
)
class StraxSimulator(strax.Plugin):
    provides = ('events_tpc', 'events_nveto')
    depends_on = ()
    data_kind = immutabledict(zip(provides, provides))
    dtype = dict(events_tpc=[('time', np.int64),
                             ('endtime', np.int64),
                             ('s1_area', np.float),
                             ('s2_area', np.float),
                             ('cs1', np.float),
                             ('cs2', np.float),
                             ('alt_s1_area', np.float),
                             ('alt_s2_area', np.float),
                             ('alt_cs1', np.float),
                             ('alt_cs2', np.float),
                             ('x', np.float),
                             ('y', np.float),
                             ('z', np.float),
                             ('alt_s2_x', np.float),
                             ('alt_s2_y', np.float),
                             ('alt_s2_z', np.float),
                             ('drift_time', np.float),
                             ('alt_s2_drift_time', np.float)],
                 events_nveto=[('time', np.float),
                               ('endtime', np.float),
                               ('event_id', np.int),
                               ('channel', np.int), ])

    def load_config(self):
        """First load the config through wfsim, then we add some things we'd like to have"""
        self.resource = load_config(self.config)
        
        self.resource.s1_map = self.resource.s1_lce_correction_map
        self.resource.s2_map = self.resource.s2_correction_map

        # Terrible way to handle the config
        # we should have only one file 
        # instead of several config
        if 'nv_pmt_qe' in self.config['configuration_files'].keys():
                    self.resource.nv_pmt_qe = straxen.get_resource(
                        self.config['configuration_files']['nv_pmt_qe'], fmt='json')
        else:
            warnings.warn('The configuration_files should not exist!'
                          'Everything should come by one config!'
                          'Since nv_pmt_qe config is missing in configuration_files, '
                          'the default one nveto_pmt_qe.json will be used')
            self.resource.nv_pmt_qe = straxen.get_resource('nveto_pmt_qe.json', fmt='json')

        if 'photon_area_distribution' in self.config['configuration_files'].keys():
                self.resource.photon_area_distribution = epix.Helpers.average_spe_distribution(
                                get_resource(self.config['configuration_files']['photon_area_distribution'], 
                                fmt='csv'))
        else:
            warnings.warn('The configuration_files should not exist!'
                          'Everything should come by one config!'
                          'Since photon_area_distribution config is missing in configuration_files, '
                          'the one in fax_config will be used')
            self.resource.photon_area_distribution = epix.Helpers.average_spe_distribution(
                                get_resource(self.config['photon_area_distribution'], fmt='csv'))

    def setup(self, ):
        overrides = self.config['fax_config_overrides']
        self.config.update(straxen.get_resource(self.config['fax_config'], fmt='json'))
        if overrides is not None:
            self.config.update(overrides)

        if 'gains' not in self.config.keys():
            warnings.warn('Why we have to provide gains here? '
                          'No gains are passed in fax_config_overrides, default equal to 1')
            self.config['gains'] =  [1 for i in range(494)]
        if 'n_top_pmts' not in self.config.keys():
            warnings.warn('This is a deault value, why we have to give it in fax_config_overrides? '
                          'No n_top_pmts are passed in fax_config_overrides, default equal to 253')
            self.config['n_top_pmts'] =  253
        if 'n_tpc_pmts' not in self.config.keys():
            warnings.warn('This is a deault value, why we have to give it in fax_config_overrides? '
                          'No n_tpc_pmts are passed in fax_config_overrides, default equal to 494')
            self.config['n_tpc_pmts'] =  494

        # Is this useful ?
        if self.config['epix_config']['debug']:
            print('\nConfiguraton: \n',  self.config, '\n')     

        self.load_config()

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
        warnings.warn('The epix_config is subtle!'
                       'detector_config_override is given by epix_config, '
                       'while detector is from config '
                       'I do not like it!')
        detector = epix.init_detector(self.config['detector'].lower(), 
                        self.config['epix_config']['detector_config_override'])

        epix_config = deepcopy(self.config['epix_config'])
        epix_config['detector_config'] = detector

        outer_cylinder = getattr(epix.detectors, 'xenonnt')
        _, outer_cylinder = outer_cylinder()
        epix_config['outer_cylinder'] = outer_cylinder

        epix_ins = epix.run_epix.main(epix_config, return_wfsim_instructions=True)
        print('Saving epix instruction: ', self.config['epix_config']['save_epix'])
        warnings.warn('The epix_config change here!'
                       'Is this fine?')
        if self.config['epix_config']['save_epix']:
            epix_path = self.config['epix_config']['path'] + self.config['epix_config']['file_name'][:-5] +'_epix'
            np.save(epix_path, epix_ins)

        return epix_ins

    def compute(self):
        simulated_data_nveto = self.get_nveto_data()
        epix_instructions = self.get_epix_instructions()
        self.Simulator = Simulator(instructions_epix=epix_instructions,
                                   config=self.config,
                                   resource=self.resource)
        simulated_data = self.Simulator.run_simulator()

        simulated_data_chunk = self.chunk(
            start=simulated_data['time'][0],
            end=simulated_data['endtime'][-1],
            data=simulated_data,
            data_type='events_tpc')

        # write empty chunk if nveto data isn't there
        if simulated_data_nveto is None or len(simulated_data_nveto['time']) < 1:
            simulated_data_nveto_chunk = self.chunk(
                start=0,
                end=1,
                data=simulated_data_nveto,
                data_type='events_nveto')
        else:
            simulated_data_nveto_chunk=self.chunk(
                           start=np.floor(simulated_data_nveto['time'][0]).astype(np.int64),
                           end=np.ceil(np.max(simulated_data_nveto['endtime'])).astype(np.int64),
                           data=simulated_data_nveto,
                           data_type='events_nveto')

        return {'events_tpc':simulated_data_chunk,
                'events_nveto':simulated_data_nveto_chunk}
    
    def is_ready(self, chunk):
        # For this plugin we'll smash everything into 1 chunk, should be oke
        return True if chunk == 0 else False

    def source_finished(self):
        return True
